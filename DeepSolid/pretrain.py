# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

import functools

import numpy as np
from absl import logging
import jax
import jax.numpy as jnp
import optax

from DeepSolid import hf
from DeepSolid import qmc
from DeepSolid import constants


def _batch_slater_slogdet(scf: hf.SCF, dim=3):

    def batch_slater_slogdet(params, x):
        del params
        batch = x.shape[0]
        x = x.reshape([batch, -1, dim])
        result = scf.eval_slogdet(x)[1]
        return result

    return batch_slater_slogdet


def make_pretrain_step(batch_orbitals,
                       batch_network,
                       latvec,
                       optimizer,
                       full_det=False,
                       ):
    """
    generate the low-level pretrain function
    :param batch_orbitals: batched function return the orbital matrix of wavefunction
    :param batch_network: batched function return the slogdet of wavefunction
    :param latvec: lattice vector of primitive cell
    :param optimizer: optimizer function
    :return: the low-level pretrain function
    """

    def pretrain_step(data, target, params, state, key):
        """
        One iteration of pretraining to match HF.
        :param data: batched input data, a [batch, 3N] dimensional vector.
        :param target: corresponding HF matrix values.
        :param params: A dictionary of parameters.
        :param state: optimizer state.
        :param key: PRNG key.
        :return: pretrained params, data, state, loss value, slogdet of neural network,
        and number of accepted MCMC moves.
        """

        def loss_fn(x, p, target):
            """
            loss function
            :param x: batched input data, a [batch, 3N] dimensional vector.
            :param p: A dictionary of parameters.
            :param target: corresponding HF matrix values.
            :return: value of loss function
            """
            predict = batch_orbitals(p, x)
            if full_det:
                batch_size = predict[0].shape[0]
                na = target[0].shape[1]
                nb = target[1].shape[1]
                target = [jnp.concatenate(
                    (jnp.concatenate((target[0], jnp.zeros((batch_size, na, nb))), axis=-1),
                     jnp.concatenate((jnp.zeros((batch_size, nb, na)), target[1]), axis=-1)),
                    axis=-2)]
            result = jnp.array([jnp.mean(jnp.abs(tar[:, None, ...] - pre)**2)
                                for tar, pre in zip(target, predict)]).mean()
            return constants.pmean_if_pmap(result, axis_name=constants.PMAP_AXIS_NAME)

        val_and_grad = jax.value_and_grad(loss_fn, argnums=1)
        loss_val, search_direction = val_and_grad(data, params, target)
        search_direction = constants.pmean_if_pmap(
            search_direction, axis_name=constants.PMAP_AXIS_NAME)
        updates, state = optimizer.update(search_direction, state, params)
        params = optax.apply_updates(params, updates)
        logprob = 2 * batch_network(params, data)
        data, key, logprob, num_accepts = qmc.mh_update(params=params,
                                                        f=batch_network,
                                                        x1=data,
                                                        key=key,
                                                        lp_1=logprob,
                                                        num_accepts=0,
                                                        latvec=latvec)
        return data, params, state, loss_val, logprob, num_accepts

    return pretrain_step


def pretrain_hartree_fock(params,
                          data,
                          batch_network,
                          batch_orbitals,
                          sharded_key,
                          cell,
                          scf_approx: hf.SCF,
                          full_det=False,
                          iterations=1000,
                          learning_rate=5e-3,
                          ):
    """
    generates a function used for pretrain, and neural network is used as the target sample.
    :param params: A dictionary of parameters.
    :param data: The input data, a 3N dimensional vector.
    :param batch_network: batched function return the slogdet of wavefunction
    :param batch_orbitals: batched function return the orbital matrix of wavefunction
    :param sharded_key: PRNG key
    :param cell: pyscf object of simulation cell
    :param scf_approx: hf.SCF object in DeepSolid. Used to eval the orbital value of Hartree Fock ansatz.
    :param full_det: If true, the determinants are dense, rather than block-sparse.
     True by default, false is still available for backward compatibility.
     Thus, the output shape of the orbitals will be (ndet, nalpha+nbeta,
     nalpha+nbeta) if True, and (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)
     if False.
    :param iterations: pretrain iterations
    :param learning_rate: learning rate of pretrain
    :return: pretrained parameters and electron positions.
    """

    optimizer = optax.adam(learning_rate)
    opt_state_pt = constants.pmap(optimizer.init)(params)
    leading_shape = data.shape[:-1]

    pretrain_step = make_pretrain_step(batch_orbitals=batch_orbitals,
                                       batch_network=batch_network,
                                       latvec=cell.lattice_vectors(),
                                       optimizer=optimizer,
                                       full_det=full_det,)
    pretrain_step = constants.pmap(pretrain_step)

    for t in range(iterations):
        target = scf_approx.eval_orb_mat(np.array(data.reshape([-1, cell.nelectron, 3]), dtype=np.float64))
        # PYSCF PBC eval_gto seems only accept float64 array, float32 array will easily cause nan or underflow.
        target = [jnp.array(tar) for tar in target]
        target = [tar.reshape([*leading_shape, ne, ne]) for tar, ne in zip(target, cell.nelec) if ne > 0]

        slogprob_target = [2 * jnp.linalg.slogdet(tar)[1] for tar in target]
        slogprob_target = functools.reduce(lambda x, y: x+y, slogprob_target)
        sharded_key, subkeys = constants.p_split(sharded_key)
        data, params, opt_state_pt, loss, logprob, num_accepts = pretrain_step(
            data, target, params, opt_state_pt, subkeys)
        logging.info('Pretrain iter %05d: Loss=%03.6f, pmove=%0.2f, '
                     'Norm of Net prob=%03.4f, Norm of HF prob=%03.4f',
                     t, loss[0],
                     jnp.mean(num_accepts) / leading_shape[-1],
                     jnp.mean(logprob),
                     jnp.mean(slogprob_target))

    return params, data


def pretrain_hartree_fock_usingHF(params,
                                  data,
                                  batch_orbitals,
                                  sharded_key,
                                  cell,
                                  scf_approx: hf.SCF,
                                  iterations=1000,
                                  learning_rate=5e-3,
                                  nsteps=1,
                                  full_det=False,
                                  ):
    """
    generates a function used for pretrain, and HF ansatz is used as the target sample.
    :param params: A dictionary of parameters.
    :param data: The input data, a 3N dimensional vector.
    :param batch_network: batched function return the slogdet of wavefunction
    :param batch_orbitals: batched function return the orbital matrix of wavefunction
    :param sharded_key: PRNG key
    :param cell: pyscf object of simulation cell
    :param scf_approx: hf.SCF object in DeepSolid. Used to eval the orbital value of Hartree Fock ansatz.
    :param full_det: If true, the determinants are dense, rather than block-sparse.
     True by default, false is still available for backward compatibility.
     Thus, the output shape of the orbitals will be (ndet, nalpha+nbeta,
     nalpha+nbeta) if True, and (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)
     if False.
    :param iterations: pretrain iterations
    :param learning_rate: learning rate of pretrain
    :return: pretrained parameters and electron positions.
    """

    optimizer = optax.adam(learning_rate)
    opt_state_pt = constants.pmap(optimizer.init)(params)
    leading_shape = data.shape[:-1]

    def make_pretrain_step(batch_orbitals,
                           latvec,
                           optimizer,
                           ):
        """
        generate the low-level pretrain function
        :param batch_orbitals: batched function return the orbital matrix of wavefunction
        :param latvec: lattice vector of primitive cell
        :param optimizer: optimizer function
        :return: the low-level pretrain function
        """

        def pretrain_step(data, target, params, state):
            """
            One iteration of pretraining to match HF.
            :param data: batched input data, a [batch, 3N] dimensional vector.
            :param target: corresponding HF matrix values.
            :param params: A dictionary of parameters.
            :param state: optimizer state.
            :return: pretrained params, data, state, loss value.
            """

            def loss_fn(x, p, target):
                """
                loss function
                :param x: batched input data, a [batch, 3N] dimensional vector.
                :param p: A dictionary of parameters.
                :param target: corresponding HF matrix values.
                :return: value of loss function
                """
                predict = batch_orbitals(p, x)
                if full_det:
                    batch_size = predict[0].shape[0]
                    na = target[0].shape[1]
                    nb = target[1].shape[1]
                    target = [jnp.concatenate(
                        (jnp.concatenate((target[0], jnp.zeros((batch_size, na, nb))), axis=-1),
                         jnp.concatenate((jnp.zeros((batch_size, nb, na)), target[1]), axis=-1)),
                        axis=-2)]
                result = jnp.array([jnp.mean(jnp.abs(tar[:, None, ...] - pre) ** 2)
                                    for tar, pre in zip(target, predict)]).mean()
                return constants.pmean_if_pmap(result, axis_name=constants.PMAP_AXIS_NAME)

            val_and_grad = jax.value_and_grad(loss_fn, argnums=1)
            loss_val, search_direction = val_and_grad(data, params, target)
            search_direction = constants.pmean_if_pmap(
                search_direction, axis_name=constants.PMAP_AXIS_NAME)
            updates, state = optimizer.update(search_direction, state, params)
            params = optax.apply_updates(params, updates)

            return params, state, loss_val

        return pretrain_step


    pretrain_step = make_pretrain_step(batch_orbitals=batch_orbitals,
                                       latvec=cell.lattice_vectors(),
                                       optimizer=optimizer,)
    pretrain_step = constants.pmap(pretrain_step)
    batch_network = _batch_slater_slogdet(scf_approx)
    logprob = 2 * batch_network(None, data.reshape([-1, cell.nelectron * 3]))

    def step_fn(inputs):
        return qmc.mh_update(params,
                             batch_network,
                             *inputs,
                             latvec=cell.lattice_vectors(),
                             )

    for t in range(iterations):

        for _ in range(nsteps):
            sharded_key, subkeys = constants.p_split(sharded_key)
            inputs = (data.reshape([-1, cell.nelectron * 3]),
                      sharded_key[0],
                      logprob,
                      0.)
            data, _,  logprob, num_accepts = step_fn(inputs)

        data = data.reshape([*leading_shape, -1])
        target = scf_approx.eval_orb_mat(data.reshape([-1, cell.nelectron, 3]))
        target = [tar.reshape([*leading_shape, ne, ne]) for tar, ne in zip(target, cell.nelec) if ne > 0]

        slogprob_net = [2 * jnp.linalg.slogdet(net_mat)[1] for net_mat in constants.pmap(batch_orbitals)(params, data)]
        slogprob_net = functools.reduce(lambda x, y: x+y, slogprob_net)

        sharded_key, subkeys = constants.p_split(sharded_key)
        params, opt_state_pt, loss = pretrain_step(data, target, params, opt_state_pt)

        logging.info('Pretrain iter %05d: Loss=%03.6f, pmove=%0.2f, '
                     'Norm of Net prob=%03.4f, Norm of HF prob=%03.4f',
                     t, loss[0],
                     jnp.mean(num_accepts) / functools.reduce(lambda x, y: x*y, leading_shape),
                     jnp.mean(slogprob_net),
                     jnp.mean(logprob))

    return params, data
