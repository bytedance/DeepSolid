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

import chex
import jax
import jax.numpy as jnp
import functools

from DeepSolid import hamiltonian
from DeepSolid import constants
from DeepSolid.utils.kfac_ferminet_alpha import loss_functions


@chex.dataclass
class AuxiliaryLossData:
    variance: jnp.DeviceArray
    local_energy: jnp.DeviceArray
    imaginary: jnp.DeviceArray
    kinetic: jnp.DeviceArray
    ewald: jnp.DeviceArray


def make_loss(network, batch_network,
              simulation_cell,
              clip_local_energy=5.0,
              clip_type='real',
              mode='for',
              partition_number=3):
    """
    generates loss function used for wavefunction trains.
    :param network: unbatched logdet function of wavefunction
    :param batch_network: batched logdet function of wavefunction
    :param simulation_cell: pyscf object of simulation cell.
    :param clip_local_energy: clip window width of local energy.
    :param clip_type: specify the clip style. real mode clips the local energy in Cartesion style,
    and complex mode in polar style
    :param mode: specify the evaluation style of local energy.
    'for' mode calculates the laplacian of each electron one by one, which is slow but save GPU memory
    'hessian' mode calculates the laplacian in a highly parallized mode, which is fast but require GPU memory
    'partition' mode calculate the laplacian in a moderate way.
    :param partition_number: Only used if 'partition' mode is employed.
    partition_number must be divisivle by (dim * number of electrons).
    The smaller the faster, but requires more memory.
    :return: the loss function
    """
    el_fun = hamiltonian.local_energy_seperate(network,
                                               simulation_cell=simulation_cell,
                                               mode=mode,
                                               partition_number=partition_number)
    batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(params, data):
        """

        :param params: a dictionary of parameters
        :param data: batch electron coord with shape [Batch, Nelec * Ndim]
        :return: energy expectation of corresponding walkers (only take real part) with shape [Batch]
        """
        ke, ew = batch_local_energy(params, data)
        e_l = ke + ew
        mean_e_l = jnp.mean(e_l)

        pmean_loss = constants.pmean_if_pmap(mean_e_l, axis_name=constants.PMAP_AXIS_NAME)
        variance = constants.pmean_if_pmap(jnp.mean(jnp.abs(e_l)**2) - jnp.abs(mean_e_l.real) ** 2,
                                           axis_name=constants.PMAP_AXIS_NAME)
        loss = pmean_loss.real
        imaginary = pmean_loss.imag

        return loss, AuxiliaryLossData(variance=variance,
                                       local_energy=e_l,
                                       imaginary=imaginary,
                                       kinetic=ke,
                                       ewald=ew,
                                       )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        """
        customised jvp function of loss function.
        :param primals: inputs of total_energy function (params, data)
        :param tangents: tangent vectors corresponding to the primal (params, data)
        :return: Jacobian-vector product of total energy.
        """
        params, data = primals
        loss, aux_data = total_energy(params, data)
        diff = (aux_data.local_energy - loss)
        if clip_local_energy > 0.0:
            if clip_type == 'complex':
                radius, phase = jnp.abs(diff), jnp.angle(diff)
                radius_tv = constants.pmean_if_pmap(radius.std(), axis_name=constants.PMAP_AXIS_NAME)
                radius_mean = jnp.median(radius)
                radius_mean = constants.pmean_if_pmap(radius_mean, axis_name=constants.PMAP_AXIS_NAME)
                clip_radius = jnp.clip(radius,
                                       radius_mean - radius_tv * clip_local_energy,
                                       radius_mean + radius_tv * clip_local_energy)
                clip_diff = clip_radius * jnp.exp(1j * phase)
            elif clip_type == 'real':
                tv_re = jnp.mean(jnp.abs(diff.real))
                tv_re = constants.pmean_if_pmap(tv_re, axis_name=constants.PMAP_AXIS_NAME)
                tv_im = jnp.mean(jnp.abs(diff.imag))
                tv_im = constants.pmean_if_pmap(tv_im, axis_name=constants.PMAP_AXIS_NAME)
                clip_diff_re = jnp.clip(diff.real,
                                        -clip_local_energy * tv_re,
                                        clip_local_energy * tv_re)
                clip_diff_im = jnp.clip(diff.imag,
                                        -clip_local_energy * tv_im,
                                        clip_local_energy * tv_im)
                clip_diff = clip_diff_re + clip_diff_im * 1j
            else:
                raise ValueError('Unrecognized clip type.')
        else:
            clip_diff = diff

        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
        conj_psi_tangent = jnp.conjugate(psi_tangent)
        conj_psi_primal = jnp.conjugate(psi_primal)

        loss_functions.register_normal_predictive_distribution(conj_psi_primal[:, None])

        primals_out = loss, aux_data
        # tangents_dot = jnp.dot(clip_diff, conj_psi_tangent).real
        # dot causes the gradient to be extensive with batch size, which does matter for KFAC.
        tangents_dot = jnp.mean((clip_diff * conj_psi_tangent).real)

        tangents_out = (tangents_dot, aux_data)

        return primals_out, tangents_out

    return total_energy


def make_training_step(mcmc_step, val_and_grad, opt_update):
    """
    generates the function used for wavefunction train.
    :param mcmc_step: MCMC sample function
    :param val_and_grad: value and grad of loss function
    :param opt_update: optimizer state
    :return: one iteration train function
    """

    @functools.partial(constants.pmap, donate_argnums=(1, 2, 3, 4))
    def step(t, data, params, state, key, mcmc_width):
        """
        one iteration train of energy
        :param t: current step of energy train.
        :param data: batched electron walkes with shape [batch, ne * ndim]
        :param params: a dictionary of parameters.
        :param state: optimizer state.
        :param key: PRNG key.
        :param mcmc_width: mcmc move width
        :return:
        data: moved electron walkers
        params: updated paramters
        state: updated optimizer state
        loss: value of loss function
        aux_data: auxiliary value of loss function
        pmove: accept rate of MCMC move
        search_direction: calculated gradient.
        """
        data, pmove = mcmc_step(params, data, key, mcmc_width)

        # Optimization step
        (loss, aux_data), search_direction = val_and_grad(params, data)
        search_direction = constants.pmean_if_pmap(search_direction,
                                                   axis_name=constants.PMAP_AXIS_NAME)
        state, params = opt_update(t, search_direction, params, state)
        return data, params, state, loss, aux_data, pmove, search_direction

    return step


@functools.partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def direct_product(x, y):
    return x.ravel()[:, None] * y.ravel()[None, :]


def make_sr_matrix(network):
    '''
    which is used to calculate the fisher matrix, abandoned now.
    :param network:
    :return:
    '''
    network_grad = jax.grad(network.apply, argnums=0, holomorphic=True)
    batch_network_grad = jax.vmap(network_grad, in_axes=(None, 0))

    def sr_matrix(params, data):
        complex_params = jax.tree_map(lambda x: x+0j, params)
        batch_diffs = batch_network_grad(complex_params, data)

        s1 = jax.tree_map(lambda x: jnp.mean(direct_product(jnp.conjugate(x), x),
                                             axis=0),
                          batch_diffs)
        s2 = jax.tree_map(lambda x: (jnp.mean(jnp.conjugate(x), axis=0).ravel()[:, None] *
                                     jnp.mean(x, axis=0).ravel()[None, :]
                                     ),
                          batch_diffs)
        s1 = constants.pmean_if_pmap(s1, axis_name=constants.PMAP_AXIS_NAME)
        s2 = constants.pmean_if_pmap(s2, axis_name=constants.PMAP_AXIS_NAME)
        matrix = jax.tree_multimap(lambda x, y: x - y, s1, s2)
        return matrix

    return sr_matrix





