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

import jax
import jax.numpy as jnp
import numpy as np
import datetime
import ml_collections
import logging
import time
import optax
import chex
import pandas as pd

from DeepSolid.utils.kfac_ferminet_alpha import optimizer as kfac_optim
from DeepSolid.utils.kfac_ferminet_alpha import utils as kfac_utils

from DeepSolid import constants
from DeepSolid import network
from DeepSolid import train
from DeepSolid import pretrain
from DeepSolid import qmc
from DeepSolid import init_guess
from DeepSolid import hf
from DeepSolid import checkpoint
from DeepSolid.utils import writers
from DeepSolid import estimator


def get_params_initialization_key(deterministic):
    '''
    The key point here is to make sure different hosts uses the same RNG key
    to initialize network parameters.
    '''
    if deterministic:
        seed = 888
    else:

        # The overly complicated action here is to make sure different hosts get
        # the same seed.
        @constants.pmap
        def average_seed(seed_array):
            return jax.lax.pmean(jnp.mean(seed_array), axis_name=constants.PMAP_AXIS_NAME)

        local_seed = time.time()
        float_seed = average_seed(jnp.ones(jax.local_device_count()) * local_seed)[0]
        seed = int(1e6 * float_seed)
    print(f'params initialization seed: {seed}')
    return jax.random.PRNGKey(seed)


def process(cfg: ml_collections.ConfigDict):

    num_hosts, host_idx = 1, 0

    # Device logging
    num_devices = jax.local_device_count()
    local_batch_size = cfg.batch_size // num_hosts
    logging.info('Starting QMC with %i XLA devices', num_devices)
    if local_batch_size % num_devices != 0:
        raise ValueError('Batch size must be divisible by number of devices, '
                         'got batch size {} for {} devices.'.format(
            local_batch_size, num_devices))
    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path,)
    ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)
    ckpt_restore_filename = (
            checkpoint.find_last_checkpoint(ckpt_save_path) or
            checkpoint.find_last_checkpoint(ckpt_restore_path))

    simulation_cell = cfg.system.pyscf_cell
    cfg.system.internal_cell = init_guess.pyscf_to_cell(cell=simulation_cell)

    hartree_fock = hf.SCF(cell=simulation_cell, twist=jnp.array(cfg.network.twist))
    hartree_fock.init_scf()


    if cfg.system.ndim != 3:
        # The network (at least the input feature construction) and initial MCMC
        # molecule configuration (via system.Atom) assume 3D systems. This can be
        # lifted with a little work.
        raise ValueError('Only 3D systems are currently supported.')
    data_shape = (num_devices, local_batch_size // num_devices)

    if cfg.debug.deterministic:
        seed = 666
    else:
        seed = int(1e6 * time.time())

    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, host_idx)

    system_dict = {
        'klist': hartree_fock.klist,
        'simulation_cell': simulation_cell,
    }
    system_dict.update(cfg.network.detnet)

    slater_mat = network.make_solid_fermi_net(**system_dict, method_name='eval_mats')
    slater_logdet = network.make_solid_fermi_net(**system_dict, method_name='eval_logdet')
    slater_slogdet = network.make_solid_fermi_net(**system_dict, method_name='eval_slogdet')

    batch_slater_logdet = jax.vmap(slater_logdet.apply, in_axes=(None, 0), out_axes=0)
    batch_slater_slogdet = jax.vmap(slater_slogdet.apply, in_axes=(None, 0), out_axes=0)
    batch_slater_mat = jax.vmap(slater_mat.apply, in_axes=(None, 0), out_axes=0)

    if ckpt_restore_filename:
        t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
            ckpt_restore_filename, local_batch_size)

    else:
        logging.info('No checkpoint found. Training new model.')
        t_init = 0
        opt_state_ckpt = None
        mcmc_width_ckpt = None
        data = init_guess.init_electrons(key=key, cell=cfg.system.internal_cell,
                                         latvec=simulation_cell.lattice_vectors(),
                                         electrons=simulation_cell.nelec,
                                         batch_size=local_batch_size,
                                         init_width=cfg.mcmc.init_width)
        data = jnp.reshape(data, data_shape + data.shape[1:])
        data = constants.broadcast_all_local_devices(data)
        params_initialization_key = get_params_initialization_key(cfg.debug.deterministic)
        params = slater_logdet.init(key=params_initialization_key, data=None)
        params = constants.replicate_all_local_devices(params)

    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    shared_t = constants.replicate_all_local_devices(jnp.zeros([]))
    shared_mom = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_utils.replicate_all_local_devices(
        jnp.asarray(cfg.optim.kfac.damping))
    sharded_key = constants.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = constants.p_split(sharded_key)

    if (t_init == 0 and cfg.pretrain.method == 'net' and
            cfg.pretrain.iterations > 0):
        logging.info('Pretrain using Net distribution.')
        sharded_key, subkeys = constants.p_split(sharded_key)
        params, data = pretrain.pretrain_hartree_fock(params=params,
                                                      data=data,
                                                      batch_network=batch_slater_slogdet,
                                                      batch_orbitals=batch_slater_mat,
                                                      sharded_key=subkeys,
                                                      scf_approx=hartree_fock,
                                                      cell=simulation_cell,
                                                      iterations=cfg.pretrain.iterations,
                                                      learning_rate=cfg.pretrain.lr,
                                                      full_det=cfg.network.detnet.full_det,
                                                      )

    if (t_init == 0 and cfg.pretrain.method == 'hf' and
            cfg.pretrain.iterations > 0):
        logging.info('Pretrain using Hartree Fock distribution.')
        sharded_key, subkeys = constants.p_split(sharded_key)
        params, data = pretrain.pretrain_hartree_fock_usingHF(params=params,
                                                              data=data,
                                                              batch_orbitals=batch_slater_mat,
                                                              sharded_key=sharded_key,
                                                              cell=simulation_cell,
                                                              scf_approx=hartree_fock,
                                                              iterations=cfg.pretrain.iterations,
                                                              learning_rate=cfg.pretrain.lr,
                                                              full_det=cfg.network.detnet.full_det,
                                                              nsteps=cfg.pretrain.steps)
    if (t_init == 0 and cfg.pretrain.iterations > 0):
        logging.info('Saving pretrain params')
        checkpoint.save(ckpt_save_path, 0, data, params, None, None,)

    sampling_func = slater_slogdet.apply if cfg.mcmc.importance_sampling else None
    mcmc_step = qmc.make_mcmc_step(batch_slog_network=batch_slater_slogdet,
                                   batch_per_device=local_batch_size//jax.local_device_count(),
                                   latvec=jnp.asarray(simulation_cell.lattice_vectors()),
                                   steps=cfg.mcmc.steps,
                                   one_electron_moves=cfg.mcmc.one_electron,
                                   importance_sampling=sampling_func,
                                   )

    total_energy = train.make_loss(network=slater_logdet.apply,
                                   batch_network=batch_slater_logdet,
                                   simulation_cell=simulation_cell,
                                   clip_local_energy=cfg.optim.clip_el,
                                   clip_type=cfg.optim.clip_type,
                                   mode=cfg.optim.laplacian_mode,
                                   partition_number=cfg.optim.partition_number,
                                   )

    def learning_rate_schedule(t):
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t / cfg.optim.lr.delay))), cfg.optim.lr.decay)

    val_and_grad = jax.value_and_grad(total_energy, argnums=0, has_aux=True)
    if cfg.optim.optimizer == 'adam':
        optimizer = optax.chain(optax.scale_by_adam(**cfg.optim.adam),
                                optax.scale_by_schedule(learning_rate_schedule),
                                optax.scale(-1.))
    elif cfg.optim.optimizer == 'kfac':
        optimizer = kfac_optim.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=True,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode='fisher_exact',
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME
            # debug=True
        )
        sharded_key, subkeys = kfac_utils.p_split(sharded_key)
        opt_state = optimizer.init(params, subkeys, data)
        opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    elif cfg.optim.optimizer == 'none':
        total_energy = constants.pmap(total_energy)
        opt_state = None
    else:
        raise ValueError('Unrecognized Optimizer.')

    if cfg.optim.optimizer != 'kfac' and cfg.optim.optimizer != 'none':
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=cfg.optim.ministeps)

        opt_state = jax.pmap(optimizer.init)(params)
        opt_state = opt_state if opt_state_ckpt is None else optax._src.wrappers.MultiStepsState(*opt_state)

        def opt_update(t, grad, params, opt_state):
            del t  # Unused.
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return opt_state, params

        step = train.make_training_step(mcmc_step, val_and_grad, opt_update)

    mcmc_step = constants.pmap(mcmc_step)

    if mcmc_width_ckpt is not None:
        mcmc_width = constants.broadcast_all_local_devices(jnp.asarray(mcmc_width_ckpt))
    else:
        mcmc_width = constants.replicate_all_local_devices(jnp.asarray(cfg.mcmc.move_width))

    if t_init == 0:
        logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
        for t in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = constants.p_split(sharded_key)
            data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
        logging.info('Completed burn-in MCMC steps')
        logging.info('Initial energy for primitive cell: %03.4f E_h',
                     constants.pmap(total_energy)(params, data)[0][0] / simulation_cell.scale)

    time_of_last_ckpt = time.time()

    if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
        # If opt_state_ckpt is None, then we're restarting from a previous inference
        # run (most likely due to preemption) and so should continue from the last
        # iteration in the checkpoint. Otherwise, starting an inference run from a
        # training run.
        logging.info('No optimizer provided. Assuming inference run.')
        logging.info('Setting initial iteration to 0.')
        t_init = 0

    train_schema = ['step', 'energy', 'variance', 'pmove', 'imaginary', 'kinetic', 'ewald']
    if cfg.log.complex_polarization:
        train_schema.append('complex_polarization')
        polarization = estimator.make_complex_polarization(simulation_cell)
        pmap_polarization = constants.pmap(polarization)
    if cfg.log.structure_factor:
        structure_factor = estimator.make_structure_factor(simulation_cell)
        pmap_structure_factor = constants.pmap(structure_factor)
    with writers.Writer(name=cfg.log.stats_file_name,
                        schema=train_schema,
                        directory=ckpt_save_path,
                        iteration_key=None,
                        log=False) as writer:
        for t in range(t_init, cfg.optim.iterations):
            sharded_key, subkeys = constants.p_split(sharded_key)
            if cfg.optim.optimizer == 'kfac':
                new_data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
                # Need this split because MCMC step above used subkeys already
                sharded_key, subkeys = kfac_utils.p_split(sharded_key)
                new_params, new_opt_state, new_stats = optimizer.step(  # pytype: disable=attribute-error
                    params=params,
                    state=opt_state,
                    rng=subkeys,
                    data_iterator=iter([new_data]),
                    momentum=shared_mom,
                    damping=shared_damping)
                tree = {'params': new_params, 'loss': new_stats['loss'], 'optim': new_opt_state}
                try:
                    # We don't do check_nan by default due to efficiency concern.
                    # We noticed ~0.2s overhead when performing this nan check
                    # at transitional medals.
                    if cfg.debug.check_nan:
                        chex.assert_tree_all_finite(tree)
                    data = new_data
                    params = new_params
                    opt_state = new_opt_state
                    stats = new_stats
                    loss = stats['loss']
                    aux_data = stats['aux']
                except AssertionError as e:
                    # data, params, opt_state, and stats are not updated
                    logging.warn(str(e))
                    loss = aux_data = None
            elif cfg.optim.optimizer == 'none':
                data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
                loss, aux_data = total_energy(params, data)
            else:
                data, params, opt_state, loss, aux_data, pmove, search_direction = step(shared_t,
                                                                                        data,
                                                                                        params,
                                                                                        opt_state,
                                                                                        subkeys,
                                                                                        mcmc_width)
            shared_t = shared_t + 1
            loss = loss[0] / simulation_cell.scale if loss is not None else None
            variance = aux_data.variance[0] / simulation_cell.scale ** 2 if aux_data is not None else None
            imaginary = aux_data.imaginary[0] / simulation_cell.scale if aux_data is not None else None
            kinetic = jnp.mean(aux_data.kinetic) / simulation_cell.scale if aux_data is not None else None
            ewald = jnp.mean(aux_data.ewald) / simulation_cell.scale if aux_data is not None else None
            pmove = pmove[0]

            if cfg.log.complex_polarization:
                polarization_data = pmap_polarization(data)[0]
            if cfg.log.structure_factor:
                structure_factor_data = pmap_structure_factor(data)[0][None, :]
                pd_tabel = pd.DataFrame(structure_factor_data)
                pd_tabel.to_csv(str(ckpt_save_path) + '/structure_factor.csv', mode="a", sep=',', header=False)


            if t % cfg.log.stats_frequency == 0 and loss is not None:
                logging.info(
                    '%s Step %05d: %03.4f E_h, variance=%03.4f E_h^2, pmove=%0.2f, imaginary part=%03.4f, '
                    'kinetic=%03.4f E_h, ewald=%03.4f E_h',
                    datetime.datetime.now(), t,
                    loss, variance, pmove, imaginary,
                    kinetic.real, ewald)
                result_dict = {
                               'step': t,
                               'energy': np.asarray(loss),
                               'variance': np.asarray(variance),
                               'pmove': np.asarray(pmove),
                               'imaginary': np.asarray(imaginary),
                               'kinetic': np.asarray(kinetic),
                               'ewald': np.asarray(ewald),
                               }
                if cfg.log.complex_polarization:
                    result_dict['complex_polarization'] = np.asarray(polarization_data)
                writer.write(t,
                             **result_dict,
                             )

            # Update MCMC move width
            if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
                if np.mean(pmoves) > 0.55:
                    mcmc_width *= 1.1
                if np.mean(pmoves) < 0.5:
                    mcmc_width /= 1.1
                pmoves[:] = 0
            pmoves[t % cfg.mcmc.adapt_frequency] = pmove

            if (time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60
              or t >= cfg.optim.iterations - 1
              or (cfg.log.save_frequency_in_step > 0 and t % cfg.log.save_frequency_in_step == 0)):
                # no checkpointing in inference mode
                if cfg.optim.optimizer != 'none':
                    checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width,)

                time_of_last_ckpt = time.time()
