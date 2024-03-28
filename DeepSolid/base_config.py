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

import ml_collections
from ml_collections import config_dict


def default() -> ml_collections.ConfigDict:
    """Create set of default parameters for running qmc.py.

    Note: placeholders (cfg.system.molecule and cfg.system.electrons) must be
    replaced with appropriate values.

    Returns:
      ml_collections.ConfigDict containing default settings.
    """
    # wavefunction output.
    cfg = ml_collections.ConfigDict({
        'batch_size': 100,  # batch size
        # Config module used. Should be set in get_config function as either the
        # absolute module or relative to the configs subdirectory. Relative
        # imports must start with a '.' (e.g. .atom). Do *not* override on
        # command-line. Do *not* set using __name__ from inside a get_config
        # function, as config_flags overrides this when importing the module using
        # importlib.import_module.
        'config_module': __name__,
        'use_x64': True, # use float64 or 32
        'optim': {
            'iterations': 1000000,  # number of iterations
            'optimizer': 'kfac',
            'local_energy_outlier_width': 5.0,
            'lr': {
                'rate': 5.e-2,  # learning rate, different from the reported lr in FermiNet
                                # since DeepSolid energy gradient is not batch-size dependent
                'decay': 1.0,  # exponent of learning rate decay
                'delay': 10000.0,  # term that sets the scale of the rate decay
            },
            'clip_el': 5.0,  # If not none, scale at which to clip local energy
            'clip_type': 'real', # Clip real and imag part of gradient.
            'gradient_clip': 5.0,
            # ADAM hyperparameters. See optax documentation for details.
            'adam': {
                'b1': 0.9,
                'b2': 0.999,
                'eps': 1.e-8,
                'eps_root': 0.0,
            },
            'kfac': {
                'invert_every': 1,
                'cov_update_every': 1,
                'damping': 0.001,
                'cov_ema_decay': 0.95,
                'momentum': 0.0,
                'momentum_type': 'regular',
                # Warning: adaptive damping is not currently available.
                'min_damping': 1.e-4,
                'norm_constraint': 0.001,
                'mean_center': True,
                'l2_reg': 0.0,
                'register_only_generic': False,
            },
            'ministeps': 1,
            'laplacian_mode': 'for', # specify the laplacian evaluation mode, mode is one of 'for', 'partition' or 'hessian'
            # 'for' mode calculates the laplacian of each electron one by one, which is slow but save GPU memory
            # 'hessian' mode calculates the laplacian in a highly parallized mode, which is fast but require GPU memory
            # 'partition' mode calculate the laplacian in a moderate way.
            'partition_number': 3,
            # Only used for 'partition' mode.
            # partition_number must be divisivle by (dim * number of electrons). The smaller the faster, but requires more memory.
        },
        'log': {
            'stats_frequency': 1,  # iterations between logging of stats
            'save_frequency': 10.0,  # minutes between saving network params
            'save_frequency_in_step': -1,
            'save_path': '',
            # specify the local save path
            'restore_path': '',
            # specify the restore path which contained saved Model parameters.
            'local_energies': False,
            'complex_polarization': False, # log polarization order parameter which is useful for hydrogen chain.
            'structure_factor': False,
            # return the strture factor S(k) at reciprocal lattices of supercell
            # log S(k) requires a lot of storage space, be careful.
            'stats_file_name': 'train_stats'
        },
        'system': {
            'pyscf_cell': None, # simulation cell obj
            'ndim': 3, #dimension of the system
            'internal_cell': None,
        },
        'mcmc': {
            # Note: HMC options are not currently used.
            # Number of burn in steps after pretraining.  If zero do not burn in
            # or reinitialize walkers.
            'burn_in': 100,
            'steps': 20,  # Number of MCMC steps to make between network updates.
            # Width of (atom-centred) Gaussian used to generate initial electron
            # configurations.
            'init_width': 0.8,
            # Width of Gaussian used for random moves for RMW or step size for
            # HMC.
            'move_width': 0.02,
            # Number of steps after which to update the adaptive MCMC step size
            'adapt_frequency': 100,
            'init_means': (),  # Not implemented in JAX.
            # If true, scale the proposal width for each electron by the harmonic
            # mean of the distance to the nuclei.
            'importance_sampling': False,
            # whether to use importance sampling in MCMC step, untested yet
            # Metropolis sampling will be used if false
            'one_electron': False
            # If true, use one-electron moves, untested yet
        },
        'network': {
            'detnet': {
                'envelope_type': 'isotropic',
                # only isotropic mode has been tested
                'bias_orbitals':  False,
                'use_last_layer': False,
                'full_det': False,
                'hidden_dims':  ((256, 32), (256, 32), (256, 32)),
                'determinants':  8,
                'after_determinants':  1,
                'distance_type': 'nu',
            },
            'twist': (0.0, 0.0, 0.0), # Difine the twist of wavefunction,
                                      # twists are given in terms of fractions of supercell reciprocal vectors
        },
        'debug': {
            # Check optimizer state, parameters and loss and raise an exception if
            # NaN is found.
            'check_nan': False, # check whether the gradient contain nans before optimize, if True, retry.
            'deterministic': False,  # Use a deterministic seed.
        },
        'pretrain': {
            'method': 'net',  # Method is one of 'hf', 'net'.
            'iterations': 1000,
            'lr': 3e-4,
            'steps': 1, #mcmc steps between each pretrain iterations
        },
    })

    return cfg


def resolve(cfg):
    cfg = cfg.copy_and_resolve_references()
    return cfg
