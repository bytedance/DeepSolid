# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp
import numpy as np
import logging
from time import time

logging.basicConfig(level=logging.INFO)
jax.config.update("jax_enable_x64", True)

import test_cell
from DeepSolid import network
from DeepSolid import supercell
from DeepSolid import init_guess
from DeepSolid import hf
from DeepSolid import base_config

# Define your test system
S = np.eye(3)  # Define how to tile primitive cell
cell = test_cell.cell
simulation_cell = supercell.get_supercell(cell, S=S)

# Define the scaled twist momentum k_s
scaled_twist = 0.0
twist = scaled_twist * jnp.ones(3)

# Do HF calculation to get k-points
scf_approx = hf.SCF(simulation_cell, twist=twist)
scf_approx.init_scf()

# Define your neural network settings
cfg = base_config.default()
cfg.network.detnet.determinants = 8
system_dict = {'klist': scf_approx.klist,  # occupied k points from HF
               'simulation_cell': simulation_cell,
               }
system_dict.update(cfg.network.detnet)
system_dict['envelope_type'] = 'isotropic'
system_dict['full_det'] = False

# quantum number of periodic boundary condition
kp = sum([jnp.sum(k, axis=0) for k in system_dict['klist']])

# make callable neural network functions
slater_forward = network.make_solid_fermi_net(**system_dict, method_name='eval_logdet')
slater_phase_and_slogdet = network.make_solid_fermi_net(**system_dict, method_name='eval_phase_and_slogdet')
slater_mat_forward = network.make_solid_fermi_net(**system_dict, method_name='eval_mats')

# initialize parameters and electron positions
key = jax.random.PRNGKey(int(time()))
internal_sim_cell = init_guess.pyscf_to_cell(simulation_cell)
coord = init_guess.init_electrons(key,
                                  internal_sim_cell,
                                  simulation_cell.a,
                                  simulation_cell.nelec,
                                  batch_size=1)[0]
p = slater_forward.init(key, data=coord)


def test_periodic_bc(params, x):
    """
    test periodic boundary condition of wf
    :param params:
    :param x:
    :return:
    """
    trans = cell.lattice_vectors()[2]
    x1 = x
    x2 = x + jnp.tile(trans, simulation_cell.nelectron)
    # simultaneous translation of all electron over a primitive lattice vector
    p1, s1 = slater_phase_and_slogdet.apply(params=params, x=x1)
    p2, s2 = slater_phase_and_slogdet.apply(params=params, x=x2)
    logging.info(f'original:{p1, s1}')
    logging.info(f'translated:{p2, s2}')
    logging.info(f'kp angle:{jnp.angle(p2 / p1) / np.pi} pi')
    assert jnp.allclose(s1, s2)
    assert jnp.allclose(p1 * jnp.exp(1j * jnp.dot(kp, trans)), p2)
    logging.info('Periodic BC checked')


def test_twisted_bc(params, x):
    """
    test twist boundary condition of wf
    :param params:
    :param x:
    :return:
    """
    x1 = x
    x2 = x + jnp.concatenate([simulation_cell.lattice_vectors()[1][None, ...],
                              jnp.zeros(shape=[simulation_cell.nelectron - 1, 3])
                              ], axis=0).ravel()
    # translation of a single electron over a supercell lattice vector
    p1, s1 = slater_phase_and_slogdet.apply(params=params, x=x1)
    p2, s2 = slater_phase_and_slogdet.apply(params=params, x=x2)
    logging.info(f'original:{p1, s1}')
    logging.info(f'translated:{p2, s2}')
    logging.info(f'ks angle:{jnp.angle(p2 / p1) / jnp.pi} pi')
    assert jnp.allclose(s1, s2)
    assert jnp.allclose(p2 / p1,
                        jnp.exp(1j * scaled_twist * 2 * jnp.pi))
    logging.info('Twisted BC checked')


def test_anti_symmetry(params, x):
    """
    test anti-symmetry condition of wf
    :param params:
    :param x:
    :return:
    """
    x1 = x
    x2 = jnp.concatenate([x1[3:6], x1[:3], x1[6:]])
    p1, s1 = slater_phase_and_slogdet.apply(params=params, x=x1)
    p2, s2 = slater_phase_and_slogdet.apply(params=params, x=x2)
    assert jnp.allclose(p1, -p2)
    assert jnp.allclose(s1, s2)
    logging.info('Anti symmetry checked')


if __name__ == '__main__':
    test_periodic_bc(x=coord, params=p)
    test_twisted_bc(x=coord, params=p)
    test_anti_symmetry(x=coord, params=p)
