# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp
import functools

import pyscf.pbc.gto
from DeepSolid import constants


def make_complex_polarization(simulation_cell: pyscf.pbc.gto.Cell,
                              direction: int = 0,
                              ndim=3):
    '''
    the order parameter which is used to specify the hydrogen chain
    :param simulation_cell:
    :param direction:
    :param ndim:
    :return:
    '''

    rec_vec = simulation_cell.reciprocal_vectors()[direction]

    def complex_polarization(data):
        """

        :param data: electron walkers with shape [batch, ne * ndim]
        :return: complex polarization with shape []
        """
        leading_shape = list(data.shape[:-1])
        data = data.reshape(leading_shape + [-1, ndim])
        dots = jnp.einsum('i,...i->...', rec_vec, data)
        dots = jnp.sum(dots, axis=-1)
        polarization = jnp.exp(1j * dots)
        polarization = jnp.mean(polarization, axis=-1)
        polarization = constants.pmean_if_pmap(polarization, axis_name=constants.PMAP_AXIS_NAME)
        return polarization

    return complex_polarization

def make_structure_factor(simulation_cell: pyscf.pbc.gto.Cell,
                          nq=4,
                          ndim=3):
    mesh_grid = jnp.meshgrid(*[jnp.array(range(0, nq)) for _ in range(3)])
    point_list = jnp.stack([m.ravel() for m in mesh_grid], axis=0).T
    rec_vec = simulation_cell.reciprocal_vectors()

    qvecs = point_list @ rec_vec
    rec_vec = qvecs
    nelec = simulation_cell.nelectron

    def structure_factor(data):
        """

        :param data: electron walkers with shape [batch, ne * ndim]
        :return: complex polarization with shape []
        """
        leading_shape = list(data.shape[:-1])
        data = data.reshape(leading_shape + [-1, ndim])
        dots = jnp.einsum('kj,...j->...k', rec_vec, data)
        # batch ne npoint
        rho_k = jnp.exp(1j * dots)
        rho_k = jnp.sum(rho_k, axis=1)
        rho_k_one = jnp.mean(rho_k, axis=0)
        rho_k_one_mean = constants.pmean_if_pmap(rho_k_one, axis_name=constants.PMAP_AXIS_NAME)
        rho_k_two = jnp.mean(jnp.abs(rho_k)**2, axis=0)
        rho_k_two_mean = constants.pmean_if_pmap(rho_k_two, axis_name=constants.PMAP_AXIS_NAME)

        sk = rho_k_two_mean - jnp.abs(rho_k_one_mean)**2
        sk = sk / nelec

        return sk

    return structure_factor