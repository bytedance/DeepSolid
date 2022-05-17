# MIT License
#
# Copyright (c) 2019 Lucas K Wagner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

import logging

import jax
import jax.numpy as jnp
from DeepSolid import distance


class EwaldSum:
    def __init__(self, cell, ewald_gmax=200, nlatvec=1):
        """
        :parameter cell: pyscf Cell object (simulation cell)
        :parameter int ewald_gmax: how far to take reciprocal sum; probably never needs to be changed.
        :parameter int nlatvec: how far to take real-space sum; probably never needs to be changed.
        """
        self.nelec = cell.nelec
        self.atom_coords = jnp.asarray(cell.atom_coords())
        self.atom_charges = jnp.asarray(cell.atom_charges())
        self.latvec = jnp.asarray(cell.lattice_vectors())
        self.dist = distance.MinimalImageDistance(self.latvec)
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec):
        """
        Generates list of lattice-vector displacements to add together for real-space sum

        :parameter int nlatvec: sum goes from `-nlatvec` to `nlatvec` in each lattice direction.
        """
        XYZ = jnp.meshgrid(*[jnp.arange(-nlatvec, nlatvec + 1)] * 3, indexing="ij")
        xyz = jnp.stack(XYZ, axis=-1).reshape((-1, 3))
        self.lattice_displacements = jnp.asarray(jnp.dot(xyz, self.latvec))

    def set_up_reciprocal_ewald_sum(self, ewald_gmax):
        cellvolume = jnp.linalg.det(self.latvec)
        recvec = jnp.linalg.inv(self.latvec).T

        # Determine alpha
        smallestheight = jnp.amin(1 / jnp.linalg.norm(recvec, axis=1))
        self.alpha = 5.0 / smallestheight
        logging.info(f"Setting Ewald alpha to {self.alpha.item()}")

        # Determine G points to include in reciprocal Ewald sum
        gptsXpos = jnp.meshgrid(
            jnp.arange(1, ewald_gmax + 1),
            *[jnp.arange(-ewald_gmax, ewald_gmax + 1)] * 2,
            indexing="ij"
        )
        zero = jnp.asarray([0])
        gptsX0Ypos = jnp.meshgrid(
            zero,
            jnp.arange(1, ewald_gmax + 1),
            jnp.arange(-ewald_gmax, ewald_gmax + 1),
            indexing="ij",
        )
        gptsX0Y0Zpos = jnp.meshgrid(
            zero, zero, jnp.arange(1, ewald_gmax + 1), indexing="ij"
        )
        gs = zip(
            *[
                select_big(x, cellvolume, recvec, self.alpha)
                for x in (gptsXpos, gptsX0Ypos, gptsX0Y0Zpos)
            ]
        )
        self.gpoints, self.gweight = [jnp.concatenate(x, axis=0) for x in gs]
        self.set_ewald_constants(cellvolume)

    def set_ewald_constants(self, cellvolume):
        self.i_sum = jnp.sum(self.atom_charges)
        ii_sum2 = jnp.sum(self.atom_charges ** 2)
        ii_sum = (self.i_sum ** 2 - ii_sum2) / 2

        self.ijconst = -jnp.pi / (cellvolume * self.alpha ** 2)
        self.squareconst = -self.alpha / jnp.sqrt(jnp.pi) + self.ijconst / 2

        self.ii_const = ii_sum * self.ijconst + ii_sum2 * self.squareconst
        self.e_single_test = -self.i_sum * self.ijconst + self.squareconst
        self.ion_ion = self.ewald_ion()

        # XC correction not used, so we can compare to other codes
        # rs = lambda ne: (3 / (4 * np.pi) / (ne * cellvolume)) ** (1 / 3)
        # cexc = 0.36
        # xc_correction = lambda ne: cexc / rs(ne)

    def ee_const(self, ne):
        return ne * (ne - 1) / 2 * self.ijconst + ne * self.squareconst

    def ei_const(self, ne):
        return -ne * self.i_sum * self.ijconst

    def e_single(self, ne):
        return (
            0.5 * (ne - 1) * self.ijconst - self.i_sum * self.ijconst + self.squareconst
        )

    def ewald_ion(self):
        # Real space part
        if len(self.atom_charges) == 1:
            ion_ion_real = 0
        else:
            ion_distances = self.dist.dist_matrix(self.atom_coords.ravel())
            rvec = ion_distances[None, :, :, :] + self.lattice_displacements[:, None, None, :]
            r = jnp.linalg.norm(rvec, axis=-1)
            charge_ij = self.atom_charges[..., None] * self.atom_charges[None, ...]
            ion_ion_real = jnp.sum(jnp.triu(charge_ij * jax.lax.erfc(self.alpha * r) / r, k=1))
        # Reciprocal space part
        GdotR = jnp.dot(self.gpoints, jnp.asarray(self.atom_coords.T))
        self.ion_exp = jnp.dot(jnp.exp(1j * GdotR), self.atom_charges)
        ion_ion_rec = jnp.dot(self.gweight, jnp.abs(self.ion_exp) ** 2)

        ion_ion = ion_ion_real + ion_ion_rec
        return ion_ion

    def _real_cij(self, dists):
        r = dists[:, :, None, :] + self.lattice_displacements
        r = jnp.linalg.norm(r, axis=-1)
        cij = jnp.sum(jax.lax.erfc(self.alpha * r) / r, axis=-1)
        return cij

    def ewald_electron(self, configs):
        nelec = sum(self.nelec)

        # Real space electron-ion part
        # ei_distances shape (elec, atom, dim)
        ei_distances = self.dist.dist_i(self.atom_coords.ravel(), configs)
        ei_cij = self._real_cij(ei_distances)
        ei_real_separated = jnp.sum(-self.atom_charges[None, :] * ei_cij)

        # Real space electron-electron part
        ee_real_separated = jnp.array(0.)
        if nelec > 1:
            ee_distances = self.dist.dist_matrix(configs)
            rvec = ee_distances[None, :, :, :] + self.lattice_displacements[:, None, None, :]
            r = jnp.linalg.norm(rvec, axis=-1)
            ee_real_separated = jnp.sum(jnp.triu(jax.lax.erfc(self.alpha * r) / r, k=1))

            # ee_distances = self.dist.dist_matrix(configs)
            # ee_cij = self._real_cij(ee_distances)
            #
            # for ((i, j), val) in zip(ee_inds, ee_cij.T):
            #     ee_real_separated[:, i] += val
            #     ee_real_separated[:, j] += val
            # ee_real_separated /= 2

        ee_recip, ei_recip = self.reciprocal_space_electron(configs)
        ee = ee_real_separated + ee_recip
        ei = ei_real_separated + ei_recip
        return ee, ei

    def reciprocal_space_electron(self, configs):
        # Reciprocal space electron-electron part
        e_GdotR = jnp.einsum("ik,jk->ij", configs.reshape(sum(self.nelec), -1), self.gpoints)
        sum_e_sin = jnp.sin(e_GdotR).sum(axis=0)
        sum_e_cos = jnp.cos(e_GdotR).sum(axis=0)
        ee_recip = jnp.dot(sum_e_sin ** 2 + sum_e_cos ** 2, self.gweight)
        ## Reciprocal space electron-ion part
        coscos_sinsin = -self.ion_exp.real * sum_e_cos - self.ion_exp.imag * sum_e_sin
        ei_recip = 2 * jnp.dot(coscos_sinsin, self.gweight)
        return ee_recip, ei_recip

    def energy(self, configs):
        nelec = sum(self.nelec)
        ee, ei = self.ewald_electron(configs)
        ee += self.ee_const(nelec)
        ei += self.ei_const(nelec)
        ii = self.ion_ion + self.ii_const
        return jnp.asarray(ee), jnp.asarray(ei), jnp.asarray(ii)


def select_big(gpts, cellvolume, recvec, alpha):
    gpoints = jnp.einsum("j...,jk->...k", gpts, recvec) * 2 * jnp.pi
    gsquared = jnp.einsum("...k,...k->...", gpoints, gpoints)
    gweight = 4 * jnp.pi * jnp.exp(-gsquared / (4 * alpha ** 2))
    gweight /= cellvolume * gsquared
    bigweight = gweight > 1e-12
    return gpoints[bigweight], gweight[bigweight]
