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

from pyscf.pbc import gto, scf
from DeepSolid import supercell
from DeepSolid import distance
import numpy as np

_gldict = {"laplacian": np.s_[:1], "gradient_laplacian": np.s_[0:4]}


def _aostack_mol(ao, gl):
    return np.concatenate(
        [ao[_gldict[gl]], ao[[4, 7, 9]].sum(axis=0, keepdims=True)], axis=0
    )


def _aostack_pbc(ao, gl):
    return [_aostack_mol(ak, gl) for ak in ao]


class SCF:
    def __init__(self, cell, twist=np.ones(3)*0.5):
        """
        Hartree Fock wave function class for QMC simulation

        :param cell: pyscf.pbc.gto.Cell, simulation object
        :param twist:np.array with shape [3]
        """
        self._aostack = _aostack_pbc
        self.coeff_key = ("mo_coeff_alpha", "mo_coeff_beta")
        self.param_split = {}
        self.parameters = {}
        self.k_split = {}
        self.ns_tol = cell.scale
        self.simulation_cell = cell
        self.primitive_cell = cell.original_cell
        self.sim_nelec = self.simulation_cell.nelec
        self.kpts = supercell.get_supercell_kpts(self.simulation_cell)
        self.kpts = self.kpts + np.dot(np.linalg.inv(cell.a), np.mod(twist, 1.0)) * 2 * np.pi
        if hasattr(self.simulation_cell, 'hf_type'):
            hf_type = self.simulation_cell.hf_type
        else:
            hf_type = 'rhf'

        if hf_type == 'uhf':
            self.kmf = scf.KUHF(self.primitive_cell, exxdiv='ewald', kpts=self.kpts).density_fit()

            # break initial guess symmetry for UHF
            dm_up, dm_down = self.kmf.get_init_guess()
            dm_down[:, :2, :2] = 0
            dm = (dm_up, dm_down)
        elif hf_type == 'rhf':
            self.kmf = scf.KHF(self.primitive_cell, exxdiv='ewald', kpts=self.kpts).density_fit()
            dm = self.kmf.get_init_guess()
        else:
            raise ValueError('Unrecognized Hartree Fock type.')

        self.kmf.kernel(dm)
        # self.init_scf()

    def init_scf(self):
        """
        initialization function to set up HF ansatz.
        """
        self.klist = []
        for s, key in enumerate(self.coeff_key):
            mclist = []
            for k in range(self.kmf.kpts.shape[0]):
                # restrict or not
                if len(self.kmf.mo_coeff[0][0].shape) == 2:
                    mca = self.kmf.mo_coeff[s][k][:, np.asarray(self.kmf.mo_occ[s][k] > 0.9)]
                else:
                    minocc = (0.9, 1.1)[s]
                    mca = self.kmf.mo_coeff[k][:, np.asarray(self.kmf.mo_occ[k] > minocc)]
                mclist.append(mca)
            self.param_split[key] = np.cumsum([m.shape[1] for m in mclist])
            self.parameters[key] = np.concatenate(mclist, axis=-1)
            self.k_split[key] = np.array([m.shape[1] for m in mclist])
            self.klist.append(np.concatenate([np.tile(kpt[None, :], (split, 1))
                                              for kpt, split in
                                              zip(self.kmf.kpts, self.k_split[self.coeff_key[s]])]))

    def eval_orbitals_pbc(self, coord, eval_str="GTOval_sph"):
        """
        eval the atomic orbital valus of HF.
        :param coord: electron walkers with shape [batch, ne * ndim].
        :param eval_str:
        :return: atomic orbital valus of HF.
        """
        prim_coord, wrap = distance.np_enforce_pbc(self.primitive_cell.a, coord.reshape([coord.shape[0], -1]))
        prim_coord = prim_coord.reshape([-1, 3])
        wrap = wrap.reshape([-1, 3])
        ao = self.primitive_cell.eval_gto("PBC" + eval_str, prim_coord, kpts=self.kmf.kpts)

        kdotR = np.einsum('ij,kj,nk->in', self.kmf.kpts, self.primitive_cell.a, wrap)
        wrap_phase = np.exp(1j*kdotR)
        ao = [ao[k] * wrap_phase[k][:, None] for k in range(len(self.kmf.kpts))]

        return ao

    def eval_mos_pbc(self, aos, s):
        """
        eval the molecular orbital values.
        :param aos: atomic orbital values.
        :param s: spin index.
        :return: molecular orbital values.
        """
        c = self.coeff_key[s]
        p = np.split(self.parameters[c], self.param_split[c], axis=-1)
        mo = [ao.dot(p[k]) for k, ao in enumerate(aos)]
        return np.concatenate(mo, axis=-1)

    def eval_orb_mat(self, coord):
        """
        eval the orbital matrix of HF.
        :param coord: electron walkers with shape [batch, ne * ndim].
        :return: orbital matrix of HF.
        """
        batch, nelec, ndim = coord.shape
        aos = self.eval_orbitals_pbc(coord)
        aos_shape = (self.ns_tol, batch, nelec, -1)

        aos = np.reshape(aos, aos_shape)
        mos = []
        for s in [0, 1]:
            i0, i1 = s * self.sim_nelec[0], self.sim_nelec[0] + s * self.sim_nelec[1]
            ne = self.sim_nelec[s]
            mo = self.eval_mos_pbc(aos[:, :, i0:i1], s).reshape([batch, ne, ne])
            mos.append(mo)
        return mos

    def eval_slogdet(self, coord):
        """
        eval the slogdet of HF
        :param coord: electron walkers with shape [batch, ne * ndim].
        :return: slogdet of HF.
        """
        mos = self.eval_orb_mat(coord)
        slogdets = [np.linalg.slogdet(mo) for mo in mos]
        phase, slogdet = list(map(lambda x, y: [x[0] * y[0], x[1] + y[1]], *zip(slogdets)))[0]

        return phase, slogdet

    def eval_phase(self, coord):
        """

        :param coord:
        :return: a list of phase with shape [B, nk * nao]
        """
        coords = np.split(coord, (self.sim_nelec[0], sum(self.sim_nelec)), axis=1)
        kdots = [np.einsum('ijl, kl->ijk', cor, kpt) for cor, kpt in zip(coords, self.klist)]
        phase = [np.exp(1j * kdot) for kdot in kdots]
        return phase

    def pure_periodic(self, coord):
        orbitals = self.eval_orb_mat(coord)
        ## minus symbol makes mos to be periodical
        phases = self.eval_phase(-coord)
        return [orbital * phase for orbital, phase in zip(orbitals, phases)]

    def eval_inverse(self, coord):
        mats = self.eval_orb_mat(coord)
        inverse = [np.linalg.inv(mat) for mat in mats]

        return inverse

    def _testrow(self, e, vec, inverse, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self.sim_nelec[0]) if spin is None else spin
        elec = e - s * self.sim_nelec[0]
        if mask is None:
            return np.einsum("i...j,ij...->i...", vec, inverse[s][:, :, elec])

        return np.einsum("i...j,ij...->i...", vec, inverse[s][mask][:, :, elec])

    def laplacian(self, e, coord, inverse):
        s = int(e >= self.sim_nelec[0])
        ao = self.eval_orbitals_pbc(coord, eval_str="GTOval_sph_deriv2")
        mo = self.eval_mos_pbc(self._aostack(ao, "laplacian"), s)
        ratios = np.asarray([self._testrow(e, x, inverse=inverse) for x in mo])
        return ratios[1] / ratios[0]

    def kinetic(self, coord):
        ke = np.zeros(coord.shape[0])
        inverse = self.eval_inverse(coord)
        for e in range(self.simulation_cell.nelectron):
            ke = ke - 0.5 * np.real(self.laplacian(e,
                                                   coord[:, e, :],
                                                   inverse=inverse))
        return ke

    def __call__(self, coord):
        phase, slogdet = self.eval_slogdet(coord)
        psi = np.exp(slogdet) * phase
        return psi
