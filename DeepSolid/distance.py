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

from functools import partial
import jax
import jax.numpy as jnp
import logging


class MinimalImageDistance:
    """Computer minimal image distance between particles and its images"""

    def __init__(self, latvec, verbose=0):
        """

        :param latvec: array with shape [3,3], each row with a lattice vector
        """

        latvec = jnp.asarray(latvec)
        ortho_tol = 1e-10
        diagonal = jnp.all(jnp.abs(latvec - jnp.diag(jnp.diagonal(latvec))) < ortho_tol)
        if diagonal:
            self.dist_i = self.diagonal_dist_i
            if verbose == 0:
                logging.info("Diagonal lattice vectors")
        else:
            orthogonal = (
                jnp.dot(latvec[0], latvec[1]) < ortho_tol
                and jnp.dot(latvec[1], latvec[2]) < ortho_tol
                and jnp.dot(latvec[2], latvec[0]) < ortho_tol
            )
            if orthogonal:
                self.dist_i = self.orthogonal_dist_i
                if verbose == 0:
                    logging.info("Orthogonal lattice vectors")
            else:
                self.dist_i = self.general_dist_i
                if verbose == 0:
                    logging.info("Non-orthogonal lattice vectors")
        self._latvec = latvec
        self._invvec = jnp.linalg.inv(latvec)
        self.dim = self._latvec.shape[-1]
        # list of all 26 neighboring cells
        mesh_grid = jnp.meshgrid(*[jnp.array([0, 1, 2]) for _ in range(3)])
        self.point_list = jnp.stack([m.ravel() for m in mesh_grid], axis=0).T - 1
        self.shifts = self.point_list @ self._latvec

    def general_dist_i(self, configs, vec, return_wrap=False):
        """
        calculate minimal distance between electron and ion in the most general lattice vector

        :param configs: ion coordinate with shape [N_atom * 3]
        :param vec: electron coordinate with shape [N_ele * 3]
        :return: minimal image distance between electron and atom with shape [N_ele, N_atom, 3]
        """
        configs = configs.reshape([1, -1, self.dim])
        v = vec.reshape([-1, 1, self.dim])
        d1 = v - configs
        shifts = self.shifts.reshape((-1, *[1] * (len(d1.shape) - 1), 3))
        d1all = d1[None] + shifts
        dists = jnp.linalg.norm(d1all, axis=-1)
        mininds = jnp.argmin(dists, axis=0)
        inds = jnp.meshgrid(*[jnp.arange(n) for n in mininds.shape], indexing='ij')
        if return_wrap:
            return d1all[(mininds, *inds)], -self.point_list[mininds]
        else:
            return d1all[(mininds, *inds)]

    def orthogonal_dist_i(self, configs, vec, return_wrap=False):
        """
        calculate minimal distance between electron and ion in the orthogonal lattice vector

        :param configs: ion coordinate with shape [N_atom * 3]
        :param vec: electron coordinate with shape [N_ele * 3]
        :return: minimal image distance between electron and atom with shape [N_ele, N_atom, 3]
        """
        configs = configs.reshape([1, -1, self.dim]).real
        v = vec.reshape([-1, 1, self.dim]).real
        d1 = v - configs
        frac_disps = jnp.einsum("...ij,jk->...ik", d1, self._invvec)
        replace_frac_disps = (frac_disps + 0.5) % 1 - 0.5
        if return_wrap == False:
            return jnp.einsum("...ij,jk->...ik", replace_frac_disps, self._latvec)
        else:
            wrap = -((frac_disps + 0.5) // 1)
            return jnp.einsum("...ij,jk->...ik", replace_frac_disps, self._latvec), wrap

    def diagonal_dist_i(self, configs, vec, return_wrap=False):
        """
        calculate minimal distance between electron and ion in the diagonal lattice vector

        :param configs: ion coordinate with shape [N_atom * 3]
        :param vec: electron coordinate with shape [N_ele * 3]
        :return: minimal image distance between electron and atom with shape [N_ele, N_atom, 3]
        """
        configs = configs.reshape([1, -1, self.dim]).real
        v = vec.reshape([-1, 1, self.dim]).real
        d1 = v - configs
        latvec_diag = jnp.diagonal(self._latvec)
        replace_d1 = (d1 + latvec_diag / 2) % latvec_diag - latvec_diag / 2
        if return_wrap == False:
            return replace_d1
        else:
            ## minus applies after //, order of // and - sign matters
            wrap = -((d1 + latvec_diag / 2) // latvec_diag)
            return replace_d1, wrap

    def dist_matrix(self, configs):
        """
        calculate minimal distance between electrons

        :param configs: electron coordinate with shape [N_ele * 3]
        :return: vs: electron coordinate diffs with shape [N_ele, N_ele,  3]
        """

        vs = self.dist_i(configs, configs)
        vs = vs * (1 - jnp.eye(vs.shape[0]))[..., None]

        return vs


@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
def enforce_pbc(latvec, epos):
    """
    Enforces periodic boundary conditions on a set of configs.

    :param lattvecs: orthogonal lattice vectors defining 3D torus: (3,3)
    :param epos: attempted new electron coordinates: (N_ele * 3)
    :return: final electron coordinates with PBCs imposed: (N_ele * 3)
    """

    # Writes epos in terms of (lattice vecs) fractional coordinates
    dim = latvec.shape[-1]
    epos = epos.reshape(-1, dim)
    recpvecs = jnp.linalg.inv(latvec)
    epos_lvecs_coord = jnp.einsum("ij,jk->ik", epos, recpvecs)

    tmp = jnp.divmod(epos_lvecs_coord, 1)
    wrap = tmp[0]
    final_epos = jnp.matmul(tmp[1], latvec).ravel()
    return final_epos, wrap

import numpy as np

def np_enforce_pbc(latvec, epos):
    """
    Enforces periodic boundary conditions on a set of configs. Used in float 32 mode.

    :param lattvecs: orthogonal lattice vectors defining 3D torus: (3,3)
    :param epos: attempted new electron coordinates: (N_ele * 3)
    :return: final electron coordinates with PBCs imposed: (N_ele * 3)
    """

    # Writes epos in terms of (lattice vecs) fractional coordinates
    dim = latvec.shape[-1]
    epos = epos.reshape(-1, dim)
    recpvecs = np.linalg.inv(latvec)
    epos_lvecs_coord = np.einsum("ij,jk->ik", epos, recpvecs)

    tmp = np.divmod(epos_lvecs_coord, 1)
    wrap = tmp[0]
    final_epos = np.matmul(tmp[1], latvec).ravel()
    return final_epos, wrap
