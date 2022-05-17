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

import numpy as np
import pyscf.pbc.gto


def get_supercell_kpts(supercell):
    """

    :param supercell:
    :return:supercell k points which belong to the unit box primitive cell k point space
    """
    Sinv = np.linalg.inv(supercell.S).T
    u = [0, 1]
    unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box_ = np.dot(unit_box, supercell.S.T)
    xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
    kptmesh = np.meshgrid(*[np.arange(*r) for r in xyz_range], indexing="ij")
    possible_kpts = np.dot(np.stack([x.ravel() for x in kptmesh]).T, Sinv)
    in_unit_box = (possible_kpts >= 0) * (possible_kpts < 1 - 1e-12)
    select = np.where(np.all(in_unit_box, axis=1))[0]
    reclatvec = np.linalg.inv(supercell.original_cell.lattice_vectors()).T * 2 * np.pi
    return np.dot(possible_kpts[select], reclatvec)


def get_supercell_copies(latvec, S):
    Sinv = np.linalg.inv(S).T
    u = [0, 1]
    unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box_ = np.dot(unit_box, S)
    xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
    mesh = np.meshgrid(*[np.arange(*r) for r in xyz_range], indexing="ij")
    possible_pts = np.dot(np.stack([x.ravel() for x in mesh]).T, Sinv.T)
    in_unit_box = (possible_pts >= 0) * (possible_pts < 1 - 1e-12)
    select = np.where(np.all(in_unit_box, axis=1))[0]
    return np.linalg.multi_dot((possible_pts[select], S, latvec))


def get_supercell(cell, S, sym_type='minimal') -> pyscf.pbc.gto.Cell:
    """
    generate supercell from primitive cell with S specified

    :param cell: pyscf Cell object
    :param S: (3, 3) supercell matrix for QMC from cell defined by cell.a.
    :return: QMC simulation cell
    """
    import pyscf.pbc
    scale = np.abs(int(np.round(np.linalg.det(S))))
    superlattice = np.dot(S, cell.lattice_vectors())
    Rpts = get_supercell_copies(cell.lattice_vectors(), S)
    atom = []
    for (name, xyz) in cell._atom:
        atom.extend([(name, xyz + R) for R in Rpts])
    supercell = pyscf.pbc.gto.Cell()
    supercell.a = superlattice
    supercell.atom = atom
    supercell.ecp = cell.ecp
    supercell.basis = cell.basis
    supercell.exp_to_discard = cell.exp_to_discard
    supercell.unit = "Bohr"
    supercell.spin = cell.spin * scale
    supercell.build()
    supercell.original_cell = cell
    supercell.S = S
    supercell.scale = scale
    supercell.output = None
    supercell.stdout = None
    supercell = set_symmetry_lat(supercell, sym_type)
    logging.info(f'Use {sym_type} type feature.')
    return supercell


def set_symmetry_lat(supercell, sym_type='minimal'):
    '''
    Attach corresponding lattice vectors to the simulation cell.

    :param supercell:
    :param sym_type:specify the symmetry of constructed distance feature,
    Minimal is used as default, and other type hasn't been tested.
    :return: simulation cell with symmetry specified.
    '''
    prim_bv = supercell.original_cell.reciprocal_vectors()
    sim_bv = supercell.reciprocal_vectors()
    if sym_type == 'minimal':
        mat = np.eye(3)
    elif sym_type == 'fcc':
        mat = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 1]])
    elif sym_type == 'bcc':
        mat = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, -1, 0],
                        [1, 0, -1],
                        [0, 1, -1]])
    elif sym_type == 'hexagonal':
        mat = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, -1, 0]])
    else:
        mat = np.eye(3)

    prim_bv = mat @ prim_bv
    sim_bv = mat @ sim_bv

    prim_av = np.linalg.pinv(prim_bv).T
    sim_av = np.linalg.pinv(sim_bv).T
    supercell.BV = sim_bv
    supercell.AV = sim_av
    supercell.original_cell.BV = prim_bv
    supercell.original_cell.AV = prim_av
    return supercell


def get_k_indices(cell, mf, kpts, tol=1e-6):
    """Given a list of kpts, return inds such that mf.kpts[inds] is a list of kpts equivalent to the input list"""
    kdiffs = mf.kpts[None] - kpts[:, None]
    frac_kdiffs = np.dot(kdiffs, cell.lattice_vectors().T) / (2 * np.pi)
    kdiffs = np.mod(frac_kdiffs + 0.5, 1) - 0.5
    return np.nonzero(np.linalg.norm(kdiffs, axis=-1) < tol)[1]
