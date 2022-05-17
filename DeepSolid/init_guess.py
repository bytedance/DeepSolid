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
import pyscf.pbc.gto
from typing import Sequence
from DeepSolid.utils import system
from DeepSolid import distance


def init_electrons(
        key,
        cell: Sequence[system.Atom],
        latvec,
        electrons: Sequence[int],
        batch_size: int,
        init_width=0.5,
) -> jnp.ndarray:
    """
    Initializes electron positions around each atom.

    :param key: jax key for random
    :param cell: internal representation of simulation cell
    :param latvec: lattice vector of cell
    :param electrons: list of up, down electrons
    :param batch_size: batch_size for simulation
    :param init_width: std of gaussian used for initialization
    :return: jnp.array with shape [Batch_size, N_ele * ndim]
    """
    if sum(atom.charge for atom in cell) != sum(electrons):
        if len(cell) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:

        atomic_spin_configs = [
            (atom.element.nalpha - int((atom.atomic_number - atom.charge) // 2),
             atom.element.nbeta - int((atom.atomic_number - atom.charge) // 2))
            for atom in cell
        ]
        # element.nalpha return the up spin number of the single element, if ecp is used, [nalpha,nbeta] should be reduce
        # with the the core charge which equals atomic_number - atom.charge
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            if atomic_spin_configs[i][0] > 0:
                atomic_spin_configs[i] = nalpha - 1, nbeta + 1

    # Assign each electron to an atom initially.
    electron_positions = []
    for i in range(2):
        for j in range(len(cell)):
            atom_position = jnp.asarray(cell[j].coords)
            electron_positions.append(jnp.tile(atom_position, atomic_spin_configs[j][i]))
    electron_positions = jnp.concatenate(electron_positions)
    # Create a batch of configurations with a Gaussian distribution about each
    # atom.
    key, subkey = jax.random.split(key)
    guess = electron_positions + init_width * jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
    replaced_guess, _ = distance.enforce_pbc(latvec, guess)
    return replaced_guess



def pyscf_to_cell(cell: pyscf.pbc.gto.Cell):
    """
    Converts the pyscf cell to the internal representation.

    :param cell: pyscf.cell object
    :return: internal cell representation
    """
    internal_cell = [system.Atom(cell.atom_symbol(i),
                                 cell.atom_coords()[i],
                                 charge=cell.atom_charges()[i], )
                     for i in range(cell.natm)]
    ##  cfg.system.pyscf_mol.atom_charges()[i] return the screen charge of i atom if ecp is used
    return internal_cell