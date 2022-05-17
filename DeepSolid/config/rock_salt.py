# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from pyscf.pbc import gto

from DeepSolid import base_config
from DeepSolid import supercell
from DeepSolid.utils import units


def get_config(input_str):
    X, Y, L_Ang, S, basis= input_str.split(',')
    S = np.eye(3) * int(S)
    cfg = base_config.default()
    L_Ang = float(L_Ang)
    L_Bohr = units.angstrom2bohr(L_Ang)

    # Set up cell
    cell = gto.Cell()
    cell.atom = [[X, [0.0,     0.0,     0.0]],
                 [Y, [0.5 * L_Bohr, 0.5 * L_Bohr, 0.5 * L_Bohr]]]


    cell.basis = basis
    cell.a = (np.ones((3, 3)) - np.eye(3)) * L_Bohr / 2
    cell.unit = "B"
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()
    simulation_cell = supercell.get_supercell(cell, S)
    cfg.system.pyscf_cell = simulation_cell

    return cfg