# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
from pyscf.pbc import gto

from DeepSolid import base_config
from DeepSolid import supercell


def get_config(input_str):
    symbol, Sx, Sy, Sz, L, spin, basis= input_str.split(',')
    Sx = int(Sx)
    Sy = int(Sy)
    Sz = int(Sz)
    S = np.diag([Sx, Sy, Sz])
    L = float(L)
    spin = int(spin)
    cfg = base_config.default()

    # Set up cell
    cell = gto.Cell()
    cell.atom = f"""
    {symbol} {L}   {0}   {0}
    {symbol} 0 0 0
    """
    cell.basis = basis
    cell.a = np.array([[2*L, 0,   0],
                       [0, 100, 0],
                       [0, 0, 100]])
    cell.unit = "B"
    cell.spin = spin
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()
    simulation_cell = supercell.get_supercell(cell, S)
    simulation_cell.hf_type = 'uhf'
    cfg.system.pyscf_cell = simulation_cell

    return cfg