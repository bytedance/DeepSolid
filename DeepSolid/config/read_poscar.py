# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from DeepSolid import base_config
from DeepSolid import supercell
from DeepSolid.utils import poscar_to_cell
import numpy as np


def get_config(input_str):
    poscar_path, S, basis = input_str.split(',')
    cell = poscar_to_cell.read_poscar(poscar_path)
    S = int(S)
    S = np.diag([S, S, S])
    cell.verbose = 5
    cell.basis = basis
    cell.exp_to_discard = 0.1
    cell.build()
    cfg = base_config.default()

    # Set up cell

    simulation_cell = supercell.get_supercell(cell, S)
    if cell.spin != 0:
        simulation_cell.hf_type = 'uhf'
    cfg.system.pyscf_cell = simulation_cell

    return cfg