# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf.pbc import gto
import numpy as np

# Define your test system
nk = 1
cell = gto.Cell()
L = 2 / 0.529177
cell.atom = f"""
Li 0 0 0
H {L/2} {L/2} {L/2}
"""
cell.basis = "sto-3g"
cell.a = (1 - np.eye(3)) * L / 2
cell.unit = "B"
cell.verbose = 5
cell.spin = 0
cell.exp_to_discard = 0.1
cell.build()
