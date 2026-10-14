# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from .. import solve as solvetools


def solve(K, f, field, dof0, dof1, offsets, ext0):
    "Solve linear equation system K dx = b"
    system = solvetools.partition(field, K, dof1, dof0, -f)
    dfields = np.split(solvetools.solve(*system, ext0), offsets)
    return dfields
