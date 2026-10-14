# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


class Indices:
    def __init__(self, cai, ai, region, dim):
        """Indices for cell "c", point "a" and component "i"."""

        self.cai = cai
        self.ai = ai
        self.dof = np.arange(region.mesh.npoints * dim).reshape(-1, dim)
        self.shape = (region.mesh.npoints * dim, 1)
