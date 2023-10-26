# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np


class Indices:
    def __init__(self, cai, ai, region, dim):
        """Indices for cell "c", point "a" and component "i"."""

        self.cai = cai
        self.ai = ai
        self.dof = np.arange(region.mesh.npoints * dim).reshape(-1, dim)
        self.shape = (region.mesh.npoints * dim, 1)
