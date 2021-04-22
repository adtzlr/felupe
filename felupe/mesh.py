# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import meshzoo


class Cube:
    def __init__(self, a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
        self.a = a
        self.b = b
        self.n = n

        self.nodes, self.connectivity = meshzoo.cube_hexa(a, b, n)
        self.nnodes, self.ndim = self.nodes.shape
        self.ndof = self.nodes.size
        self.nelements = self.connectivity.shape[0]

        _, self.elements_per_node = np.unique(self.connectivity, return_counts=True)


class Rectangle:
    def __init__(self, a=(0, 0), b=(1, 1), n=(2, 2)):
        self.a = a
        self.b = b
        self.n = n

        self.nodes, self.connectivity = meshzoo.rectangle_quad(a, b, n)
        self.nnodes, self.ndim = self.nodes.shape
        self.ndof = self.nodes.size
        self.nelements = self.connectivity.shape[0]

        _, self.elements_per_node = np.unique(self.connectivity, return_counts=True)
