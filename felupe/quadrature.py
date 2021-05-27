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

class Constant:
    
    def __init__(self, dim=3):
        # integration point weights and coordinates
        self.dim = dim
        self.npoints = 1
        self.weights = np.ones(1)
        if dim == 3:
            self.points = np.array([[ 0, 0, 0]])
        elif dim == 2:
            self.points = np.array([[ 0, 0]])
        elif dim == 1:
            self.points = np.array([[ 0]])


class Linear:
    
    def __init__(self, dim=3):
        # integration point weights and coordinates
        self.dim = dim
        self.npoints = 2**dim
        self.weights = np.ones(self.npoints)
        if dim == 3:
            self.points = np.array([[-1,-1,-1],
                                    [ 1,-1,-1],
                                    [ 1, 1,-1],
                                    [-1, 1,-1],
                                    [-1,-1, 1],
                                    [ 1,-1, 1],
                                    [ 1, 1, 1],
                                    [-1, 1, 1]]) * np.sqrt(1/3)
        elif dim == 2:
            self.points = np.array([[-1,-1],
                                    [ 1,-1],
                                    [ 1, 1],
                                    [-1, 1]]) * np.sqrt(1/3)
        elif dim == 1:
            self.points = np.array([[-1],
                                    [ 1]]) * np.sqrt(1/3)