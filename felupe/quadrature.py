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

import quadpy

class Constant:
    
    def __init__(self, dim=3):
        # integration point weights and coordinates
        if dim == 3:
            self.scheme = _P0_D3()
        elif dim == 2:
            self.scheme = _P0_D2()
        elif dim == 1:
            self.scheme = _P0_D1()
        self.npoints = self.scheme.npoints
        self.weights = self.scheme.weights
        self.points = self.scheme.points
    
class _P0_D3:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 1
        self.weights = np.ones(1)
        self.points = np.array([[ 0, 0, 0]])

class _P0_D2:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 1
        self.weights = np.ones(1)
        self.points = np.array([[ 0, 0]])

class _P0_D1:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 1
        self.weights = np.ones(1)
        self.points = np.array([[ 0]])

class Linear:
    
    def __init__(self, dim=3):
        # integration point weights and coordinates
        if dim == 3:
            self.scheme = _P1_D3()
        elif dim == 2:
            self.scheme = _P1_D2()
        elif dim == 1:
            self.scheme = _P1_D1()
        self.npoints = self.scheme.npoints
        self.weights = self.scheme.weights
        self.points = self.scheme.points
    
class _P1_D3:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 2**3
        self.weights = np.ones(8)
        self.points = np.array([[-1,-1,-1],
                                [ 1,-1,-1],
                                [ 1, 1,-1],
                                [-1, 1,-1],
                                [-1,-1, 1],
                                [ 1,-1, 1],
                                [ 1, 1, 1],
                                [-1, 1, 1]]) * np.sqrt(1/3)

class _P1_D2:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 2**2
        self.weights = np.ones(4)
        self.points = np.array([[-1,-1],
                                [ 1,-1],
                                [ 1, 1],
                                [-1, 1]]) * np.sqrt(1/3)

class _P1_D1:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 2
        self.weights = np.ones(2)
        self.points = np.array([[-1],
                                [ 1]]) * np.sqrt(1/3)
        
        
class Quadratic:
    
    def __init__(self, dim=3):
        # integration point weights and coordinates
        if dim == 3:
            self.scheme = _P2_D3()
        elif dim == 2:
            self.scheme = _P2_D2()
        elif dim == 1:
            self.scheme = _P2_D1()
        self.npoints = self.scheme.npoints
        self.weights = self.scheme.weights
        self.points = self.scheme.points
    
class _P2_D3:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 3**3

        scheme = quadpy.c3.product(quadpy.c1.gauss_legendre(3))
        
        self.points = scheme.points.T
        self.weights = scheme.weights

class _P2_D2:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 3**2
        
        scheme = quadpy.c2.product(quadpy.c1.gauss_legendre(3))
        
        self.points = scheme.points.T
        self.weights = scheme.weights

class _P2_D1:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 3

        scheme = quadpy.c1.gauss_legendre(3)
        
        self.points = scheme.points.T
        self.weights = scheme.weights
        
class Cubic:
    
    def __init__(self, dim=3):
        # integration point weights and coordinates
        if dim == 3:
            self.scheme = _P3_D3()
        elif dim == 2:
            self.scheme = _P3_D2()
        elif dim == 1:
            self.scheme = _P3_D1()
        self.npoints = self.scheme.npoints
        self.weights = self.scheme.weights
        self.points = self.scheme.points
    
class _P3_D3:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 4**3
        
        scheme = quadpy.c3.product(quadpy.c1.gauss_legendre(4))
        
        self.points = scheme.points.T
        self.weights = scheme.weights

class _P3_D2:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 4**2
        
        scheme = quadpy.c2.product(quadpy.c1.gauss_legendre(4))
        
        self.points = scheme.points.T
        self.weights = scheme.weights

class _P3_D1:
    
    def __init__(self):
        # integration point weights and coordinates
        self.npoints = 4
        
        scheme = quadpy.c1.gauss_legendre(4)
        
        self.points = scheme.points.T
        self.weights = scheme.weights