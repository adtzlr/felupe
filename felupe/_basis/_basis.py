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


class Basis:
    r"""A basis and its gradient built on top of a scalar- or vector-valued 
    field. The first two indices are used for looping over the element shape
    functions ``a`` and its components ``i``. The third index represents the
    vector component ``k`` of the field. The two trailing axes ``(p, c)`` 
    contain the evaluated element shape functions at quadrature points per 
    cell. For gradients, the fourth index is used for the vector component of
    the partial derivative ``k``.
    
    ..  math::
        
        \text{grad}(b)_{aijkpe} = \delta_{ij} 
            \left( \frac{\partial h_a}{\partial X_K} \right)_{pc}
        
    Parameters
    ----------
    field : Field
        A field on which the basis should be created.
    
    Attributes
    ----------
    basis : ndarray
        The evaluated basis.
    grad : ndarray
        The evaluated gradient of the basis.
    
    """
    def __init__(self, field):
        
        self.field = field
        
        self.basis = np.einsum(
            "ij,ap,c->aijpc", 
            np.eye(self.field.region.element.dim), 
            self.field.region.h, 
            np.ones(self.field.region.mesh.ncells)
        )
        
        if hasattr(self.field.region, "dhdX"):
            self.grad = np.einsum(
                "ij,akpc->aijkpc", 
                np.eye(self.field.dim), 
                self.field.region.dhdX
            )
            
        else:
            self.grad = None