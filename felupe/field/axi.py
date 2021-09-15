# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:29:15 2021

@author: adutz
"""
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
from .base import Field


class FieldAxisymmetric(Field):
    """Axisymmetric field with
    * component 1 ...  axial component
    * component 2 ... radial component
    
    This is a modified Field class in which the radial coordinates
    are evaluated at the quadrature points. The `grad`-function is
    modified in such a way that it does not only contain the in-plane
    2d-gradient but also the circumferential stretch 
    as shown in Eq.(1).
    
                  |  dudX(2d) :   0   |
      dudX(axi) = | ..................|                  (1)
                  |     0     : u_r/R |
                  
    """

    def __init__(self, region, dim=2, values=0):
        super().__init__(region, dim=dim, values=values)
        self.scalar = Field(region, dim=1, values=region.mesh.points[:, 1])
        self.radius = self.scalar.interpolate()

    def _grad_2d(self):
        "In-plane 2d-gradient dudX_IJpe."
        # gradient as partial derivative of point values "aI"
        # w.r.t. undeformed coordinate "J" evaluated at quadrature point "p"
        # for cell "e"
        return np.einsum(
            "ea...,aJpe->...Jpe", self.values[self.region.mesh.cells], self.region.dhdX,
        )

    def grad(self):
        "Full 3d-gradient dudX_IJpe."
        g = np.pad(self._grad_2d(), ((0, 1), (0, 1), (0, 0), (0, 0)))
        g[-1, -1] = self.interpolate()[1] / self.radius
        return g
