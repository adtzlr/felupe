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


class UserMaterial:
    """A user-defined material definition with given functions for the (first
    Piola-Kirchhoff) stress tensor and the according fourth-order elasticity
    tensor. Both functions take a list of the deformation gradient and optional
    state variables as the first input argument. The stress-function also
    returns the updated state variables.

    Take this code-block as template:

    ..  code-block::

        def stress(x, **kwargs):
            "First Piola-Kirchhoff stress tensor."

            # extract variables
            F, statevars = x[0], x[-1]

            # user code for (first Piola-Kirchhoff) stress tensor
            P = None

            # update state variables
            statevars_new = None

            return [P, statevars_new]

        def elasticity(x, **kwargs):
            "Fourth-order elasticity tensor."

            # extract variables
            F, statevars = x[0], x[-1]

            # user code for fourth-order elasticity tensor
            # according to the (first Piola-Kirchhoff) stress tensor
            dPdF = None

            return [dPdF]

        umat = UserMaterial(stress, elasticity, **kwargs)

    """

    def __init__(self, stress, elasticity, nstatevars=0, **kwargs):

        self.umat = {"stress": stress, "elasticity": elasticity}
        self.kwargs = kwargs
        self.x = [np.eye(3), np.zeros(nstatevars)]

    def gradient(self, x):
        return self.umat["stress"](x, **self.kwargs)

    def hessian(self, x):
        return self.umat["elasticity"](x, **self.kwargs)
