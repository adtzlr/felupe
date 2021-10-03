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

from ..math import (
    dot,
    ddot,
    transpose,
    majortranspose,
    inv,
    dya,
    cdya,
    cdya_ik,
    cdya_il,
    det,
    identity,
    trace,
    dev,
)


class LineChange:
    def __init__(self):
        pass

    def function(self, F):
        return F

    def gradient(self, F):
        Eye = identity(F)
        return cdya_ik(Eye, Eye)


class AreaChange:
    def __init__(self):
        pass

    def function(self, F):
        J = det(F)
        return J * transpose(inv(F, J))

    def gradient(self, F):
        J = det(F)
        dJdF = self.function(F)
        return (dya(dJdF, dJdF) - cdya_il(dJdF, dJdF)) / J


class VolumeChange:
    def __init__(self):
        pass

    def function(self, F):
        return det(F)

    def gradient(self, F):
        J = self.function(F)
        return J * transpose(inv(F, J))

    def hessian(self, F):
        J = self.function(F)
        dJdF = self.gradient(F)
        return (dya(dJdF, dJdF) - cdya_il(dJdF, dJdF)) / J
