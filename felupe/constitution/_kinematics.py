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

from ..math import (
    transpose,
    inv,
    dya,
    cdya_ik,
    cdya_il,
    det,
    identity,
)


class LineChange:
    def __init__(self):
        pass

    def function(self, F):
        "line change dx = F dX"
        return F

    def gradient(self, F):
        "gradient of line change"
        Eye = identity(F)
        return cdya_ik(Eye, Eye)


class AreaChange:
    def __init__(self):
        pass

    def function(self, F):
        "area change da = J F^(-T) dA"
        J = det(F)
        return J * transpose(inv(F, J))

    def gradient(self, F):
        "gradient of area change"
        J = det(F)
        dJdF = self.function(F)
        return (dya(dJdF, dJdF) - cdya_il(dJdF, dJdF)) / J


class VolumeChange:
    def __init__(self):
        pass

    def function(self, F):
        "volume change dv = J dV"
        return det(F)

    def gradient(self, F):
        "gradient of volume change"
        J = self.function(F)
        return J * transpose(inv(F, J))

    def hessian(self, F):
        "hessian of volume change"
        J = self.function(F)
        dJdF = self.gradient(F)
        return (dya(dJdF, dJdF) - cdya_il(dJdF, dJdF)) / J
