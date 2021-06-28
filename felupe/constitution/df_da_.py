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

from types import SimpleNamespace

from ..math import (
    dot,
    ddot,
    ddot44,
    ddot444,
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


class Composite:
    def __init__(self, *args):

        self.materials = args
        self.kind = SimpleNamespace(**{"df": None, "da": None})

    def stress(self, *args, **kwargs):
        return np.sum([m.stress(*args) for m in self.materials], 0)

    def elasticity(self, *args, **kwargs):
        return np.sum([m.elasticity(*args, **kwargs) for m in self.materials], 0)


class Material:
    def __init__(self, stress, elasticity):

        self.stress = stress
        self.elasticity = elasticity
        self.kind = SimpleNamespace(**{"df": None, "da": None})


class Hydrostatic:
    def __init__(self, bulk):
        self.bulk = bulk
        self.kind = SimpleNamespace(**{"df": None, "da": None})

    def dUdJ(self, J):
        return self.bulk * (J - 1)

    def d2UdJdJ(self, J):
        return self.bulk

    def stress(self, F, J, b, invb):
        return self.dUdJ(J) * J * identity(b)

    def elasticity(self, F, J, b, invb):
        eye = identity(b)
        p = self.dUdJ(J)
        q = p + self.d2UdJdJ(J) * J
        return J * (q * dya(eye, eye) - 2 * p * cdya(eye, eye))


class AsIsochoric:
    def __init__(self, material_isochoric):
        self.isochoric = material_isochoric
        self.kind = SimpleNamespace(**{"df": None, "da": None})

    def stress(self, F, J, b, invb):
        bu = J ** (-2 / 3) * b
        tb = self.isochoric.stress(bu)
        return dev(tb)

    def elasticity(self, F, J, b, invb):
        eye = identity(b)
        p4 = cdya(eye, eye) - dya(eye, eye) / 3

        bu = J ** (-2 / 3) * b
        tb = self.isochoric.stress(bu)

        Jc4b = self.isochoric.elasticity(bu)
        if np.all(Jc4b == 0):
            PJc4bP = Jc4b
        else:
            PJc4bP = ddot444(p4, Jc4b, p4)

        return (
            PJc4bP
            - 2 / 3 * (dya(tb, eye) + dya(eye, tb))
            + 2 / 9 * trace(tb) * dya(eye, eye)
            + 2 / 3 * trace(tb) * cdya(eye, eye)
        )
