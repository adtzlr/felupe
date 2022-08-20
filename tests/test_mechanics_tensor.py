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

import pytest
import felupe as fe
import numpy as np


def pre_umat():

    NH = fe.NeoHooke(mu=1, bulk=5000)

    class NHTensor:
        def __init__(self, NH):
            self.NH = NH
            self.x = [np.zeros((3, 3)), np.zeros((1, 1))]

        def function(self, x):
            F, statevars = x[:-1], x[-1]
            # return dummy state variables along with stress
            return [*self.NH.gradient(F), statevars]

        def gradient(self, x):
            F, statevars = x[:-1], x[-1]
            return self.NH.hessian(F)

    return NHTensor(NH)


def pre_umat_mixed():

    NH = fe.ThreeFieldVariation(fe.NeoHooke(mu=1, bulk=5000))

    class NHTensor:
        def __init__(self, NH):
            self.NH = NH
            self.x = [
                np.zeros((3, 3)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
            ]

        def function(self, x):
            F, statevars = x[:-1], x[-1]
            # return dummy state variables along with stress
            return [*self.NH.gradient(F), statevars]

        def gradient(self, x):
            F, statevars = x[:-1], x[-1]
            return self.NH.hessian(F)

    return NHTensor(NH)


def test_solidbody_tensor():

    umat = pre_umat()

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    v = fe.FieldsMixed(r, n=1)

    sv = np.zeros((1, 1, r.quadrature.npoints, m.ncells))

    for statevars in [sv, None]:

        b = fe.SolidBodyTensor(umat, v, statevars)
        r = b.assemble.vector()

        K = b.assemble.matrix(v)
        K = b.assemble.matrix()
        r = b.assemble.vector(v)
        r = b.assemble.vector()
        F = b.results.kinematics
        P = b.results.stress
        s = b.evaluate.cauchy_stress()
        t = b.evaluate.kirchhoff_stress()
        C = b.results.elasticity
        z = b.results.statevars

        assert K.shape == (81, 81)
        assert r.shape == (81, 1)
        assert F[0].shape == (3, 3, 8, 8)
        assert P[0].shape == (3, 3, 8, 8)
        assert s.shape == (3, 3, 8, 8)
        assert t.shape == (3, 3, 8, 8)
        assert C[0].shape == (3, 3, 3, 3, 8, 8)
        assert z.shape == (1, 1, 8, 8)


def test_solidbody_tensor_mixed():

    umat = pre_umat_mixed()

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    v = fe.FieldsMixed(r, n=3)

    sv = np.zeros((1, 1, r.quadrature.npoints, m.ncells))

    for statevars in [sv, None]:

        b = fe.SolidBodyTensor(umat, v, statevars)
        r = b.assemble.vector()

        K = b.assemble.matrix(v)
        K = b.assemble.matrix()
        r = b.assemble.vector(v)
        r = b.assemble.vector()
        F = b.results.kinematics
        P = b.results.stress
        s = b.evaluate.cauchy_stress()
        t = b.evaluate.kirchhoff_stress()
        C = b.results.elasticity
        z = b.results.statevars

        assert K.shape == (97, 97)
        assert r.shape == (97, 1)
        assert F[0].shape == (3, 3, 8, 8)
        assert P[0].shape == (3, 3, 8, 8)
        assert s.shape == (3, 3, 8, 8)
        assert t.shape == (3, 3, 8, 8)
        assert C[0].shape == (3, 3, 3, 3, 8, 8)
        assert z.shape == (1, 1, 8, 8)


if __name__ == "__main__":
    test_solidbody_tensor()
    test_solidbody_tensor_mixed()
