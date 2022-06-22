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

from matadi.models import displacement_pressure_split
from matadi import MaterialTensor, Variable
from matadi.math import trace, det, log, gradient

def pre_umat():
    
    F = Variable("F", 3, 3)
    z = Variable("z", 5, 16)

    def fun(x, C10=0.5, bulk=50):
        """Compressible Neo-Hookean material model formulation
        with some random (unused) state variables."""

        F, z = x[0], x[-1]
        
        J = det(F)
        C = F.T @ F
        I1 = trace(C)
        
        W = C10 * (I1 - 3) - 2 * C10 * log(J) + bulk * (J - 1) ** 2 / 2
        
        return gradient(W, F), z

    NH = MaterialTensor(x=[F, z], fun=fun, triu=True, statevars=1)
    
    return fe.MatadiMaterial(NH)


def test_simple_tensor():

    umat = pre_umat()

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    u = fe.Field(r, dim=3)
    
    sv = np.zeros((5, 16, r.quadrature.npoints, m.ncells))

    b = fe.SolidBodyTensor(umat, u, sv)
    r = b.assemble.vector()

    K = b.assemble.matrix()
    r = b.assemble.vector(u)
    F = b.results.kinematics[0]
    P = b.results.stress[0]
    s = b.evaluate.cauchy_stress()
    t = b.evaluate.kirchhoff_stress()
    C = b.results.elasticity
    z = b.results.statevars

    assert K.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F.shape == (3, 3, 8, 8)
    assert P.shape == (3, 3, 8, 8)
    assert s.shape == (3, 3, 8, 8)
    assert t.shape == (3, 3, 8, 8)
    assert C.shape == (3, 3, 3, 3, 8, 8)
    assert z.shape == (5, 16, 8, 8)

if __name__ == "__main__":
    test_simple_tensor()
