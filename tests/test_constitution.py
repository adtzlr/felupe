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
import numpy as np
import felupe as fe


def pre(sym, add_identity):
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    F = u.extract(grad=True, sym=sym, add_identity=add_identity)

    return F


def test_nh():
    F = pre(sym=False, add_identity=True)

    nh = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    P = nh.gradient(F)
    A = nh.hessian(F)

    assert P.shape == (3, 3, *F.shape[-2:])
    assert A.shape == (3, 3, 3, 3, *F.shape[-2:])

    assert np.allclose(P, 0)


def test_linear():
    strain = pre(sym=True, add_identity=False)

    le = fe.constitution.LinearElastic(E=1.0, nu=0.3)

    stress = le.gradient(strain)
    dsde = le.hessian(strain)
    dsde = le.hessian(strain, stress=stress)

    assert stress.shape == (3, 3, *strain.shape[-2:])
    assert dsde.shape == (3, 3, 3, 3, *strain.shape[-2:])

    assert np.allclose(stress, 0)


def test_kinematics():
    F = pre(sym=False, add_identity=True)

    lc = fe.constitution.LineChange()
    ac = fe.constitution.AreaChange()
    vc = fe.constitution.VolumeChange()

    xf = lc.function(F)
    xg = lc.gradient(F)

    yf = ac.function(F)
    yg = ac.gradient(F)

    zf = vc.function(F)
    zg = vc.gradient(F)
    zh = vc.hessian(F)

    assert np.allclose(xf, F)

    assert xf.shape == (3, 3, *F.shape[-2:])
    assert xg.shape == (3, 3, 3, 3, *F.shape[-2:])

    assert yf.shape == (3, 3, *F.shape[-2:])
    assert yg.shape == (3, 3, 3, 3, *F.shape[-2:])

    assert zf.shape == F.shape[-2:]
    assert zg.shape == (3, 3, *F.shape[-2:])
    assert zh.shape == (3, 3, 3, 3, *F.shape[-2:])


if __name__ == "__main__":
    test_nh()
    test_linear()
    test_kinematics()
