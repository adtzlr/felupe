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
import pytest

import felupe as fem


def test_axi():
    m = fem.Rectangle(n=11)
    r = fem.RegionQuad(m)
    f = fem.FieldsMixed(r, n=1, axisymmetric=True)
    u = fem.NeoHooke(mu=1, bulk=1)
    b = fem.SolidBody(u, f)
    loadcase = fem.dof.uniaxial(f, clamped=True)[-1]
    res = fem.newtonrhapson(items=[b], **loadcase)

    u = f[0].interpolate()
    strain = f[0].grad(sym=True)

    assert len(u) == 3
    assert strain.shape[:-2] == (3, 3)

    assert res.success


def test_axi_mixed():
    m = fem.Rectangle(n=6)
    r = fem.RegionQuad(m)
    f = fem.FieldsMixed(r, n=3, axisymmetric=True)
    u = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    b = fem.SolidBody(u, f)
    loadcase = fem.dof.uniaxial(f, clamped=True)[-1]
    res = fem.newtonrhapson(items=[b], **loadcase)

    u = f[0].interpolate()
    strain = f[0].grad(sym=True)

    assert len(u) == 3
    assert strain.shape[:-2] == (3, 3)

    assert res.success


def test_planestrain():
    m = fem.Rectangle(n=6)
    r = fem.RegionQuad(m)

    f = fem.FieldsMixed(r, n=1, planestrain=True)
    g = fem.FieldsMixed(r, n=1, planestrain=False)

    u = fem.LinearElastic(E=1, nu=0.3)
    v = fem.LinearElasticPlaneStrain(E=1, nu=0.3)

    b = fem.SolidBody(u, f)
    c = fem.SolidBody(v, g)

    r = fem.newtonrhapson(items=[b], **fem.dof.uniaxial(f, clamped=True)[-1])
    s = fem.newtonrhapson(items=[c], **fem.dof.uniaxial(f, clamped=True)[-1])

    assert np.allclose(r.x.values(), s.x.values())
    assert np.allclose(r.fun, s.fun)


def test_planestrain_nh():
    m = fem.Rectangle(n=6)
    r = fem.RegionQuad(m)
    f = fem.FieldsMixed(r, n=1, planestrain=True)
    u = fem.NeoHooke(mu=1, bulk=2)
    b = fem.SolidBody(u, f)
    loadcase = fem.dof.uniaxial(f, clamped=True)[-1]
    res = fem.newtonrhapson(items=[b], **loadcase)

    u = f[0].interpolate()
    strain = f[0].grad(sym=True)

    assert len(u) == 3
    assert strain.shape[:-2] == (3, 3)

    assert res.success


def test_planestrain_nh_mixed():
    m = fem.Rectangle(n=6)
    r = fem.RegionQuad(m)
    f = fem.FieldsMixed(r, n=3, planestrain=True)
    u = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    b = fem.SolidBody(u, f)
    loadcase = fem.dof.uniaxial(f, clamped=True)[-1]
    res = fem.newtonrhapson(items=[b], **loadcase)

    u = f[0].interpolate()
    strain = f[0].grad(sym=True)

    assert len(u) == 3
    assert strain.shape[:-2] == (3, 3)

    assert res.success


if __name__ == "__main__":
    test_axi()
    test_axi_mixed()
    test_planestrain()
    test_planestrain_nh()
    test_planestrain_nh_mixed()
