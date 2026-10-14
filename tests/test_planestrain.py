# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import felupe as fem


def test_axi():
    m = fem.Rectangle(n=11)
    r = fem.RegionQuad(m)
    f = fem.FieldsMixed(r, n=1, axisymmetric=True)
    u = fem.NeoHooke(mu=1, bulk=1)
    b = fem.SolidBody(u, f)
    loadcase = fem.dof.uniaxial(f, clamped=True, return_loadcase=True)[-1]
    res = fem.newtonraphson(items=[b], **loadcase)

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
    loadcase = fem.dof.uniaxial(f, clamped=True, return_loadcase=True)[-1]
    res = fem.newtonraphson(items=[b], **loadcase)

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
    v = fem.constitution.LinearElasticPlaneStrain(E=1, nu=0.3)

    b = fem.SolidBody(u, f)
    c = fem.SolidBody(v, g)

    kwargs = dict(clamped=True, return_loadcase=True)
    r = fem.newtonraphson(items=[b], **fem.dof.uniaxial(f, **kwargs)[-1])
    s = fem.newtonraphson(items=[c], **fem.dof.uniaxial(f, **kwargs)[-1])

    assert np.allclose(r.x.values(), s.x.values())
    assert np.allclose(r.fun, s.fun)


def test_planestrain_nh():
    m = fem.Rectangle(n=6)
    r = fem.RegionQuad(m)
    f = fem.FieldsMixed(r, n=1, planestrain=True)
    u = fem.NeoHooke(mu=1, bulk=2)
    b = fem.SolidBody(u, f)
    loadcase = fem.dof.uniaxial(f, clamped=True, return_loadcase=True)[-1]
    res = fem.newtonraphson(items=[b], **loadcase)

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
    loadcase = fem.dof.uniaxial(f, clamped=True, return_loadcase=True)[-1]
    res = fem.newtonraphson(items=[b], **loadcase)

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
