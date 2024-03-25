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


def pre(values=0):
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3, values=values)
    v = fem.FieldContainer([u])

    return r, v


def pre_axi():
    m = fem.Rectangle(n=3)
    e = fem.Quad()
    q = fem.GaussLegendre(1, 2)
    r = fem.Region(m, e, q)

    u = fem.FieldAxisymmetric(r)
    v = fem.FieldContainer([u])

    print(m), print(r), print(v)

    return r, v


def pre_mixed():
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)
    J = fem.Field(r, values=1)

    f = fem.FieldContainer([u, p, J])
    g = fem.FieldsMixed(fem.RegionHexahedron(m), n=3, disconnect=True)
    f2 = u & p & J
    f3 = (u & p) & J
    f4 = u & (p & J)
    f5 = u.as_container() & (p & J)
    assert [np.allclose(fi, f2i) for fi, f2i in zip(f.extract(), f2.extract())]
    assert [np.allclose(fi, f3i) for fi, f3i in zip(f.extract(), f3.extract())]
    assert [np.allclose(fi, f4i) for fi, f4i in zip(f.extract(), f4.extract())]
    assert [np.allclose(fi, f5i) for fi, f5i in zip(f.extract(), f5.extract())]

    f & None, u & None

    print(m), print(r), print(f)

    u.values[0] = np.ones(3)
    assert np.all(f.values()[0][0] == 1)
    assert len(g.fields) == 3

    fem.Field(r, dim=9, values=np.eye(3))

    return r, f, u, p, J


def pre_axi_mixed():
    m = fem.Rectangle(n=3)
    e = fem.Quad()
    q = fem.GaussLegendre(1, 2)
    r = fem.Region(m, e, q)

    u = fem.FieldAxisymmetric(r, dim=2)
    p = fem.Field(r)
    J = fem.Field(r, values=1)

    f = fem.FieldContainer((u, p, J))

    region = fem.RegionQuad(m)
    fem.FieldsMixed(region, axisymmetric=True)
    fem.FieldsMixed(region, planestrain=True)
    with pytest.raises(ValueError):
        fem.FieldsMixed(region, axisymmetric=True, planestrain=True)

    u.values[0] = np.ones(2)
    assert np.all(f.values()[0][0] == 1)

    return r, f, u, p, J


def test_axi():
    r, u = pre_axi()
    u += u.values()

    r, f, u, p, J = pre_axi_mixed()

    u.extract()
    u.extract(grad=False)
    u.extract(grad=True, sym=True)
    u.extract(grad=True, add_identity=False)


def test_mixed_lagrange():
    order = 4

    m = fem.Cube(n=order + 1)
    md = fem.Cube(n=order)

    m.update(
        cells=np.arange(m.npoints).reshape(1, -1), cell_type="VTK_LAGRANGE_HEXAHEDRON"
    )
    md.update(
        cells=np.arange(md.npoints).reshape(1, -1), cell_type="VTK_LAGRANGE_HEXAHEDRON"
    )

    m = fem.mesh.CubeArbitraryOrderHexahedron(order=order)
    md = fem.mesh.CubeArbitraryOrderHexahedron(order=order - 1)

    r = fem.RegionLagrange(m, order=order, dim=3)
    g = fem.FieldsMixed(r, mesh=md)

    assert len(g.fields) == 3


def test_3d():
    r, u = pre()
    u += u.values()

    with pytest.raises(ValueError):
        r, u = pre(values=np.ones(2))


def test_3d_mixed():
    r, f, u, p, J = pre_mixed()

    f.evaluate.deformation_gradient()
    f.evaluate.strain()
    f.evaluate.log_strain()
    f.evaluate.green_lagrange_strain()

    f.extract()
    f.extract(grad=False)
    f.extract(grad=(False,))
    f.extract(grad=True, sym=True)
    f.extract(grad=True, add_identity=False)

    u.extract()
    u.extract(grad=False)
    u.extract(grad=True, sym=True)
    u.extract(grad=True, add_identity=False)

    J.fill(1.0)

    u + u.values
    u - u.values
    u * u.values
    J / J.values

    u + u
    u - u
    u * u
    J / J

    J /= J.values
    J /= J

    J *= J.values
    J += J.values
    J -= J.values

    J *= J
    J += J
    J -= J

    dof = [0, 1]
    u[dof]
    f[0][dof]

    u.values.fill(1)
    p.values.fill(1)
    J.values.fill(1)

    df = [u.values.copy(), p.values.copy(), J.values.copy()]

    f + df
    f - df
    f * df
    f / df

    f += df
    f -= df
    f *= df
    f /= df

    df_1d = np.concatenate([dfi.ravel() for dfi in df])

    f + df_1d
    f - df_1d
    f * df_1d
    f / df_1d

    f += df_1d
    f -= df_1d
    f *= df_1d
    f /= df_1d

    v = u.copy()
    g = f.copy()

    assert np.allclose(v.values, u.values)
    assert np.allclose(g[0].values, f[0].values)


def test_view():
    mesh = fem.Rectangle(n=6)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    plotter = field.plot(off_screen=True)
    # img = mesh.screenshot(transparent_background=True)
    # ax = mesh.imshow()


if __name__ == "__main__":
    test_axi()
    test_3d()
    test_3d_mixed()
    test_mixed_lagrange()
    test_view()
