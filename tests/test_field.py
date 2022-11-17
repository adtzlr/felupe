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


def pre(values=0):

    m = fe.Cube(n=3)
    e = fe.Hexahedron()
    q = fe.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3, values=values)
    v = fe.FieldContainer([u])
    return r, v


def pre_axi():

    m = fe.Rectangle(n=3)
    e = fe.Quad()
    q = fe.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)

    u = fe.FieldAxisymmetric(r)
    v = fe.FieldContainer([u])

    return r, v


def pre_mixed():

    m = fe.Cube(n=3)
    e = fe.Hexahedron()
    q = fe.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)

    u = fe.Field(r, dim=3)
    p = fe.Field(r)
    J = fe.Field(r, values=1)

    f = fe.FieldContainer((u, p, J))
    g = fe.FieldsMixed(fe.RegionHexahedron(m), n=3)

    u.values[0] = np.ones(3)
    assert np.all(f.values()[0][0] == 1)
    assert len(g.fields) == 3

    return r, f, u, p, J


def pre_axi_mixed():

    m = fe.Rectangle(n=3)
    e = fe.Quad()
    q = fe.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)

    u = fe.FieldAxisymmetric(r, dim=2)
    p = fe.Field(r)
    J = fe.Field(r, values=1)

    f = fe.FieldContainer((u, p, J))

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


def test_3d():

    r, u = pre()
    u += u.values()

    with pytest.raises(ValueError):
        r, u = pre(values=np.ones(2))


def test_3d_mixed():

    r, f, u, p, J = pre_mixed()

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


if __name__ == "__main__":
    test_axi()
    test_3d()
    test_3d_mixed()
