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
from copy import deepcopy


def pre1d():
    m = fe.mesh.Line()
    e = fe.element.Line()
    q = fe.quadrature.GaussLegendre(1, 1)
    r = fe.Region(m, e, q)
    u = fe.Field(r)
    v = fe.FieldContainer([u])
    return v


def pre2d():
    m = fe.mesh.Rectangle()
    e = fe.element.Quad()
    q = fe.quadrature.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=2)
    v = fe.FieldContainer([u])
    return v


def pre3d():
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    v = fe.FieldContainer([u])
    return v


def test_boundary():

    u = pre3d()
    bounds = {"boundary-label": fe.Boundary(u[0])}

    v = fe.dof.apply(u, bounds, dof0=None)
    assert np.allclose(u[0].values.ravel(), v)

    mask = np.ones(u.region.mesh.npoints, dtype=bool)
    bounds = {"boundary-label": fe.Boundary(u[0], mask=mask)}

    v = fe.dof.apply(u, bounds, dof0=None)
    assert np.allclose(u[0].values.ravel(), v)


def test_loadcase():

    for u in [pre1d(), pre2d(), pre3d()]:
        v = fe.FieldContainer([u[0], deepcopy(u[0])])

        ux = fe.dof.uniaxial(u, right=1.0, move=0.2, clamped=False)
        assert len(ux) == 2

        ux = fe.dof.uniaxial(u, right=1.0, move=0.2, clamped=True)
        assert len(ux) == 2
        assert "right" in ux[0]

        ux = fe.dof.uniaxial(u, right=2.0, move=0.2, clamped=True)
        assert len(ux) == 2
        assert "right" in ux[0]

        bx = fe.dof.biaxial(u, right=1.0, move=0.2, clamped=False)
        assert len(bx) == 2

        bx = fe.dof.biaxial(u, right=1.0, move=0.2, clamped=True)
        assert len(bx) == 2
        assert "right" in bx[0]

        bx = fe.dof.biaxial(u, right=2.0, move=0.2, clamped=True)
        assert len(bx) == 2
        assert "right" in bx[0]

        bx = fe.dof.biaxial(v, right=1.0, move=0.2, clamped=True)
        assert len(bx) == 2
        assert "right" in bx[0]

        ps = fe.dof.planar(u, right=1.0, move=0.2, clamped=False)
        assert len(ps) == 2

        ps = fe.dof.planar(u, right=1.0, move=0.2, clamped=True)
        assert len(ps) == 2
        assert "right" in ps[0]

        ps = fe.dof.planar(u, right=2.0, move=0.2, clamped=True)
        assert len(ps) == 2
        assert "right" in ps[0]

        ps = fe.dof.planar(v, right=1.0, move=0.2, clamped=True)
        assert len(ps) == 2
        assert "right" in ps[0]

        sh = fe.dof.shear(u, bottom=0.0, top=1.0, move=0.2, sym=True)
        assert len(sh) == 2
        assert "top" in sh[0]

        sh = fe.dof.shear(v, bottom=0.0, top=1.0, move=0.2, sym=False)
        assert len(sh) == 2
        assert "top" in sh[0]


if __name__ == "__main__":
    test_boundary()
    test_loadcase()
