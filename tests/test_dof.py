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
from copy import deepcopy

import numpy as np
import pytest

import felupe as fem


def pre1d():
    m = fem.mesh.Line()
    e = fem.element.Line()
    q = fem.quadrature.GaussLegendre(1, 1)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=1)
    v = fem.FieldContainer([u])
    return v


def pre2d():
    m = fem.mesh.Rectangle()
    e = fem.element.Quad()
    q = fem.quadrature.GaussLegendre(1, 2)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=2)
    v = fem.FieldContainer([u])
    return v


def pre3d():
    m = fem.mesh.Cube()
    e = fem.element.Hexahedron()
    q = fem.quadrature.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=2)
    v = fem.FieldContainer([u])
    return v


def test_boundary():
    u = pre3d()
    bounds = {"boundary-label": fem.Boundary(u[0])}

    v = fem.dof.apply(u, bounds, dof0=None)
    assert np.allclose(u[0].values.ravel(), v)

    mask = np.ones(u.region.mesh.npoints, dtype=bool)
    bounds = {"boundary-label": fem.Boundary(u[0], mask=mask)}

    mask = np.ones((u.region.mesh.npoints, u[0].dim), dtype=bool)
    bounds = {"boundary-label": fem.Boundary(u[0], mask=mask)}

    v = fem.dof.apply(u, bounds, dof0=None)
    assert np.allclose(u[0].values.ravel(), v)

    with pytest.raises(ValueError):
        wrong_mask = np.ones((u.region.mesh.npoints, 4), dtype=bool)
        bounds = {"boundary-label": fem.Boundary(u[0], mask=wrong_mask)}


def test_loadcase():
    ux = fem.dof.uniaxial(pre1d())
    assert len(ux) == 2

    for u in [pre2d(), pre3d()]:
        v = fem.FieldContainer([u[0], deepcopy(u[0])])

        ux = fem.dof.uniaxial(u, right=1.0, move=0.2, clamped=False, sym=True)
        assert len(ux) == 2

        ux = fem.dof.uniaxial(u, right=1.0, move=0.2, clamped=True, sym=False)
        assert len(ux) == 2
        assert "right" in ux[0]

        ux = fem.dof.uniaxial(
            u, right=1.0, move=0.2, clamped=True, axis=1, sym=(False, True, False)
        )
        assert len(ux) == 2
        assert "right" in ux[0]

        ux = fem.dof.uniaxial(
            u, right=1.0, move=0.2, clamped=True, axis=1, sym=(True, False, True)
        )
        assert len(ux) == 2
        assert "right" in ux[0]

        ux = fem.dof.uniaxial(u, right=None, move=0.2, clamped=True)
        assert len(ux) == 2
        assert "right" in ux[0]

        bx = fem.dof.biaxial(
            u, rights=(1.0, 1.0), moves=(0.2, 0.2), clampes=(False, False)
        )
        assert len(bx) == 2

        bx = fem.dof.biaxial(
            u, rights=(1.0, 1.0), moves=(0.2, 0.2), clampes=(True, True), sym=False
        )
        assert len(bx) == 2
        assert "left-0" in bx[0]

        bx = fem.dof.biaxial(
            u, rights=(None, None), moves=(0.2, 0.2), clampes=(True, True)
        )
        assert len(bx) == 2
        assert "right-0" in bx[0]

        bx = fem.dof.biaxial(
            v, rights=(1.0, 1.0), moves=(0.2, 0.2), clampes=(True, True)
        )
        assert len(bx) == 2
        assert "right-0" in bx[0]

        sh = fem.dof.shear(u, bottom=0.0, top=1.0, moves=(0.2, 0, 0), sym=True)
        assert len(sh) == 2
        assert "top" in sh[0]

        sh = fem.dof.shear(v, bottom=None, top=None, moves=(0.2, 0, 0), sym=False)
        assert len(sh) == 2
        assert "top" in sh[0]


if __name__ == "__main__":
    test_boundary()
    test_loadcase()
