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


def test_dof():

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(1, 3)

    region = fe.Region(mesh, element, quadrature)
    field = fe.Field(region, dim=3)

    dof = np.arange(mesh.ndof)

    f0 = lambda x: x == 0
    f1 = lambda x: x == 1

    v = 13.2

    bounds = fe.doftools.symmetry(field, axes=(0, 1, 0))
    bounds["left"] = fe.doftools.Boundary(field, fx=f0, skip=(0, 0, 1), value=v)
    bounds["right"] = fe.doftools.Boundary(field, fx=f1, skip=(0, 1, 1))

    dof0, dof1 = fe.doftools.partition(field, bounds)

    uext = fe.doftools.apply(field, bounds, dof0=None)
    u0ext = fe.doftools.apply(field, bounds, dof0=dof0)

    dof01 = np.sort(np.append(dof0, dof1))

    assert np.all(np.equal(dof, dof01))
    assert np.allclose(uext.ravel()[dof0], u0ext)
    assert np.allclose(uext.ravel()[bounds["left"].dof], v)


def test_dof_extend():

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(1, 3)

    mesh0 = fe.mesh.convert(mesh, order=0)
    element0 = fe.element.ConstantHexahedron()

    region = fe.Region(mesh, element, quadrature)
    region0 = fe.Region(mesh0, element0, quadrature)

    field = fe.Field(region, dim=3)
    field0 = fe.Field(region0, dim=1)

    fields = (field, field0)

    f0 = lambda x: x == 0
    f1 = lambda x: x == 1

    v = 13.2

    bounds = fe.doftools.symmetry(field, axes=(0, 1, 0))
    bounds["left"] = fe.doftools.Boundary(field, fx=f0, skip=(0, 0, 1), value=v)
    bounds["right"] = fe.doftools.Boundary(field, fx=f1, skip=(0, 1, 1))

    dof0, dof1, unstack = fe.doftools.partition(fields, bounds)


if __name__ == "__main__":
    test_dof()
    test_dof_extend()
