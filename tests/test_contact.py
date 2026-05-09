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


def pre_contact_mixed(point, values):
    mesh = fem.mesh.Cube(n=3)
    mesh.points = np.vstack((mesh.points, point))
    mesh.update(cells=mesh.cells)

    region = fem.RegionHexahedron(mesh)
    dV = region.dV

    fields = fem.FieldsMixed(region, n=3)
    fields[0].values[-1] = values
    F, p, J = fields.extract()

    nh = fem.NeoHooke(mu=1.0, bulk=2.0)
    umat = fem.ThreeFieldVariation(nh)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fem.Boundary(fields[0], fx=f0)
    boundaries["right"] = fem.Boundary(fields[0], fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fem.Boundary(fields[0], fx=f2, skip=(0, 1, 1), value=0.5)

    bnd = fem.Boundary(fields[0], fx=f1).points
    cpoint = -1

    CONT = fem.ContactRigidPlane(
        fields, points=bnd, centerpoint=cpoint, normal=[1, 0, 0], friction=np.inf
    )

    CONT = fem.ContactRigidPlane(
        fields, points=f1(mesh.x), centerpoint=cpoint, normal=[1, 0, 0], friction=np.inf
    )

    try:
        CONT.plot()
    except ModuleNotFoundError:
        pass

    for f in [None, fields]:
        K_CONT = CONT.assemble.matrix(f)
        r_CONT = CONT.assemble.vector(f)

    linearform = fem.IntegralForm(umat.gradient([F, p, J])[:-1], fields, dV)
    r = linearform.assemble()

    r_CONT.resize(*r.shape)
    r = r + r_CONT

    bilinearform = fem.IntegralForm(umat.hessian([F, p, J]), fields, dV, fields)
    K = bilinearform.assemble()

    K_CONT.resize(*K.shape)
    K = K + K_CONT


def test_contact_mixed():
    pre_contact_mixed(point=[2, 0, 0], values=[0, 0, 0])
    pre_contact_mixed(point=[2, 0, 0], values=[-5, 0, 0])


def test_contact_isolated():
    mesh = fem.mesh.Line(n=3)
    mesh.update(cells=mesh.cells[:1])
    mesh.points = np.pad(mesh.points, ((0, 0), (0, 2)))
    mesh.points[-1] = np.array([1, 0.5, 0.5])
    mesh.dim = 3
    mesh.ndof = 9

    element = fem.Line()
    quadrature = fem.GaussLegendre(order=0, dim=1)
    region = fem.Region(mesh, element, quadrature, grad=False)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    # constraint

    field[0].values[-1] = [-0.6, 0, -0.6]

    # in x
    mpc = fem.MultiPointConstraint(
        field, points=[0, 1], centerpoint=2, skip=(0, 1, 1), multiplier=1e3
    )
    r = mpc.assemble.vector().toarray()
    K = mpc.assemble.matrix().toarray()

    assert np.allclose(r[[0, 3, 6]].ravel(), [600, 600, -1200])
    assert np.allclose(
        K[[0, 0, 3, 3, 6, 6, 6], [0, 6, 3, 6, 0, 3, 6]].ravel(),
        [1000, -1000, 1000, -1000, -1000, -1000, 2000],
    )

    # contact
    field[0].values[-1] = [-1.1, 0, -0.6]

    # in x
    contact = fem.ContactRigidPlane(
        field, points=[0, 1], centerpoint=2, multiplier=1e3, normal=[-1, 0, 0]
    )
    r = contact.assemble.vector().toarray()
    K = contact.assemble.matrix().toarray()

    assert np.allclose(r[[0, 3, 6]].ravel(), [100, 600, -700])
    assert np.allclose(
        K[[0, 0, 3, 3, 6, 6, 6], [0, 6, 3, 6, 0, 3, 6]].ravel(),
        [1000, -1000, 1000, -1000, -1000, -1000, 2000],
    )

    # contact with partial active points
    field[0].values[-1] = [-0.6, 0, -0.6]

    # in x
    contact = fem.ContactRigidPlane(
        field, points=[0, 1], centerpoint=2, multiplier=1e3, normal=[-1, 0, 0]
    )
    r = contact.assemble.vector().toarray()
    K = contact.assemble.matrix().toarray()

    assert np.allclose(r[[3, 6]].ravel(), [100, -100])
    assert np.allclose(
        K[[3, 3, 6, 6], [3, 6, 3, 6]].ravel(), [1000, -1000, -1000, 1000]
    )

    mesh.points[-2, 2] = -100

    # in z
    contact = fem.ContactRigidPlane(
        field, points=[0, 1], centerpoint=2, multiplier=1e3, normal=[0, 0, -1]
    )
    r = contact.assemble.vector().toarray()
    K = contact.assemble.matrix().toarray()

    assert np.allclose(r[[2, 8]].ravel(), [100, -100])
    assert np.allclose(
        K[[2, 2, 8, 8], [2, 8, 2, 8]].ravel(), [1000, -1000, -1000, 1000]
    )


def test_contact_plot_2d():
    mesh = fem.Rectangle(n=3)
    mesh.add_points([[0.8, 0.8]])

    field = fem.FieldContainer([fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2)])
    solid = fem.SolidBody(fem.LinearElastic(E=2.1e5, nu=0.3), field)
    plane = fem.ContactRigidPlane(field, [0, 1], -1, items=[solid], normal=[0, 1, 0])

    v = plane.assemble.vector()
    m = plane.assemble.matrix()

    assert plane.multipliers is not None

    try:
        plotter = mesh.plot(off_screen=True)
        plane.plot(plotter=plotter, line_width=8)
    except ModuleNotFoundError:
        pass

    contact = fem.ContactRigidPlane(
        field, points=[], centerpoint=-1, multiplier=1e3, normal=[-1, 0]
    )

    try:
        contact.plot(sym=(True, False))

        plane.mesh.x -= 1

        plotter = mesh.plot(off_screen=True)
        plane.plot(plotter=plotter, sym=(True, False))

    except ModuleNotFoundError:
        pass


def test_contact_coulomb_sliding_limit():
    mesh = fem.mesh.Line(n=3)
    mesh.update(cells=mesh.cells[:1])
    mesh.points = np.pad(mesh.points, ((0, 0), (0, 2)))
    mesh.points[-1] = np.array([1, 0.5, 0.5])
    mesh.dim = 3
    mesh.ndof = 9

    element = fem.Line()
    quadrature = fem.GaussLegendre(order=0, dim=1)
    region = fem.Region(mesh, element, quadrature, grad=False)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    contact = fem.ContactRigidPlane(
        field,
        points=[0, 1],
        centerpoint=2,
        multiplier=1e3,
        multiplier_tangential=1e2,
        normal=[-1, 0, 0],
        friction=0.1,
    )

    # initialize contact reference in compression
    field[0].values[-1] = [-1.1, 0.0, 0.0]
    contact.assemble.vector()

    # apply tangential relative motion: both points are in sliding regime
    field[0].values[-1] = [-1.1, 0.5, 0.0]
    r = contact.assemble.vector().toarray()
    K = contact.assemble.matrix().toarray()

    # Coulomb limits
    assert np.allclose(r[[1, 4, 7]].ravel(), [-10, -50, 60])


if __name__ == "__main__":
    test_contact_mixed()
    test_contact_isolated()
    test_contact_plot_2d()
    test_contact_coulomb_sliding_limit()
