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


def test_mpc():
    mesh = fem.Cube(n=3)
    mesh.points = np.vstack((mesh.points, [2, 0, 0]))
    mesh.update(cells=mesh.cells)

    region = fem.RegionHexahedron(mesh)

    u = fem.FieldContainer([fem.Field(region, dim=3)])
    F = u.extract()

    umat = fem.constitution.NeoHooke(mu=1.0, bulk=2.0)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fem.Boundary(u[0], fx=f0)
    boundaries["right"] = fem.Boundary(u[0], fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fem.Boundary(u[0], fx=f2, skip=(0, 1, 1), value=0.5)

    mpc = fem.Boundary(u[0], fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fem.MultiPointConstraint(field=u, points=mpc, centerpoint=cpoint)

    for f in [None, u]:
        K_RBE2 = RBE2.assemble.matrix(u)
        r_RBE2 = RBE2.assemble.vector(u)

        linearform = fem.IntegralForm(umat.gradient(F)[:-1], u, region.dV)
        r = linearform.assemble() + r_RBE2

        bilinearform = fem.IntegralForm(umat.hessian(F), u, region.dV, u)
        K = bilinearform.assemble() + K_RBE2

        assert r.shape == (84, 1)
        assert K.shape == (84, 84)


def pre_mpc_mixed(point, values):
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

    mpc = fem.Boundary(fields[0], fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fem.MultiPointConstraint(fields, points=mpc, centerpoint=cpoint)
    CONT = fem.MultiPointContact(fields, points=mpc, centerpoint=cpoint)

    try:
        CONT.plot()
    except ModuleNotFoundError:
        pass

    for f in [None, fields]:
        K_RBE2 = RBE2.assemble.matrix(f)
        r_RBE2 = RBE2.assemble.vector(f)

        K_CONT = CONT.assemble.matrix(f)
        r_CONT = CONT.assemble.vector(f)

        assert K_RBE2.shape == K_CONT.shape
        assert r_RBE2.shape == r_CONT.shape

    linearform = fem.IntegralForm(umat.gradient([F, p, J])[:-1], fields, dV)
    r = linearform.assemble()

    r_RBE2.resize(*r.shape)
    r = r + r_RBE2

    bilinearform = fem.IntegralForm(umat.hessian([F, p, J]), fields, dV, fields)
    K = bilinearform.assemble()

    K_RBE2.resize(*K.shape)
    K = K + K_RBE2


def test_mpc_mixed():
    pre_mpc_mixed(point=[2, 0, 0], values=[0, 0, 0])
    pre_mpc_mixed(point=[2, 0, 0], values=[-5, 0, 0])


def test_mpc_isolated():
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

    # in z
    mpc = fem.MultiPointConstraint(
        field, points=[0, 1], centerpoint=2, skip=(1, 1, 0), multiplier=1e3
    )
    r = mpc.assemble.vector().toarray()
    K = mpc.assemble.matrix().toarray()

    assert np.allclose(r[[2, 5, 8]].ravel(), [600, 600, -1200])
    assert np.allclose(
        K[[2, 2, 5, 5, 8, 8, 8], [2, 8, 5, 8, 2, 5, 8]].ravel(),
        [1000, -1000, 1000, -1000, -1000, -1000, 2000],
    )

    # contact

    field[0].values[-1] = [-1.1, 0, -0.6]

    # in x
    mpc = fem.MultiPointContact(
        field, points=[0, 1], centerpoint=2, skip=(0, 1, 1), multiplier=1e3
    )
    r = mpc.assemble.vector().toarray()
    K = mpc.assemble.matrix().toarray()

    assert np.allclose(r[[0, 3, 6]].ravel(), [100, 600, -700])
    assert np.allclose(
        K[[0, 0, 3, 3, 6, 6, 6], [0, 6, 3, 6, 0, 3, 6]].ravel(),
        [1000, -1000, 1000, -1000, -1000, -1000, 2000],
    )

    # in z
    mpc = fem.MultiPointConstraint(
        field, points=[0, 1], centerpoint=2, skip=(1, 1, 0), multiplier=1e3
    )
    r = mpc.assemble.vector().toarray()
    K = mpc.assemble.matrix().toarray()

    assert np.allclose(r[[2, 5, 8]].ravel(), [600, 600, -1200])
    assert np.allclose(
        K[[2, 2, 5, 5, 8, 8, 8], [2, 8, 5, 8, 2, 5, 8]].ravel(),
        [1000, -1000, 1000, -1000, -1000, -1000, 2000],
    )

    # contact with partial active points

    field[0].values[-1] = [-0.6, 0, -0.6]

    # in x
    mpc = fem.MultiPointContact(
        field, points=[0, 1], centerpoint=2, skip=(0, 1, 1), multiplier=1e3
    )
    r = mpc.assemble.vector().toarray()
    K = mpc.assemble.matrix().toarray()

    assert np.allclose(r[[3, 6]].ravel(), [100, -100])
    assert np.allclose(
        K[[3, 3, 6, 6], [3, 6, 3, 6]].ravel(), [1000, -1000, -1000, 1000]
    )

    mesh.points[-2, 2] = -100

    # in z
    mpc = fem.MultiPointContact(
        field, points=[0, 1], centerpoint=2, skip=(1, 1, 0), multiplier=1e3
    )
    r = mpc.assemble.vector().toarray()
    K = mpc.assemble.matrix().toarray()

    assert np.allclose(r[[2, 8]].ravel(), [100, -100])
    assert np.allclose(
        K[[2, 2, 8, 8], [2, 8, 2, 8]].ravel(), [1000, -1000, -1000, 1000]
    )


def test_mpc_plot_2d():
    mesh = fem.Rectangle(n=3)
    field = fem.FieldContainer([fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2)])
    plane = fem.MultiPointContact(field, [0, 1], -1, skip=(0, 1))

    try:
        plotter = mesh.plot(off_screen=True)
        plane.plot(plotter=plotter, line_width=8)
    except ModuleNotFoundError:
        pass

    mpc = fem.MultiPointConstraint(field, [0, 1], -1)

    try:
        mpc.plot()
    except ModuleNotFoundError:
        pass


if __name__ == "__main__":
    test_mpc()
    test_mpc_mixed()
    test_mpc_isolated()
    test_mpc_plot_2d()
