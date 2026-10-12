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

    # initialize contact reference in compression (the state variables are only
    # updated after a converged increment)
    field[0].values[-1] = [-1.1, 0.0, 0.0]
    contact.assemble.vector()
    contact.results.update_statevars()

    # apply tangential relative motion: both points are in sliding regime
    field[0].values[-1] = [-1.1, 0.5, 0.0]
    r = contact.assemble.vector().toarray()
    K = contact.assemble.matrix().toarray()

    # Coulomb limits
    assert np.allclose(r[[1, 4, 7]].ravel(), [-10, -50, 60])


def contact_model(dim, friction, items=False, **kwargs):
    "A contact with a converged state in contact and a trial state of the center."

    rng = np.random.default_rng(0)

    if dim == 3:
        mesh = fem.Cube(n=4)
        mesh.add_points([0.5, 0.5, 1.0])
        region = fem.RegionHexahedron(mesh)
        normal = [0.1, -0.2, -1.0]
    else:
        mesh = fem.Rectangle(n=6)
        mesh.add_points([0.5, 1.0])
        region = fem.RegionQuad(mesh)
        normal = [0.2, -1.0]

    displacement = fem.Field(region, dim=dim)
    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)
    points = np.arange(mesh.npoints - 1)[np.isclose(mesh.points[:-1, -1], 1.0)]

    contact = fem.ContactRigidPlane(
        field,
        points,
        centerpoint=-1,
        normal=normal,
        friction=friction,
        multiplier=50.0,
        multiplier_tangential=20.0,
        items=[solid] if items else None,
        **kwargs,
    )

    # converged state with a pressed center-point
    displacement.values[-1, -1] = -0.05
    displacement.values[:-1] += 0.01 * rng.standard_normal((mesh.npoints - 1, dim))
    contact.assemble.vector(field)
    contact.results.update_statevars()

    # trial state with an additional normal and tangential movement
    displacement.values[-1] += np.append(0.03 * rng.standard_normal(dim - 1), -0.04)
    displacement.values[:-1] += 0.01 * rng.standard_normal((mesh.npoints - 1, dim))

    return field, solid, contact


def stiffness_fd(contact, field, h=1e-7):
    "Return the stiffness matrix by central finite differences of the vector."

    values = field[0].values
    values0 = values.copy()
    K = np.zeros((values.size, values.size))

    for i in range(values.size):
        values[:] = values0
        values.ravel()[i] += h
        rp = contact.assemble.vector(field).toarray().ravel()
        values[:] = values0
        values.ravel()[i] -= h
        rm = contact.assemble.vector(field).toarray().ravel()
        K[:, i] = (rp - rm) / (2 * h)

    values[:] = values0

    return K


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("friction", [0.0, 0.05, 0.3, np.inf])
def test_contact_consistent_stiffness(dim, friction):
    "The stiffness matrix is the derivative of the vector (stick and slip)."

    field, solid, contact = contact_model(dim, friction)

    K = contact.assemble.matrix(field).toarray()
    r = contact.assemble.vector(field).toarray().ravel()

    active = contact.results.active
    active_old = contact.results.statevars["active"]

    # points in contact at the beginning of the increment and new points
    assert np.any(active & active_old)
    assert np.any(active & ~active_old)

    if 0 < friction < 0.1:
        assert np.any(contact.results.slip)

    # the vector does not change the state variables
    assert np.allclose(contact.assemble.vector(field).toarray().ravel(), r)

    K_fd = stiffness_fd(contact, field)
    assert np.allclose(K, K_fd, rtol=0, atol=1e-6 * np.abs(K_fd).max())

    # the stiffness matrix of frictional contact is not symmetric
    if friction > 0:
        assert not np.allclose(K, K.T)

    contact.symmetric = True
    K_sym = contact.assemble.matrix(field).toarray()
    assert np.allclose(K_sym, K_sym.T)
    assert np.allclose(K_sym, (K + K.T) / 2)


def test_contact_statevars():
    "The state variables are only updated after a converged increment."

    field, solid, contact = contact_model(dim=3, friction=0.1)

    statevars = {k: v.copy() for k, v in contact.results.statevars.items()}
    r = contact.assemble.vector(field).toarray()

    # a trial evaluation does not change the state variables
    field[0].values[-1] += [0.1, 0.0, -0.01]
    contact.assemble.vector(field)
    field[0].values[-1] -= [0.1, 0.0, -0.01]

    for key, value in statevars.items():
        assert np.allclose(contact.results.statevars[key], value)

    assert np.allclose(contact.assemble.vector(field).toarray(), r)

    # checkpoint, update and restore
    checkpoint = contact.checkpoint()
    contact.results.update_statevars()
    assert not np.allclose(contact.results.statevars["dx"], statevars["dx"])

    contact.restore(checkpoint)
    for key, value in statevars.items():
        assert np.allclose(contact.results.statevars[key], value)

    assert np.allclose(contact.assemble.vector(field).toarray(), r)


def test_contact_multipliers():
    "The stiffness-based multipliers are evaluated at the points in contact."

    field, solid, contact = contact_model(dim=3, friction=0.1, items=True)
    dim = 3

    K = solid.assemble.matrix(field)
    multipliers = contact.contact_multipliers()

    dof = field[0].indices.dof[contact.points]
    blocks = np.array([K[d.reshape(-1, 1), d].toarray() for d in dof])
    stiffness = blocks.sum(axis=0)

    normal = np.sum(stiffness * contact.projection_normal) / len(contact.points)
    tangential = np.sum(stiffness * contact.projection_tangential) / (
        len(contact.points) * (dim - 1)
    )

    assert np.isclose(multipliers.normal, 50.0 * normal)
    assert np.isclose(multipliers.tangential, 20.0 * tangential)

    # the multipliers are constant within an increment
    contact.assemble.vector(field)
    multipliers = contact.multipliers

    field[0].values[:-1] *= 1.5
    solid.assemble.matrix(field)
    contact.assemble.vector(field)
    assert contact.multipliers is multipliers

    # and they are updated after a converged increment
    contact.results.update_statevars()
    contact.assemble.vector(field)
    assert contact.multipliers is not multipliers
    assert not np.isclose(contact.multipliers.normal, multipliers.normal)

    # multipliers which are updated in each evaluation
    contact.multiplier_update = "iteration"
    multipliers = contact.multipliers
    contact.assemble.vector(field)
    assert contact.multipliers is not multipliers

    # multipliers which are evaluated only once
    contact.multiplier_update = "once"
    multipliers = contact.multipliers
    contact.results.update_statevars()
    contact.assemble.vector(field)
    assert contact.multipliers is multipliers

    with pytest.raises(ValueError):
        contact_model(dim=3, friction=0.1, multiplier_update="never")


if __name__ == "__main__":
    test_contact_mixed()
    test_contact_isolated()
    test_contact_plot_2d()
    test_contact_coulomb_sliding_limit()
    for dim in [2, 3]:
        for friction in [0.0, 0.05, 0.3, np.inf]:
            test_contact_consistent_stiffness(dim, friction)
    test_contact_statevars()
    test_contact_multipliers()
