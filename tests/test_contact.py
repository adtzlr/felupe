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
@pytest.mark.parametrize("elastic_slip", [None, 1e-3, 2e-2])
def test_contact_consistent_stiffness(dim, friction, elastic_slip):
    "The stiffness matrix is the derivative of the vector (stick and slip)."

    field, solid, contact = contact_model(dim, friction, elastic_slip=elastic_slip)

    K = contact.assemble.matrix(field).toarray()
    r = contact.assemble.vector(field).toarray().ravel()

    active = contact.results.active
    active_old = contact.results.statevars["active"]

    # points in contact at the beginning of the increment and new points
    assert np.any(active & active_old)
    assert np.any(active & ~active_old)

    if 0 < friction < 0.1 and elastic_slip != 2e-2:
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


def test_contact_elastic_slip():
    "Points stick for an elastic slip below the allowable elastic slip."

    for elastic_slip in [1e-3, 2e-2]:
        field, solid, contact = contact_model(
            dim=3, friction=0.3, elastic_slip=elastic_slip
        )
        contact.assemble.vector(field)

        active = contact.results.active
        slip = contact.results.slip
        P = contact.projection_tangential

        # elastic slip of the trial reference gap vectors
        dx = contact.results._statevars["dx"]
        e = np.linalg.norm((dx - contact.results.dx_ref) @ P, axis=1)

        assert np.all(e[active & ~slip] <= elastic_slip * (1 + 1e-8))
        assert np.allclose(e[active & slip], elastic_slip)

    # the stick-slip state does not depend on the friction coefficient
    states = []
    for friction in [0.1, 1.0]:
        field, solid, contact = contact_model(
            dim=3, friction=friction, elastic_slip=0.02
        )
        contact.assemble.vector(field)
        states.append(contact.results.slip.copy())

    assert np.any(states[0])
    assert np.all(states[0] == states[1])

    with pytest.raises(ValueError):
        contact_model(dim=3, friction=0.3, elastic_slip=0.0)


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


def cylinder_model(dim, friction, inside=False, **kwargs):
    "A cylinder contact with a converged state in contact and a trial state."

    rng = np.random.default_rng(1)

    if dim == 3:
        mesh = fem.Cube(n=4)
        region_class = fem.RegionHexahedron
        if inside:
            center, radius, axis = [0.5, 0.5, 0.5], 0.62, [0.1, 0.0, 1.0]
        else:
            center, radius, axis = [0.5, 0.5, 1.75], 0.8, [1.0, 0.3, 0.0]
    else:
        mesh = fem.Rectangle(n=6)
        region_class = fem.RegionQuad
        axis = None
        center, radius = ([0.5, 0.5], 0.62) if inside else ([0.5, 1.75], 0.8)

    mesh.add_points(center)
    displacement = fem.Field(region_class(mesh), dim=dim)
    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)
    npoints = mesh.npoints - 1

    contact = fem.ContactRigidCylinder(
        field,
        points=np.arange(npoints),
        centerpoint=-1,
        radius=radius,
        axis=axis,
        inside=inside,
        friction=friction,
        multiplier=50.0,
        multiplier_tangential=20.0,
        **kwargs,
    )

    # converged state
    if not inside:
        displacement.values[-1, -1] = -0.05
    displacement.values[:-1] += 0.01 * rng.standard_normal((npoints, dim))
    contact.assemble.vector(field)
    contact.results.update_statevars()

    # trial state
    move = 0.02 * rng.standard_normal(dim)
    if inside:
        move[0] = 0.12
    else:
        move[-1] = -0.1 if dim == 2 else -0.04
    displacement.values[-1] += move
    displacement.values[:-1] += 0.01 * rng.standard_normal((npoints, dim))

    return field, solid, contact


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("friction", [0.0, 0.05, np.inf])
@pytest.mark.parametrize("elastic_slip", [None, 2e-2])
def test_contact_cylinder_consistent_stiffness(dim, inside, friction, elastic_slip):
    "The stiffness matrix of the cylinder is consistent with curvature='trial'."

    field, solid, contact = cylinder_model(
        dim, friction, inside, elastic_slip=elastic_slip, curvature="trial"
    )

    K = contact.assemble.matrix(field).toarray()

    active = contact.results.active
    active_old = contact.results.statevars["active"]

    # points in contact at the beginning of the increment and new points
    assert np.any(active & active_old)
    assert np.any(active & ~active_old)

    K_fd = stiffness_fd(contact, field)
    assert np.allclose(K, K_fd, rtol=0, atol=1e-6 * np.abs(K_fd).max())


def test_contact_cylinder_curvature():
    "The curvature term is evaluated with the converged gaps by default."

    field, solid, contact = cylinder_model(dim=3, friction=0.0)
    K = contact.assemble.matrix(field).toarray()

    contact.curvature = "trial"
    K_trial = contact.assemble.matrix(field).toarray()

    contact.curvature = None
    K_none = contact.assemble.matrix(field).toarray()

    # the curvature term of the trial state is negative (outside contact)
    assert not np.allclose(K_trial, K_none)
    assert np.trace(K_trial) < np.trace(K_none)

    # the converged gaps are smaller, the new points have no curvature term
    assert np.trace(K_trial) < np.trace(K) < np.trace(K_none)

    # the vector does not depend on the curvature term
    r = contact.assemble.vector(field).toarray()
    contact.curvature = "converged"
    assert np.allclose(contact.assemble.vector(field).toarray(), r)

    with pytest.raises(ValueError):
        cylinder_model(dim=3, friction=0.0, curvature="consistent")


def test_contact_cylinder_plane():
    "A cylinder with a large radius is equal to a plane."

    def evaluate(kind, radius=1e3):
        mesh = fem.Cube(n=3)
        mesh.add_points([0.5, 0.5, 1.0 + (radius if kind == "cylinder" else 0)])
        displacement = fem.Field(fem.RegionHexahedron(mesh), dim=3)
        field = fem.FieldContainer([displacement])
        points = np.arange(mesh.npoints - 1)[mesh.z[:-1] == 1]
        kwargs = dict(friction=0.3, multiplier=1e2, multiplier_tangential=10.0)

        if kind == "cylinder":
            contact = fem.ContactRigidCylinder(
                field, points, -1, radius=radius, axis=[0, 1, 0], **kwargs
            )
        else:
            contact = fem.ContactRigidPlane(
                field, points, -1, normal=[0, 0, -1], **kwargs
            )

        displacement.values[-1] = [0.0, 0.0, -0.05]
        contact.assemble.vector(field)
        contact.results.update_statevars()
        displacement.values[-1] = [0.02, 0.01, -0.06]

        r = contact.assemble.vector(field).toarray()
        K = contact.assemble.matrix(field).toarray()

        return r, K

    r_plane, K_plane = evaluate("plane")
    r_cylinder, K_cylinder = evaluate("cylinder")

    assert np.allclose(r_cylinder, r_plane, rtol=0, atol=1e-2 * np.abs(r_plane).max())
    assert np.allclose(K_cylinder, K_plane, rtol=0, atol=1e-2 * np.abs(K_plane).max())


def test_contact_cylinder_errors():
    "Invalid parameters of a cylinder raise errors."

    mesh = fem.Rectangle(n=3)
    mesh.add_points([0.5, 1.5])
    field = fem.FieldContainer([fem.Field(fem.RegionQuad(mesh), dim=2)])

    with pytest.raises(ValueError):
        fem.ContactRigidCylinder(field, [0, 1], -1, radius=0.0)

    with pytest.raises(ValueError):
        fem.ContactRigidCylinder(field, [0, 1], -1, radius=0.5, axis=[1, 0, 0])

    with pytest.raises(ValueError):
        fem.ContactRigidCylinder(field, [0, 1], -1, radius=0.5, axis=[0, 0, 0])

    mesh = fem.Cube(n=3)
    mesh.add_points([0.5, 0.5, 1.5])
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])

    with pytest.raises(ValueError):
        fem.ContactRigidCylinder(field, [0, 1], -1, radius=0.5)


def test_contact_cylinder_plot():
    "Plot cylinders in 2d and 3d."

    for dim, inside in [(2, False), (3, False), (3, True)]:
        field, solid, contact = cylinder_model(dim, 0.1, inside=inside)

        try:
            plotter = field.region.mesh.plot(off_screen=True)
            contact.plot(plotter=plotter)
            contact.plot(plotter=plotter, sym=(True, False, False)[:dim])
            contact.plot(plotter=plotter, deformed=False, show_line=False)
        except ModuleNotFoundError:
            pass


def test_contact_cylinder_job():
    "A rubber strip is indented and dragged by a cylinder (with cutbacks)."

    mesh = fem.Rectangle(b=(3, 1), n=(31, 9))
    mesh.add_points([0.8, 1.5])
    displacement = fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2)
    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), field)

    contact = fem.ContactRigidCylinder(
        field,
        points=np.arange(mesh.npoints - 1)[mesh.y[:-1] == 1],
        centerpoint=-1,
        radius=0.5,
        friction=0.3,
        multiplier=10.0,
        elastic_slip=5e-3,
        items=[solid],
    )

    center = np.zeros(mesh.npoints, dtype=bool)
    center[-1] = True

    boundaries = {
        "bottom": fem.Boundary(displacement, fy=0),
        "move-y": fem.Boundary(displacement, mask=center, skip=(1, 0)),
        "move-x": fem.Boundary(displacement, mask=center, skip=(0, 1)),
    }
    ramp = {
        boundaries["move-y"]: fem.math.linsteps([0, -0.15, -0.15], num=[2, 6]),
        boundaries["move-x"]: fem.math.linsteps([0, 0, 0.6], num=[2, 6]),
    }
    step = fem.Step([solid, contact], ramp=ramp, boundaries=boundaries)

    cutback = fem.CutbackPlugin()
    fem.Job([step], plugins=[cutback]).evaluate(verbose=0)

    assert np.any(contact.results.active)

    # the reaction force of the center-point (pressed downwards, dragged forwards)
    r = contact.assemble.vector(field).toarray().reshape(-1, 2)
    assert r[-1, 1] < 0
    assert r[-1, 0] > 0


if __name__ == "__main__":
    test_contact_mixed()
    test_contact_isolated()
    test_contact_plot_2d()
    test_contact_coulomb_sliding_limit()
    for dim in [2, 3]:
        for friction in [0.0, 0.05, 0.3, np.inf]:
            for elastic_slip in [None, 1e-3, 2e-2]:
                test_contact_consistent_stiffness(dim, friction, elastic_slip)
    test_contact_elastic_slip()
    test_contact_statevars()
    test_contact_multipliers()
    test_contact_cylinder_curvature()
    test_contact_cylinder_plane()
    test_contact_cylinder_errors()
    test_contact_cylinder_plot()
    test_contact_cylinder_job()
