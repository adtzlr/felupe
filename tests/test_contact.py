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


def pre_solidbody_contact(
    n_bottom=(4, 4, 3),
    n_top=(3, 3, 3),
    offset=0.02,
    mu=(1.0, 1.0),
    bulk=(50.0, 50.0),
    distort=0.0,
    seed=4,
    **kwargs,
):
    "Return a two-body contact model with a block on top of another block."

    bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=n_bottom)
    top = fem.Cube(a=(0.15, 0.15, 1 + offset), b=(0.85, 0.85, 1.6 + offset), n=n_top)

    mesh = fem.MeshContainer([bottom, top], merge=True).stack()

    # the masks of the contact surfaces are evaluated on the undeformed mesh
    mask_primary = np.isclose(mesh.z, 1.0)
    mask_secondary = np.isclose(mesh.z, 1 + offset)

    if distort > 0:
        rng = np.random.default_rng(seed)
        mesh.points = mesh.points + distort * rng.normal(size=mesh.points.shape)

    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    lower = mesh.points[mesh.cells].mean(axis=1)[:, 2] < 1.0
    umat = fem.NeoHooke(
        mu=np.where(lower, mu[0], mu[1]), bulk=np.where(lower, bulk[0], bulk[1])
    )
    solid = fem.SolidBody(umat=umat, field=field)

    secondary = fem.FieldContainer(
        [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask_secondary), dim=3)]
    )
    primary = fem.FieldContainer(
        [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask_primary), dim=3)]
    )

    contact = fem.SolidBodyContact(secondary, primary, items=[solid], **kwargs)

    boundaries = {
        "fixed": fem.Boundary(field[0], fz=0.0),
        "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
        "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    }

    return mesh, field, solid, contact, boundaries


def test_solidbody_contact_kinematics():
    "Check the gap and the normal vectors of two parallel flat surfaces."

    offset = 0.02
    penetration = 0.05

    mesh, field, solid, contact, boundaries = pre_solidbody_contact(
        offset=offset, penalty=100.0, smoothing=0.0
    )

    # no contact for the undeformed configuration
    contact.assemble.vector(field=field)
    assert contact.results.npoints_in_contact == 0
    assert contact.results.force.nnz == 0

    # move the top block downwards
    values = field[0].values.copy()
    values[mesh.z > 1.0, 2] = -(offset + penetration)
    field[0].values = values

    contact.assemble.vector(field=field)
    kinematics = contact._kinematics[0]

    # all integration points of the secondary surface are in contact
    assert contact.results.npoints_in_contact == contact.pairs[0].dV.size
    assert np.allclose(kinematics["gap"], -penetration)
    assert np.allclose(kinematics["normal"], [0.0, 0.0, 1.0])

    # the total contact force is equal to the (constant) pressure times the area
    force = contact.results.force.toarray().reshape(-1, 3)
    area = contact.pairs[0].dV.sum()

    assert np.isclose(force[:, 2].sum(), 0.0)
    assert np.isclose(force[force[:, 2] < 0, 2].sum(), -100.0 * penetration * area)


def test_solidbody_contact_tangent():
    "Compare the contact stiffness matrix with a numerical tangent."

    mesh, field, solid, contact, boundaries = pre_solidbody_contact(
        distort=0.04, penalty=100.0
    )

    rng = np.random.default_rng(7)
    values = 0.03 * rng.normal(size=field[0].values.shape)
    values[:, 2] -= 0.05 * mesh.z
    values[mesh.z > 1.0, 2] -= 0.12
    field[0].values = values

    contact.assemble.vector(field=field)
    K = contact.assemble.matrix().toarray()

    assert contact.results.npoints_in_contact > 0

    # the tangent stiffness matrix of a frictionless contact is symmetric
    assert np.allclose(K, K.T)

    # sum of all contact forces (equilibrium of the contact tractions)
    force = contact.results.force.toarray().reshape(-1, 3)
    assert np.allclose(force.sum(axis=0), 0.0, atol=1e-12)

    # numerical tangent for a subset of the degrees of freedom
    dofs = rng.choice(K.shape[0], size=30, replace=False)
    eps = 1e-6
    K_num = np.zeros((K.shape[0], len(dofs)))

    for j, dof in enumerate(dofs):
        for sign in [1, -1]:
            u = values.copy()
            u.ravel()[dof] += sign * eps
            field[0].values = u
            contact.assemble.vector(field=field)
            K_num[:, j] += sign * contact.results.force.toarray().ravel() / (2 * eps)

    assert np.linalg.norm(K[:, dofs] - K_num) / np.linalg.norm(K_num) < 1e-6


def test_solidbody_contact_rigid_body_motion():
    "A rigid body translation must not change the contact forces."

    mesh, field, solid, contact, boundaries = pre_solidbody_contact(distort=0.04)
    contact.penalty = 100.0

    values = np.zeros_like(field[0].values)
    values[mesh.z > 1.0, 2] = -0.06
    field[0].values = values

    contact.assemble.vector(field=field)
    force = contact.results.force.toarray()

    field[0].values = values + np.array([0.3, -0.2, 0.1])
    contact.assemble.vector(field=field)

    assert np.allclose(force, contact.results.force.toarray())


def test_solidbody_contact_patch():
    "Two blocks with non-matching meshes must transmit a nearly uniform stress."

    lower = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
    upper = fem.Cube(a=(0, 0, 1), b=(1, 1, 2), n=(6, 6, 3))

    # the points of both blocks are not merged: the interface points are duplicated
    mesh = fem.MeshContainer([lower, upper], merge=False).stack()

    interface = np.isclose(mesh.z, 1.0)
    is_lower = np.arange(mesh.npoints) < lower.npoints

    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

    secondary = fem.FieldContainer(
        [
            fem.Field(
                fem.RegionHexahedronBoundary(mesh, mask=interface & ~is_lower), dim=3
            )
        ]
    )
    primary = fem.FieldContainer(
        [
            fem.Field(
                fem.RegionHexahedronBoundary(mesh, mask=interface & is_lower), dim=3
            )
        ]
    )
    contact = fem.SolidBodyContact(secondary, primary, items=[solid])

    boundaries = {
        "bottom": fem.Boundary(field[0], fz=0.0, skip=(1, 1, 0)),
        "sym-x": fem.Boundary(field[0], fx=0.0, skip=(0, 1, 1)),
        "sym-y": fem.Boundary(field[0], fy=0.0, skip=(1, 0, 1)),
        "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0), value=-0.1),
    }

    step = fem.Step(items=[solid, contact], boundaries=boundaries)
    fem.Job(steps=[step]).evaluate(verbose=0)

    stress = solid.evaluate.cauchy_stress()[2, 2]

    assert stress.mean() < 0
    assert np.abs(stress - stress.mean()).max() / np.abs(stress.mean()) < 0.02

    # the contact is closed, i.e. the gap is negative and small
    gap = contact.results.gap[0]
    assert np.all(gap < 0)
    assert -gap.max() < 0.01 * contact.size


@pytest.mark.parametrize(
    "mu, bulk",
    [((1.0, 1.0), (50.0, 50.0)), ((1e4, 1.0), (2e4, 50.0)), ((1.0, 1e4), (50.0, 2e4))],
)
def test_solidbody_contact_job(mu, bulk):
    "Rubber-to-rubber and rubber-to-metal contacts must converge."

    mesh, field, solid, contact, boundaries = pre_solidbody_contact(mu=mu, bulk=bulk)

    step = fem.Step(
        items=[solid, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.2], num=4)},
        boundaries=boundaries,
    )
    job = fem.Job(steps=[step])
    job.evaluate(verbose=0, tol=1e-8)

    assert contact.results.npoints_in_contact > 0
    assert contact.penalty > 0

    # the penetration is small compared to the size of the faces
    gap = contact.results.gap[0]
    assert -gap.min() < 0.1 * contact.size


def test_solidbody_contact_two_pass():
    "The two-pass contact must converge and must be symmetric."

    mesh, field, solid, contact, boundaries = pre_solidbody_contact(two_pass=True)

    assert len(contact.pairs) == 2

    step = fem.Step(
        items=[solid, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.2], num=4)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(verbose=0, tol=1e-8)

    K = contact.results.stiffness.toarray()
    assert np.allclose(K, K.T)
    assert contact.results.npoints_in_contact > 0


def test_solidbody_contact_penalty():
    "Check the automatic estimation of the penalty stiffness."

    # the penalty stiffness scales with the stiffness of the softer surface
    _, _, _, soft, _ = pre_solidbody_contact(mu=(1.0, 1.0), bulk=(50.0, 50.0))
    _, _, _, mixed, _ = pre_solidbody_contact(mu=(1e3, 1.0), bulk=(2e3, 50.0))

    assert np.isclose(soft.update_penalty(), mixed.update_penalty(), rtol=0.5)

    # the penalty stiffness scales with the scale factor
    _, _, _, scaled, _ = pre_solidbody_contact(penalty_scale=20.0)
    assert np.isclose(scaled.update_penalty(), 2 * soft.update_penalty())

    # a given penalty stiffness is not modified
    _, _, _, given, _ = pre_solidbody_contact(penalty=1234.0, penalty_scale=7.0)
    given.assemble.vector()
    assert given.penalty == 1234.0

    # the representation of the contact object
    assert "Penalty stiffness" in str(given)


def test_solidbody_contact_geometric_stiffness():
    "The contact must converge without the geometric part of the stiffness matrix."

    mesh, field, solid, contact, boundaries = pre_solidbody_contact(
        geometric_stiffness=False
    )

    step = fem.Step(
        items=[solid, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.2], num=4)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(verbose=0, tol=1e-8)

    assert contact.results.npoints_in_contact > 0


def test_solidbody_contact_error():
    "Check the errors of an invalid setup."

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    boundary = fem.FieldContainer(
        [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mesh.z == 1), dim=3)]
    )

    # neither items nor penalty are given
    with pytest.raises(ValueError):
        fem.SolidBodyContact(boundary, boundary)

    # a field on a region without normal vectors is given
    with pytest.raises(TypeError):
        fem.SolidBodyContact(field, boundary, penalty=1.0)

    # a boundary region of a non-hexahedron mesh is given
    rectangle = fem.Rectangle(n=3)
    edges = fem.FieldContainer([fem.Field(fem.RegionQuadBoundary(rectangle), dim=2)])

    with pytest.raises(NotImplementedError):
        fem.SolidBodyContact(edges, edges, penalty=1.0)


if __name__ == "__main__":
    test_contact_mixed()
    test_contact_isolated()
    test_contact_plot_2d()
    test_contact_coulomb_sliding_limit()
    test_solidbody_contact_kinematics()
    test_solidbody_contact_tangent()
    test_solidbody_contact_rigid_body_motion()
    test_solidbody_contact_patch()
    test_solidbody_contact_job((1.0, 1.0), (50.0, 50.0))
    test_solidbody_contact_job((1e4, 1.0), (2e4, 50.0))
    test_solidbody_contact_job((1.0, 1e4), (50.0, 2e4))
    test_solidbody_contact_two_pass()
    test_solidbody_contact_penalty()
    test_solidbody_contact_geometric_stiffness()
    test_solidbody_contact_error()
