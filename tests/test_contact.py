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


def _two_blocks_2d(friction=0.0, symmetric=False, multiplier=5.0, n=5):
    "Two stacked plane-strain blocks with a small initial gap and a contact item."

    bottom = fem.Rectangle(a=(0, 0), b=(1, 1), n=(n, n))
    top = fem.Rectangle(a=(0, 1.1), b=(1, 2.1), n=(n, n))
    container = fem.MeshContainer([bottom, top], merge=True)

    regions = [fem.RegionQuad(m) for m in container.meshes]
    fields = [fem.FieldContainer([fem.FieldPlaneStrain(r, dim=2)]) for r in regions]

    umat = fem.NeoHooke(mu=1.0, bulk=2.0)
    solids = [fem.SolidBody(umat, f) for f in fields]

    mask = container.meshes[0].points[:, 1] == 1
    boundary_bottom = fem.RegionQuadBoundary(
        container.meshes[0], mask=mask, ensure_3d=False
    )
    field_bottom = fem.FieldContainer([fem.FieldPlaneStrain(boundary_bottom, dim=2)])

    mask = container.meshes[1].points[:, 1] == 1.1
    boundary_top = fem.RegionQuadBoundary(
        container.meshes[1], mask=mask, ensure_3d=False
    )
    field_top = fem.FieldContainer([fem.FieldPlaneStrain(boundary_top, dim=2)])

    contact = fem.SolidBodyContact(
        field_bottom,
        field_top,
        items=solids,
        friction=friction,
        symmetric=symmetric,
        multiplier=multiplier,
    )

    region = fem.RegionQuad(container.stack())
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

    return container, solids, contact, field


def test_solidbody_contact_2d():
    "Frictionless 2D contact between two blocks - the upper block is pressed down."

    container, solids, contact, field = _two_blocks_2d()

    boundaries = {
        "fixed": fem.Boundary(field[0], fy=0),
        "move": fem.Boundary(field[0], fy=2.1),
    }
    move = fem.math.linsteps([0, -0.2], num=10)
    step = fem.Step(
        items=[*solids, contact],
        ramp={boundaries["move"]: move},
        boundaries=boundaries,
    )
    job = fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    # contact is active and the deformation is as prescribed
    assert contact.results.active.sum() > 0
    assert np.isclose(field[0].values[:, 1].min(), -0.2)

    # no significant penetration (deformed surfaces do not cross too much)
    points_bottom = container.meshes[0].points
    points_top = container.meshes[1].points
    y_bottom = (points_bottom + field[0].values)[points_bottom[:, 1] == 1][:, 1].max()
    y_top = (points_top + field[0].values)[points_top[:, 1] == 1.1][:, 1].min()
    assert (y_bottom - y_top) < 0.1


def test_solidbody_contact_assemble():
    "Assembly of the residual vector and the (symmetric) tangent stiffness matrix."

    container, solids, contact, field = _two_blocks_2d()

    # impose a penetration by moving the upper block down
    x = field.region.mesh.points
    values = field[0].values
    values[x[:, 1] >= 1.1, 1] = -0.2

    r = contact.assemble.vector(field=field)
    K = contact.assemble.matrix(field=field)

    ndof = np.sum(field.fieldsizes)
    assert r.shape == (ndof, 1)
    assert K.shape == (ndof, ndof)

    # the contact contributes non-zero residual forces and tangent stiffness
    assert contact.results.active.sum() > 0
    assert np.abs(r.toarray()).max() > 0
    assert K.nnz > 0

    # a larger penalty-multiplier reduces the penetration
    _, solids_a, contact_a, field_a = _two_blocks_2d(multiplier=1.0)
    _, solids_b, contact_b, field_b = _two_blocks_2d(multiplier=20.0)
    for con, sol, fld in [(contact_a, solids_a, field_a), (contact_b, solids_b, field_b)]:
        boundaries = {
            "fixed": fem.Boundary(fld[0], fy=0),
            "move": fem.Boundary(fld[0], fy=2.1),
        }
        step = fem.Step(
            items=[*sol, con],
            ramp={boundaries["move"]: fem.math.linsteps([0, -0.15], num=8)},
            boundaries=boundaries,
        )
        fem.Job(steps=[step]).evaluate(x0=fld, verbose=0)

    pb = container.meshes[0].points
    pt = container.meshes[1].points

    def penetration(fld):
        yb = (pb + fld[0].values)[pb[:, 1] == 1][:, 1].max()
        yt = (pt + fld[0].values)[pt[:, 1] == 1.1][:, 1].min()
        return max(0.0, yb - yt)

    assert penetration(field_b) <= penetration(field_a) + 1e-8


def test_solidbody_contact_friction():
    "2D contact with Coulomb friction - the upper block is pressed down."

    container, solids, contact, field = _two_blocks_2d(friction=0.4)

    boundaries = {
        "fixed": fem.Boundary(field[0], fy=0),
        "move": fem.Boundary(field[0], fy=2.1, skip=(1, 0)),
    }
    move = fem.math.linsteps([0, -0.15], num=8)
    step = fem.Step(
        items=[*solids, contact],
        ramp={boundaries["move"]: move},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    assert contact.results.active.sum() > 0
    assert contact.friction == 0.4
    assert contact.results.slip.shape == contact.results.active.shape


def test_solidbody_contact_symmetric():
    "Symmetric two-pass contact search (slave and master roles are swapped)."

    container, solids, contact, field = _two_blocks_2d(symmetric=True)

    assert len(contact._passes) == 2
    assert len(contact._states) == 2

    boundaries = {
        "fixed": fem.Boundary(field[0], fy=0),
        "move": fem.Boundary(field[0], fy=2.1),
    }
    move = fem.math.linsteps([0, -0.15], num=8)
    step = fem.Step(
        items=[*solids, contact],
        ramp={boundaries["move"]: move},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    assert contact._states[0]["active"].sum() > 0
    assert contact._states[1]["active"].sum() > 0


def test_solidbody_contact_3d():
    "Frictionless 3D contact between two cubes - the upper cube is pressed down."

    bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 4))
    top = fem.Cube(a=(0, 0, 1.1), b=(1, 1, 2.1), n=(4, 4, 4))
    container = fem.MeshContainer([bottom, top], merge=True)

    regions = [fem.RegionHexahedron(m) for m in container.meshes]
    fields = [fem.FieldContainer([fem.Field(r, dim=3)]) for r in regions]

    umat = fem.NeoHooke(mu=1.0, bulk=2.0)
    solids = [fem.SolidBody(umat, f) for f in fields]

    mask = container.meshes[0].points[:, 2] == 1
    boundary_bottom = fem.RegionHexahedronBoundary(container.meshes[0], mask=mask)
    field_bottom = fem.FieldContainer([fem.Field(boundary_bottom, dim=3)])

    mask = container.meshes[1].points[:, 2] == 1.1
    boundary_top = fem.RegionHexahedronBoundary(container.meshes[1], mask=mask)
    field_top = fem.FieldContainer([fem.Field(boundary_top, dim=3)])

    contact = fem.SolidBodyContact(field_bottom, field_top, items=solids, multiplier=5.0)

    region = fem.RegionHexahedron(container.stack())
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    boundaries = {
        "fixed": fem.Boundary(field[0], fz=0),
        "move": fem.Boundary(field[0], fz=2.1),
    }
    move = fem.math.linsteps([0, -0.2], num=10)
    step = fem.Step(
        items=[*solids, contact],
        ramp={boundaries["move"]: move},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    assert contact.results.active.sum() > 0
    assert np.isclose(field[0].values[:, 2].min(), -0.2)


if __name__ == "__main__":
    test_contact_mixed()
    test_contact_isolated()
    test_contact_plot_2d()
    test_contact_coulomb_sliding_limit()
    test_solidbody_contact_2d()
    test_solidbody_contact_assemble()
    test_solidbody_contact_friction()
    test_solidbody_contact_symmetric()
    test_solidbody_contact_3d()
