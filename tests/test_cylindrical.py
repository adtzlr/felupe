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


def annulus(a=1.0, b=2.0, phi=np.pi / 2, n=(5, 9)):
    "A 2d-mesh of a (sector of an) annulus with radii a and b."

    mesh = fem.Rectangle(a=(a, 0), b=(b, phi), n=n)
    r, t = mesh.points.T
    mesh.update(points=np.column_stack([r * np.cos(t), r * np.sin(t)]))

    return mesh


def tube(a=1.0, b=2.0, length=1.0, n=(4, 3), nphi=17):
    "A 3d-mesh of a tube with radii a and b and the z-axis as axis."

    mesh = fem.Rectangle(a=(0, a), b=(length, b), n=n).revolve(nphi, phi=360, axis=0)
    mesh.update(points=mesh.points[:, [1, 2, 0]])

    return mesh


def test_cylindrical_constraint_derivatives():
    "Check the vector and the matrix by finite differences of the potential."

    rng = np.random.default_rng(seed=56)

    for dim in [2, 3]:
        if dim == 2:
            mesh = annulus(n=(3, 4))
            region = fem.RegionQuad(mesh)
            center = (0.3, -0.2)
            axis = (0, 0, 1)
            displacement = (0.05, 0.3)
        else:
            mesh = fem.Cube(a=(1, 0.5, -1), b=(2, 1, 0), n=3)
            region = fem.RegionHexahedron(mesh)
            center = (0.1, -0.2, 0.3)
            axis = (0.2, -0.3, 1.0)
            displacement = (0.05, 0.3, -0.1)

        field = fem.FieldContainer([fem.Field(region, dim=dim)])
        field[0].values[:] = 0.2 * rng.uniform(-1, 1, size=field[0].values.shape)

        for skip in [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 1),
            (0, 1, 0),
        ]:
            constraint = fem.CylindricalConstraint(
                field,
                points=np.arange(mesh.npoints),
                center=center,
                axis=axis,
                displacement=displacement,
                skip=skip,
                multiplier=3.0,
            )

            def potential(values):
                field[0].values[:] = values
                constraint.assemble.vector(field)
                gap = constraint.results.gap
                return constraint.multiplier / 2 * np.sum(gap**2)

            u = field[0].values.copy()
            r = constraint.assemble.vector(field).toarray().ravel()
            K = constraint.assemble.matrix(field).toarray()

            h = 1e-6
            r_fd = np.zeros_like(r)
            K_fd = np.zeros_like(K)

            for i in range(u.size):
                du = np.zeros(u.size)
                du[i] = h
                up = (u.ravel() + du).reshape(u.shape)
                um = (u.ravel() - du).reshape(u.shape)

                r_fd[i] = (potential(up) - potential(um)) / (2 * h)

                field[0].values[:] = up
                rp = constraint.assemble.vector(field).toarray().ravel()
                field[0].values[:] = um
                rm = constraint.assemble.vector(field).toarray().ravel()
                K_fd[:, i] = (rp - rm) / (2 * h)

            field[0].values[:] = u

            assert np.allclose(r, r_fd, atol=1e-6)
            assert np.allclose(K, K_fd, atol=1e-6)
            assert np.allclose(K, K.T)

            # the skipped directions do not contribute
            assert np.allclose(constraint.results.gap[:, np.array(skip, bool)], 0)


def test_cylindrical_constraint_target():
    "A fully constrained mesh is moved to the prescribed cylindrical positions."

    mesh = fem.Cube(a=(1, 0.5, -1), b=(2, 1, 0), n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    center = np.array([0.1, -0.2, 0.3])
    axis = np.array([0.2, -0.3, 1.0])
    displacement = np.array([0.4, 0.9, -0.3])

    constraint = fem.CylindricalConstraint(
        field,
        points=np.ones(mesh.npoints, dtype=bool),
        center=center,
        axis=axis,
        displacement=displacement,
    )
    step = fem.Step([constraint])
    fem.Job([step]).evaluate()

    # exact cylindrical coordinates of the rotated points
    u_r, u_t, u_z = displacement
    R = constraint.radius.reshape(-1, 1)
    Z = constraint.height.reshape(-1, 1)
    e_r, e_t, e_z = constraint.basis.transpose([1, 0, 2])

    phi = u_t / R
    e_r_rotated = np.cos(phi) * e_r + np.sin(phi) * e_t
    x = center + (R + u_r) * e_r_rotated + (Z + u_z) * e_z

    assert np.allclose(mesh.points + field[0].values, x)
    assert np.allclose(constraint.results.gap, 0)

    # the radius of the points is not changed by the rotation
    dx = mesh.points + field[0].values - center
    radius = np.linalg.norm(dx @ constraint.projection, axis=1)
    assert np.allclose(radius, R.ravel() + u_r)


def test_cylindrical_constraint_lame():
    "A ring with prescribed radial displacements on the outer radius (Lamé)."

    a, b = 1.0, 2.0
    mesh = annulus(a, b, n=(13, 25))
    region = fem.RegionQuad(mesh)

    E, nu = 210.0, 0.3
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # analytic solution for plane strain u_r = A r + B / r, with S_rr(a) = 0
    # and u_r(b) = delta
    delta = -1e-4
    c = (lmbda + mu) / mu * a**2
    A = delta / (b + c / b)
    u_r_inner = A * a + A * c / a

    displacement = fem.FieldPlaneStrain(region, dim=2)
    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(fem.LinearElastic(E=E, nu=nu), field)

    radius = np.linalg.norm(mesh.points, axis=1)
    outer = fem.CylindricalConstraint(
        field,
        points=np.isclose(radius, b),
        displacement=(delta, 0),
        skip=(0, 1),
        multiplier=1e6,
    )
    boundaries = {
        "x": fem.Boundary(displacement, fx=0, skip=(0, 1)),
        "y": fem.Boundary(displacement, fy=0, skip=(1, 0)),
    }
    step = fem.Step([solid, outer], boundaries=boundaries)
    fem.Job([step]).evaluate()

    inner = np.isclose(radius, a)
    u_r = np.einsum(
        "ai,ai->a",
        displacement.values[inner],
        mesh.points[inner] / radius[inner].reshape(-1, 1),
    )
    assert np.allclose(u_r, u_r_inner, rtol=2e-3)
    assert np.allclose(outer.results.gap[:, 0], 0, atol=1e-8)


def test_cylindrical_constraint_twist():
    "A tube in a rigid sleeve is twisted by a large rotation of an end face."

    mesh = tube(length=2.0, n=(5, 3), nphi=17)
    region = fem.RegionHexahedron(mesh)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), field)

    radius = np.linalg.norm(mesh.points[:, :2], axis=1)
    sleeve = fem.CylindricalConstraint(
        field, points=np.isclose(radius, 2), skip=(0, 1, 1), multiplier=1e4
    )
    twist = fem.CylindricalConstraint(
        field, points=np.isclose(mesh.z, 2), multiplier=1e4
    )

    angles = np.linspace(0, np.pi / 3, 5)
    table = [twist.radius.reshape(-1, 1) * [0, angle, 0] for angle in angles]
    boundaries = {"fixed": fem.Boundary(displacement, fz=0)}
    step = fem.Step([solid, sleeve, twist], ramp={twist: table}, boundaries=boundaries)
    fem.Job([step]).evaluate()

    x = mesh.points + displacement.values

    # the outer points stay on the cylinder with the undeformed radius
    assert np.allclose(np.linalg.norm(x[sleeve.points, :2], axis=1), 2, atol=1e-3)

    # the end face is rotated by 60° (rigid)
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    X = mesh.points[twist.points]
    assert np.allclose(x[twist.points], X @ rotation.T, atol=1e-3)


def test_cylindrical_constraint_torsion():
    "A radial constraint does not change the torque of a tube in (small) torsion."

    torques = []

    for constrained in [False, True]:
        mesh = tube(length=2.0, n=(5, 3), nphi=17)
        region = fem.RegionHexahedron(mesh)
        displacement = fem.Field(region, dim=3)
        field = fem.FieldContainer([displacement])
        solid = fem.SolidBody(fem.LinearElastic(E=210.0, nu=0.3), field)
        items = [solid]

        if constrained:
            radius = np.linalg.norm(mesh.points[:, :2], axis=1)
            sleeve = fem.CylindricalConstraint(
                field,
                points=np.isclose(radius, 2) & (mesh.z < 2),
                skip=(0, 1, 1),
                multiplier=1e6,
            )
            items.append(sleeve)

        # a small rotation of the end face, linearized
        top = np.isclose(mesh.z, 2)
        X = mesh.points[top]
        angle = 1e-3
        boundaries = {
            "fixed": fem.Boundary(displacement, fz=0),
            "twist": fem.Boundary(
                displacement,
                mask=top,
                value=angle * np.column_stack([-X[:, 1], X[:, 0], 0 * X[:, 0]]),
            ),
        }
        step = fem.Step(items, boundaries=boundaries)
        fem.Job([step]).evaluate()

        force = sum([item.assemble.vector(field) for item in items]).toarray()
        force = force.reshape(-1, 3)[top]
        torques.append(np.sum(X[:, 0] * force[:, 1] - X[:, 1] * force[:, 0]))

        if constrained:
            assert np.allclose(sleeve.results.gap, 0, atol=1e-8)

    # identical up to terms of second order of the (linearized) rotation
    assert np.isclose(torques[0], torques[1], rtol=1e-5)


def test_cylindrical_constraint_mixed():
    "A cylindrical constraint in a job with a mixed field."

    mesh = tube(n=(3, 2), nphi=9)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldsMixed(region, n=3)
    umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1.0, bulk=50.0))
    solid = fem.SolidBody(umat, field)

    radius = np.linalg.norm(mesh.points[:, :2], axis=1)
    press = fem.CylindricalConstraint(
        field,
        points=np.isclose(radius, 2),
        skip=(0, 1, 1),
        displacement=(-0.05, 0, 0),
    )
    boundaries = {
        "bottom": fem.Boundary(field[0], fz=0, skip=(1, 1, 0)),
        "symmetry": fem.Boundary(
            field[0], mask=np.isclose(mesh.y, 0) & (mesh.x > 0), skip=(1, 0, 1)
        ),
    }

    r = press.assemble.vector(field)
    K = press.assemble.matrix(field)
    assert r.shape == (field[0].values.size, 1)
    assert K.shape == (field[0].values.size, field[0].values.size)

    step = fem.Step([solid, press], boundaries=boundaries)
    fem.Job([step]).evaluate()

    x = mesh.points + field[0].values
    radius_deformed = np.linalg.norm(x[press.points, :2], axis=1)
    assert np.allclose(radius_deformed, 1.95, atol=1e-3)


def test_cylindrical_constraint_update():
    "The prescribed displacements are broadcasted to all points."

    mesh = tube(n=(2, 2), nphi=5)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    constraint = fem.CylindricalConstraint(field, points=[0, 1, 2])
    assert constraint.displacement.shape == (3, 3)
    assert np.allclose(constraint.displacement, 0)

    constraint.update([1, 2, 3])
    assert np.allclose(constraint.displacement, [[1, 2, 3]] * 3)

    constraint.update(np.arange(9).reshape(3, 3))
    assert np.allclose(constraint.displacement, np.arange(9).reshape(3, 3))

    with pytest.raises(ValueError):
        constraint.update([1, 2, 3, 4])

    with pytest.raises(ValueError):
        constraint.update(np.zeros((4, 3)))

    # 2d-meshes support displacements with two components
    mesh = annulus(n=(2, 2))
    field = fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2).as_container()
    constraint = fem.CylindricalConstraint(field, points=[0, 1], displacement=(1, 2))
    assert np.allclose(constraint.displacement, [[1, 2, 0]] * 2)
    assert not constraint.mask[2]


def test_cylindrical_constraint_errors():
    mesh = fem.Cube(a=(-1, -1, 0), b=(1, 1, 1), n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    on_axis = np.isclose(mesh.x, 0) & np.isclose(mesh.y, 0)

    # points on the axis can't be constrained in radial or circumferential direction
    with pytest.raises(ValueError):
        fem.CylindricalConstraint(field, points=on_axis)

    with pytest.raises(ValueError):
        fem.CylindricalConstraint(field, points=on_axis, skip=(1, 0, 1))

    # but in axial direction
    constraint = fem.CylindricalConstraint(field, points=on_axis, skip=(1, 1, 0))
    r = constraint.assemble.vector(field)
    K = constraint.assemble.matrix(field)
    assert np.all(np.isfinite(r.toarray()))
    assert np.all(np.isfinite(K.toarray()))

    with pytest.raises(ValueError):
        fem.CylindricalConstraint(field, points=[0], axis=(0, 0, 0))

    # the axis must be perpendicular to 2d-meshes
    mesh = annulus(n=(2, 2))
    field = fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2).as_container()
    with pytest.raises(ValueError):
        fem.CylindricalConstraint(field, points=[0], axis=(1, 0, 0))

    # 1d-meshes are not supported
    mesh = fem.mesh.Line(n=3)
    field = fem.FieldContainer([fem.Field(fem.RegionTruss(mesh), dim=1)])
    with pytest.raises(ValueError):
        fem.CylindricalConstraint(field, points=[0])


def test_cylindrical_constraint_empty_and_plot():
    mesh = tube(n=(2, 2), nphi=5)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    constraint = fem.CylindricalConstraint(field, points=[])
    assert constraint.assemble.vector(field).nnz == 0
    assert constraint.assemble.matrix(field).nnz == 0

    constraint = fem.CylindricalConstraint(field, points=[0, 1])

    try:
        constraint.plot()
        constraint.plot(deformed=False)
    except ModuleNotFoundError:
        pass


if __name__ == "__main__":
    test_cylindrical_constraint_derivatives()
    test_cylindrical_constraint_target()
    test_cylindrical_constraint_lame()
    test_cylindrical_constraint_twist()
    test_cylindrical_constraint_torsion()
    test_cylindrical_constraint_mixed()
    test_cylindrical_constraint_update()
    test_cylindrical_constraint_errors()
    test_cylindrical_constraint_empty_and_plot()
