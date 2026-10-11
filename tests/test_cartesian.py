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


def test_cartesian_constraint_derivatives():
    "Check the vector and the matrix by finite differences of the potential."

    rng = np.random.default_rng(seed=56)

    for dim in [2, 3]:
        if dim == 2:
            mesh = fem.Rectangle(n=(3, 4))
            region = fem.RegionQuad(mesh)
            basis = fem.math.rotation_matrix(25, dim=2)
            displacement = (0.05, 0.3)
            skips = [(0, 0), (0, 1), (1, 0)]
        else:
            mesh = fem.Cube(a=(1, 0.5, -1), b=(2, 1, 0), n=3)
            region = fem.RegionHexahedron(mesh)
            basis = np.column_stack([[1, 1, 0], [-1, 1, 1], [1, -1, 2]])
            displacement = (0.05, 0.3, -0.1)
            skips = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 1)]

        field = fem.FieldContainer([fem.Field(region, dim=dim)])
        field[0].values[:] = 0.2 * rng.uniform(-1, 1, size=field[0].values.shape)

        for skip in skips:
            constraint = fem.CartesianConstraint(
                field,
                points=np.arange(mesh.npoints),
                basis=basis,
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

            for i in range(u.size):
                du = np.zeros(u.size)
                du[i] = h
                up = (u.ravel() + du).reshape(u.shape)
                um = (u.ravel() - du).reshape(u.shape)
                r_fd[i] = (potential(up) - potential(um)) / (2 * h)

            field[0].values[:] = u

            # the gaps are linear, i.e. the residuals are linear in the displacements
            field[0].values[:] = 0
            r_zero = constraint.assemble.vector(field).toarray().ravel()
            field[0].values[:] = u

            assert np.allclose(r, r_fd, atol=1e-6)
            assert np.allclose(r, r_zero + K @ u.ravel())
            assert np.allclose(K, K.T)

            # the skipped directions do not contribute
            constraint.assemble.vector(field)
            mask = np.pad(np.array(skip, bool), (0, 3 - len(skip)), constant_values=1)
            assert np.allclose(constraint.results.gap[:, mask], 0)


def test_cartesian_constraint_target():
    "A fully constrained mesh is moved to the prescribed (ramped) displacements."

    for dim in [2, 3]:
        if dim == 2:
            mesh = fem.Rectangle(n=3)
            region = fem.RegionQuad(mesh)
            basis = fem.math.rotation_matrix(-40, dim=2)
            displacement = np.array([0.4, -0.3])
        else:
            mesh = fem.Cube(n=3)
            region = fem.RegionHexahedron(mesh)
            basis = fem.math.rotation_matrix(30, dim=3, axis=2)
            displacement = np.array([0.4, 0.9, -0.3])

        field = fem.FieldContainer([fem.Field(region, dim=dim)])
        constraint = fem.CartesianConstraint(
            field, points=np.ones(mesh.npoints, dtype=bool), basis=basis
        )
        table = [factor * displacement for factor in [0, 0.5, 1]]
        step = fem.Step([constraint], ramp={constraint: table})
        fem.Job([step]).evaluate()

        assert np.allclose(field[0].values, displacement @ basis.T)
        assert np.allclose(constraint.results.gap, 0)


def test_cartesian_constraint_rotated():
    "An inclined support of a rotated block is equal to a roller support."

    results = []

    for dim in [2, 3]:
        if dim == 2:
            rotation = fem.math.rotation_matrix(30, dim=2)
            mesh = fem.Rectangle(n=4)
            region = fem.RegionQuad(mesh)
            value = np.array([0.1, -0.2])
            skip = (1, 0)
        else:
            rotation = fem.math.rotation_matrix(30, dim=3, axis=1)
            mesh = fem.Cube(n=4)
            region = fem.RegionHexahedron(mesh)
            value = np.array([0.1, 0.05, -0.2])
            skip = (1, 1, 0)

        bottom = np.isclose(mesh.points[:, -1], 0)
        top = np.isclose(mesh.points[:, -1], 1)

        for rotated in [False, True]:
            if rotated:
                mesh = mesh.rotate(30, axis=1)
                region = type(region)(mesh)

            if dim == 2:
                displacement = fem.FieldPlaneStrain(region, dim=2)
            else:
                displacement = fem.Field(region, dim=3)

            field = fem.FieldContainer([displacement])
            solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), field)
            items = [solid]

            if rotated:
                support = fem.CartesianConstraint(
                    field, points=bottom, basis=rotation, skip=skip, multiplier=1e7
                )
                items.append(support)
                boundaries = {
                    "top": fem.Boundary(displacement, mask=top, value=rotation @ value)
                }
            else:
                boundaries = {
                    "bottom": fem.Boundary(displacement, mask=bottom, skip=skip),
                    "top": fem.Boundary(displacement, mask=top, value=value),
                }

            step = fem.Step(items, boundaries=boundaries)
            fem.Job([step]).evaluate()
            results.append(displacement.values.copy())

        u, u_rotated = results[-2:]

        # the in-plane displacements of the bottom face are not zero
        assert not np.allclose(u[bottom, :-1], 0)
        assert np.allclose(support.results.gap, 0, atol=1e-6)
        assert np.allclose(u_rotated, u @ rotation.T, atol=1e-6)


def test_cartesian_constraint_update():
    "The prescribed displacements are broadcasted to all points."

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    constraint = fem.CartesianConstraint(field, points=[0, 1, 2])
    assert constraint.displacement.shape == (3, 3)
    assert np.allclose(constraint.displacement, 0)
    assert np.allclose(constraint.basis, np.eye(3))

    constraint.update([1, 2, 3])
    assert np.allclose(constraint.displacement, [[1, 2, 3]] * 3)

    constraint.update(np.arange(9).reshape(3, 3))
    assert np.allclose(constraint.displacement, np.arange(9).reshape(3, 3))

    with pytest.raises(ValueError):
        constraint.update([1, 2, 3, 4])

    with pytest.raises(ValueError):
        constraint.update(np.zeros((4, 3)))

    # 2d-meshes support displacements with two components
    mesh = fem.Rectangle(n=3)
    field = fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2).as_container()
    constraint = fem.CartesianConstraint(
        field, points=[0, 1], basis=[[1, -2], [1, 2]], displacement=(1, 2)
    )
    assert np.allclose(constraint.displacement, [[1, 2, 0]] * 2)
    assert np.allclose(constraint.basis[:2, :2], np.array([[1, -1], [1, 1]]) / 2**0.5)
    assert np.allclose(constraint.basis[:, 2], [0, 0, 1])
    assert not constraint.mask[2]


def test_cartesian_constraint_errors():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    # wrong shape of the basis
    with pytest.raises(ValueError):
        fem.CartesianConstraint(field, points=[0], basis=np.eye(2))

    # zero base vector
    with pytest.raises(ValueError):
        fem.CartesianConstraint(
            field, points=[0], basis=[[1, 0, 0], [0, 0, 0], [0, 0, 1]]
        )

    # not orthogonal
    with pytest.raises(ValueError):
        fem.CartesianConstraint(
            field, points=[0], basis=[[1, 0, 0], [1, 1, 0], [0, 0, 1]]
        )

    # 1d-meshes are not supported
    mesh = fem.mesh.Line(n=3)
    field = fem.FieldContainer([fem.Field(fem.RegionTruss(mesh), dim=1)])
    with pytest.raises(ValueError):
        fem.CartesianConstraint(field, points=[0])


def test_cartesian_constraint_empty_and_plot():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    constraint = fem.CartesianConstraint(field, points=[])
    assert constraint.assemble.vector(field).nnz == 0
    assert constraint.assemble.matrix(field).nnz == 0

    constraint = fem.CartesianConstraint(field, points=[0, 1], skip=(0, 1, 0))

    try:
        constraint.plot()
        constraint.plot(deformed=False)
    except ModuleNotFoundError:
        pass


if __name__ == "__main__":
    test_cartesian_constraint_derivatives()
    test_cartesian_constraint_target()
    test_cartesian_constraint_rotated()
    test_cartesian_constraint_update()
    test_cartesian_constraint_errors()
    test_cartesian_constraint_empty_and_plot()
