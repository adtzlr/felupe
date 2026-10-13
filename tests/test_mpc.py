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
    cpoint = -1

    RBE2 = fem.MultiPointConstraint(fields, points=f1(mesh.x), centerpoint=cpoint)
    CONT = fem.MultiPointContact(fields, points=f1(mesh.x), centerpoint=cpoint)

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
    mesh.add_points([10, 10])
    field = fem.FieldContainer([fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2)])

    plane = fem.MultiPointContact(field, [], -1, skip=(0, 1))
    plane = fem.MultiPointContact(field, [0, 1], -1, skip=(0, 1))

    try:
        plotter = mesh.plot(off_screen=True)
        plane.plot(plotter=plotter, line_width=8)
    except ModuleNotFoundError:
        pass

    mpc = fem.MultiPointConstraint(field, [], -1)
    mpc = fem.MultiPointConstraint(field, [0, 1], -1)

    try:
        mpc.plot()
    except ModuleNotFoundError:
        pass


def test_mpc_reference():
    "Compare the assembled vectors and matrices with dense reference arrays."

    rng = np.random.default_rng(seed=7)

    for dim in [2, 3]:
        mesh = fem.Cube(n=3) if dim == 3 else fem.Rectangle(n=4)
        mesh.add_points([np.full(dim, 0.5)])
        region = fem.RegionHexahedron(mesh) if dim == 3 else fem.RegionQuad(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=dim)])
        field[0].values[:] = 0.6 * rng.uniform(-1, 1, size=field[0].values.shape)

        u = field[0].values
        x = mesh.points + u
        points = np.arange(mesh.npoints - 1)
        c = mesh.npoints - 1
        dof = np.arange(mesh.ndof).reshape(mesh.points.shape)

        for skip in [(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0)]:
            mask = ~np.array(skip, dtype=bool)[:dim]

            for Item in [fem.MultiPointConstraint, fem.MultiPointContact]:
                item = Item(field, points, centerpoint=-1, skip=skip, multiplier=7.0)
                r = item.assemble.vector(field).toarray().ravel()
                K = item.assemble.matrix(field).toarray()

                # springs between the center-point and the points, i.e. the residual
                # vector k (y_t - y_c) (e_t - e_c) and the stiffness k (e_t - e_c)^2,
                # with the displacements (constraint) or the positions (contact) y
                y = x if Item is fem.MultiPointContact else u
                r_ref = np.zeros(mesh.ndof)
                K_ref = np.zeros((mesh.ndof, mesh.ndof))

                for t in points:
                    for ax in np.arange(dim)[mask]:
                        if Item is fem.MultiPointContact:
                            sign = np.sign(mesh.points[c, ax] - mesh.points[t, ax])
                            if sign == np.sign(x[c, ax] - x[t, ax]):
                                continue  # not in contact

                        e = np.zeros(mesh.ndof)
                        e[dof[t, ax]], e[dof[c, ax]] = 1.0, -1.0

                        r_ref += 7.0 * (y[t, ax] - y[c, ax]) * e
                        K_ref += 7.0 * np.outer(e, e)

                assert np.allclose(r, r_ref)
                assert np.allclose(K, K_ref)


def test_mpc_rotation_matrix():
    "Check the rotation matrix and its derivatives w.r.t. the rotation-vector."

    from scipy.spatial.transform import Rotation

    from felupe.mechanics._multipoint._rodrigues import _SKEW_BASE, _rotation

    rng = np.random.default_rng(seed=5)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    # the power series is used for norms < 1 (incl. 1 - 1e-9), the closed form else
    norms = [0, 1e-8, 0.3, 1 - 1e-9, 1 + 1e-9, 2.5, np.pi, 2 * np.pi, 9.0]
    h = 1e-6

    for norm in norms:
        w = norm * direction
        Q, dQ, ddQ = _rotation(w)

        assert np.allclose(Q, Rotation.from_rotvec(w).as_matrix(), atol=1e-14)
        assert np.allclose(Q, _rotation(w, derivatives=False))
        assert np.allclose(ddQ, ddQ.transpose([1, 0, 2, 3]))

        # central differences of the rotation matrix and its first derivative
        dQ_fd = []
        ddQ_fd = []

        for e in np.eye(3):
            dQ_fd.append(_rotation(w + h * e, False) - _rotation(w - h * e, False))
            ddQ_fd.append(_rotation(w + h * e)[1] - _rotation(w - h * e)[1])

        assert np.allclose(dQ, np.array(dQ_fd) / (2 * h), atol=1e-8)
        assert np.allclose(ddQ, np.array(ddQ_fd) / (2 * h), atol=1e-8)

    # derivatives at zero
    E = _SKEW_BASE
    Q, dQ, ddQ = _rotation(np.zeros(3))
    EE = E[:, None] @ E[None, :]

    assert np.allclose(Q, np.eye(3))
    assert np.allclose(dQ, E)
    assert np.allclose(ddQ, (EE + EE.transpose([1, 0, 2, 3])) / 2)


def test_mpc_rotation_matrix_jax():
    "Compare the derivatives of the rotation matrix with automatic differentiation."

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from felupe.mechanics._multipoint._rodrigues import _rotation

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)

    try:

        def rodrigues(w):
            t = jnp.sqrt(w @ w)
            S = jnp.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            return jnp.eye(3) + jnp.sin(t) / t * S + (1 - jnp.cos(t)) / t**2 * S @ S

        jacobian = jax.jacfwd(rodrigues)
        hessian = jax.jacfwd(jacobian)

        rng = np.random.default_rng(seed=6)

        for norm in [0.4, 1.2, 4.0, 8.0]:
            w = rng.normal(size=3)
            w *= norm / np.linalg.norm(w)

            Q, dQ, ddQ = _rotation(w)

            assert np.allclose(Q, rodrigues(w))
            assert np.allclose(dQ, np.moveaxis(jacobian(w), -1, 0))
            assert np.allclose(ddQ, np.moveaxis(hessian(w), (-2, -1), (0, 1)))

    finally:
        jax.config.update("jax_enable_x64", x64)


def mpc_rotation_model(dim, seed=None, increment=0.3):
    """A unit cube (square) with a center-point and a rotation-point. The points are
    connected to both. If a seed is given, the displacements and the state variables of
    the rotation are random."""

    mesh = fem.Cube(n=3) if dim == 3 else fem.Rectangle(n=3)
    mesh.add_points(np.full((2, dim), 0.5))
    region = fem.RegionHexahedron(mesh) if dim == 3 else fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=dim)])
    points = np.arange(mesh.npoints - 2)

    mpc = fem.MultiPointConstraint(
        field, points, centerpoint=-2, rotationpoint=-1, multiplier=7.0
    )

    if seed is not None:
        from felupe.mechanics._multipoint._rodrigues import _rotation

        rng = np.random.default_rng(seed=seed)
        axes = mpc._rotation_axes

        psi_n = rng.normal(size=dim) @ axes
        rotation_n = _rotation(rng.normal(size=dim) @ axes, derivatives=False)
        mpc.results.statevars = np.concatenate([psi_n, rotation_n.ravel()])

        field[0].values[:] = 0.2 * rng.normal(size=field[0].values.shape)
        field[0].values[-1] = axes @ psi_n + increment * rng.normal(size=dim)

    return mesh, field, mpc


def test_mpc_rotation_tangent():
    "Compare the stiffness matrix with central differences of the residuals."

    h = 1e-5

    for dim in [2, 3]:
        for seed, increment in enumerate([0.0, 1e-4, 0.3, 2.0, 5.0]):
            mesh, field, mpc = mpc_rotation_model(dim, seed=seed, increment=increment)

            u0 = field[0].values.copy()
            K = mpc.assemble.matrix(field).toarray()
            K_fd = np.zeros_like(K)

            for j in range(u0.size):
                for sign in [1, -1]:
                    u = u0.copy().ravel()
                    u[j] += sign * h
                    field[0].values[:] = u.reshape(u0.shape)
                    r = mpc.assemble.vector(field).toarray().ravel()
                    K_fd[:, j] += sign * r / (2 * h)

            field[0].values[:] = u0

            assert np.allclose(K, K.T)
            assert np.allclose(K, K_fd, rtol=1e-7, atol=1e-7)


def test_mpc_rotation_residuals():
    """Without an increment of the rotation, the residuals of the points and the
    center-point are those of the springs w.r.t. the rotated arms and the residuals of
    the rotation-point are equal to the moment."""

    for dim in [2, 3]:
        mesh, field, mpc = mpc_rotation_model(dim, seed=3, increment=0.0)

        r = mpc.assemble.vector(field).toarray().reshape(-1, dim)
        u = field[0].values
        statevars = mpc.results.statevars
        rotation_n = statevars[3:].reshape(3, 3)[:dim, :dim]

        dX = mesh.points[mpc.points] - mesh.points[mpc.centerpoint]
        dx = dX + u[mpc.points] - u[mpc.centerpoint]
        force = 7.0 * (dx - dX @ rotation_n.T)

        assert np.allclose(r[mpc.points], force)
        assert np.allclose(r[mpc.centerpoint], -force.sum(axis=0))
        assert np.allclose(mpc.results.rotation, statevars[3:].reshape(3, 3))
        assert np.allclose(mpc.results._statevars, statevars)

        if dim == 3:
            assert np.allclose(r[-1], mpc.results.moment)
        else:
            assert np.allclose(r[-1], [mpc.results.moment[2], 0])

        # with an undeformed state, the residuals are equal to the ones without
        # rotation-point
        field[0].values[:] = 0.0
        mpc.results.statevars = np.concatenate([np.zeros(3), np.eye(3).ravel()])
        field[0].values[:-2] = np.random.default_rng(4).normal(size=(len(dX), dim))

        mpc_locked = fem.MultiPointConstraint(
            field, mpc.points, centerpoint=-2, multiplier=7.0
        )
        r = mpc.assemble.vector(field).toarray()
        r_locked = mpc_locked.assemble.vector(field).toarray()
        K = mpc.assemble.matrix(field).toarray()
        K_locked = mpc_locked.assemble.matrix(field).toarray()

        dof = np.arange(mesh.ndof).reshape(mesh.points.shape)
        translational = dof[:-1].ravel()

        assert np.allclose(r[translational], r_locked[translational])
        assert np.allclose(
            K[np.ix_(translational, translational)],
            K_locked[np.ix_(translational, translational)],
        )


def rigid_rotation(dim, psi, num):
    """A body with a center-point and a rotation-point, which are connected to the
    points of a face. Only the center-point and the rotation-point are prescribed."""

    if dim == 3:
        mesh = fem.Cube(n=3)
        mesh.add_points(np.full((2, dim), 1.0))
        displacement = fem.Field(fem.RegionHexahedron(mesh), dim=3)
    else:
        mesh = fem.Rectangle(n=3)
        mesh.add_points(np.full((2, dim), 1.0))
        displacement = fem.FieldPlaneStrain(fem.RegionQuad(mesh), dim=2)

    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)

    face = np.isclose(mesh.x, 1)
    face[-2:] = False

    mpc = fem.MultiPointConstraint(field, face, centerpoint=-2, rotationpoint=-1)

    mask = np.zeros((mesh.npoints, 2), dtype=bool)
    mask[[-2, -1], [0, 1]] = True
    boundaries = {
        "center": fem.Boundary(field[0], mask=mask[:, 0]),
        "rotation": fem.Boundary(field[0], mask=mask[:, 1]),
    }

    table = np.linspace(0, 1, num + 1).reshape(-1, 1) * np.asarray(psi)
    step = fem.Step(
        [solid, mpc],
        boundaries=boundaries,
        ramp={boundaries["rotation"]: table},
    )
    job = fem.Job([step]).evaluate(verbose=0)

    return mesh, field, solid, mpc, job


def test_mpc_rotation_rigid():
    """A rotation of a free body about a fixed axis is a rigid body rotation, also for
    rotation angles beyond 2π."""

    from scipy.spatial.transform import Rotation

    for dim, psi in [(3, [2.0, -3.0, 6.0]), (2, [7.5, 0.0])]:
        mesh, field, solid, mpc, job = rigid_rotation(dim, psi, num=12)

        # the first value of the rotation-point of 2d-meshes is the rotation about z
        psi3 = np.array([0.0, 0.0, psi[0]]) if dim == 2 else np.array(psi)
        rotation = Rotation.from_rotvec(psi3).as_matrix()

        assert np.allclose(mpc.results.rotation, rotation)
        assert np.allclose(mpc.results.statevars[:3], psi3)
        assert np.allclose(field[0].values[-1], psi)

        # displacements of a rigid rotation about the (fixed) center-point
        dX = np.pad(mesh.points[:-2] - mesh.points[-2], ((0, 0), (0, 3 - dim)))
        u = (dX @ rotation.T - dX)[:, :dim]

        assert np.allclose(field[0].values[:-2], u)
        assert np.allclose(solid.evaluate.stress(), 0, atol=1e-6)
        assert np.allclose(mpc.results.moment, 0, atol=1e-6)


def torsion(turns, num, items=(), plugins=None, load=None):
    """A beam with a fixed left end face. The right end face is connected to a
    center-point and a rotation-point. Its rotation about the axis of the beam is
    ramped, the other rotations are free."""

    mesh = fem.Cube(b=(6, 1, 1), n=(13, 3, 3))
    mesh.add_points([[6.0, 0.5, 0.5], [6.0, 0.5, 0.5]])
    region = fem.RegionHexahedron(mesh)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)

    face = np.isclose(mesh.x, 6)
    face[-2:] = False
    mpc = fem.MultiPointConstraint(field, face, centerpoint=-2, rotationpoint=-1)

    centerpoint = np.zeros(mesh.npoints, dtype=bool)
    rotationpoint = np.zeros(mesh.npoints, dtype=bool)
    centerpoint[-2] = rotationpoint[-1] = True

    boundaries = {
        "fixed": fem.Boundary(displacement, fx=0),
        "center": fem.Boundary(displacement, mask=centerpoint),
    }

    if load is None:
        boundaries["twist"] = fem.Boundary(
            displacement, mask=rotationpoint, skip=(0, 1, 1)
        )
        ramp = {boundaries["twist"]: fem.math.linsteps([0, 2 * np.pi * turns], num)}
        items = [solid, mpc, *items]
    else:
        moment = fem.PointLoad(field, points=[-1])
        ramp = {moment: fem.math.linsteps([0, load], num, axis=0, axes=3)}
        items = [solid, mpc, moment, *items]

    step = fem.Step(items, boundaries=boundaries, ramp=ramp)
    job = fem.Job([step], plugins=plugins).evaluate(verbose=0)

    return mesh, displacement, solid, mpc, boundaries, job


def test_mpc_rotation_torsion():
    """The torsion of a beam by 1.5 revolutions with free tilts of the end face. The
    middle cross-section is rotated by half of the angle, i.e. no revolution is skipped
    and the singularity of the rotation-vector at 2π is avoided. The moment of the
    constraint is in equilibrium with the reaction forces of the fixed end face."""

    mesh, displacement, solid, mpc, boundaries, job = torsion(turns=1.5, num=24)
    angle = 3 * np.pi

    assert np.isclose(displacement.values[-1, 0], angle)
    assert np.allclose(displacement.values[-1, 1:], 0, atol=1e-8)
    assert np.allclose(mpc.results.statevars[:3], [angle, 0, 0], atol=1e-8)
    assert np.allclose(mpc.results.rotation, np.diag([1.0, -1.0, -1.0]), atol=1e-8)

    # rotation of the outer points of the middle cross-section (by symmetry)
    x = mesh.points + displacement.values
    dX = mesh.points[:, 1:] - 0.5
    dx = x[:, 1:] - x[-2, 1:]
    middle = np.isclose(mesh.x, 3) & (np.linalg.norm(dX, axis=1) > 0.4)
    middle[-2:] = False

    phi = np.arctan2(dx[middle, 1], dx[middle, 0])
    phi -= np.arctan2(dX[middle, 1], dX[middle, 0])

    assert np.allclose(np.exp(1j * phi), np.exp(1j * angle / 2), atol=1e-3)

    # equilibrium of moments w.r.t. the center-point
    fixed = boundaries["fixed"].points
    force = solid.results.force.toarray().reshape(-1, 3)
    moment = np.cross(x[fixed] - x[-2], force[fixed]).sum(axis=0)
    residuals = mpc.results.force.toarray().reshape(-1, 3)

    assert mpc.results.moment[0] > 0
    assert np.allclose(mpc.results.moment + moment, 0, atol=1e-8)
    assert np.isclose(residuals[-1, 0], mpc.results.moment[0])


def test_mpc_rotation_pointload():
    """A point load at the rotation-point is work-conjugate to the rotation-vector. For
    the rotation about a fixed axis, it is equal to the moment."""

    load = 0.05
    mesh, displacement, solid, mpc, boundaries, job = torsion(
        turns=None, num=6, load=load
    )

    assert displacement.values[-1, 0] > 1
    assert np.allclose(mpc.results.moment, [load, 0, 0], atol=1e-8)

    # the same rotation, prescribed instead of a point load
    turns = displacement.values[-1, 0] / (2 * np.pi)
    _, _, _, mpc_ref, _, _ = torsion(turns=turns, num=6)

    assert np.allclose(mpc_ref.results.moment, [load, 0, 0], atol=1e-6)


class FailingNewton(fem.Plugin):
    """Raise an error in the second Newton iteration of the evaluations ``calls`` of the
    Newton-Raphson method (zero-based, counted during the whole job), i.e. after the
    items were evaluated for the updated unknowns of the first iteration."""

    def __init__(self, calls):
        self.calls = set(calls)
        self.count = -1
        self.raised = 0

    def before_newton(self, context, state):
        self.count += 1

    def after_linear_solve(self, context, state):
        if self.count in self.calls and state.iteration == 1:
            self.raised += 1
            raise fem.NewtonConvergenceError("Injected failure.")


def test_mpc_rotation_cutback():
    """The state variables of the rotation are only updated for converged increments.
    A subdivided substep is identical to explicit substeps and, because the increments
    of a substep are collinear, also to the direct substep. The rotation of a path with
    varying axes is path-dependent."""

    from scipy.spatial.transform import Rotation

    def run(table, plugins=None):
        mesh = fem.Cube(n=3)
        mesh.add_points([[1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
        solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)

        face = np.isclose(mesh.x, 1)
        face[-2:] = False
        mpc = fem.MultiPointConstraint(field, face, centerpoint=-2, rotationpoint=-1)

        mask = np.zeros((mesh.npoints, 2), dtype=bool)
        mask[[-2, -1], [0, 1]] = True
        boundaries = {
            "fixed": fem.Boundary(field[0], fx=0),
            "center": fem.Boundary(field[0], mask=mask[:, 0], skip=(1, 0, 0)),
            "rotation": fem.Boundary(field[0], mask=mask[:, 1]),
        }
        step = fem.Step(
            [solid, mpc], boundaries=boundaries, ramp={boundaries["rotation"]: table}
        )
        fem.Job([step], plugins=plugins).evaluate(verbose=0)

        return field, mpc

    table = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.4, 0.0]])

    # fail the last substep after one Newton iteration
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[2])
    field, mpc = run(table, plugins=[failing, cutback])

    assert failing.raised == 1
    assert cutback.load_factors == [[1.0], [1.0], [0.5, 1.0]]

    # explicit substeps
    refined = np.array([*table[:2], (table[1] + table[2]) / 2, table[2]])
    field_ref, mpc_ref = run(refined)

    assert np.allclose(field[0].values, field_ref[0].values, rtol=1e-10, atol=1e-12)
    assert np.allclose(mpc.results.statevars, mpc_ref.results.statevars, atol=1e-12)

    # the direct substep (without cutback)
    field_direct, mpc_direct = run(table)

    assert np.allclose(field[0].values, field_direct[0].values, atol=1e-8)
    assert np.allclose(mpc.results.statevars, mpc_direct.results.statevars, atol=1e-8)

    # the rotation is path-dependent: the rotation of the second substep is applied to
    # the rotation of the first substep, i.e. it differs from the rotation-vector of
    # the accumulated values
    rotation_x = Rotation.from_rotvec(table[1]).as_matrix()
    rotation_y = Rotation.from_rotvec(table[2] - table[1]).as_matrix()
    rotation = rotation_y @ rotation_x

    assert np.allclose(mpc.results.rotation, rotation)
    assert np.allclose(mpc.results.statevars[:3], table[2])
    assert not np.allclose(rotation, Rotation.from_rotvec(table[2]).as_matrix())


def test_mpc_rotation_checkpoint():
    "Restore a checkpoint of the state variables of the rotation."

    mesh, field, mpc = mpc_rotation_model(dim=3, seed=8)
    checkpoint = mpc.checkpoint()
    statevars = mpc.results.statevars.copy()

    mpc.assemble.vector(field)
    mpc.assemble.matrix(field)
    mpc.results.update_statevars()

    assert not np.allclose(mpc.results.statevars, statevars)

    mpc.restore(checkpoint)

    assert np.allclose(mpc.results.statevars, statevars)
    assert mpc.results.statevars is not checkpoint["results.statevars"]
    assert mpc.results._statevars is None
    assert mpc.results.force is None

    # without a rotation-point, there are no state variables
    mpc = fem.MultiPointConstraint(field, mpc.points, centerpoint=-2)
    checkpoint = mpc.checkpoint()

    assert checkpoint["results.statevars"] is None

    mpc.restore(checkpoint)
    mpc.results.update_statevars()

    assert mpc.results.statevars is None


def test_mpc_rotation_errors():
    "Check the errors of invalid rotation-points."

    mesh = fem.Cube(n=3)
    mesh.add_points([[1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    points = np.arange(4)

    with pytest.raises(ValueError, match="Skipped axes"):
        fem.MultiPointConstraint(
            field, points, centerpoint=-2, rotationpoint=-1, skip=(0, 1, 0)
        )

    with pytest.raises(ValueError, match="must differ"):
        fem.MultiPointConstraint(field, points, centerpoint=-2, rotationpoint=-2)

    with pytest.raises(ValueError, match="must differ"):
        fem.MultiPointConstraint(field, points, centerpoint=-2, rotationpoint=0)

    # the center- and the rotation-point are not treated as points without cells
    fem.MultiPointConstraint(field, points, centerpoint=-2, rotationpoint=-1)

    assert len(mesh.points_without_cells) == 0

    # 1d-meshes are not supported
    mesh = fem.mesh.Line(n=3)
    mesh.add_points([[1.0], [1.0]])
    region = fem.Region(mesh, fem.Line(), fem.GaussLegendre(order=1, dim=1))
    field = fem.FieldContainer([fem.Field(region, dim=1)])

    with pytest.raises(ValueError, match="2d- or a 3d-mesh"):
        fem.MultiPointConstraint(field, [0, 1], centerpoint=-2, rotationpoint=-1)


def test_mpc_rotation_dtype():
    "The vector and the state variables keep the data type of the field."

    mesh = fem.Rectangle(n=3)
    mesh.add_points([[1.0, 0.5], [1.0, 0.5]])
    region = fem.RegionQuad(mesh).astype(np.float32, copy=False)
    field = fem.FieldContainer([fem.Field(region, dim=2, dtype=np.float32)])
    field[0].values[-1] = [0.1, 0.2]

    mpc = fem.MultiPointConstraint(
        field, np.arange(4), centerpoint=-2, rotationpoint=-1
    )
    r = mpc.assemble.vector(field).toarray().reshape(-1, 2)

    assert r.dtype == np.float32
    assert mpc.results.statevars.dtype == np.float32
    assert mpc.results._statevars.dtype == np.float32

    # the unused second value of the rotation-point is locked
    assert np.isclose(r[-1, 1], 1e3 * 0.2)


if __name__ == "__main__":
    test_mpc_reference()
    test_mpc()
    test_mpc_mixed()
    test_mpc_isolated()
    test_mpc_plot_2d()
    test_mpc_rotation_matrix()
    test_mpc_rotation_matrix_jax()
    test_mpc_rotation_tangent()
    test_mpc_rotation_residuals()
    test_mpc_rotation_rigid()
    test_mpc_rotation_torsion()
    test_mpc_rotation_pointload()
    test_mpc_rotation_cutback()
    test_mpc_rotation_checkpoint()
    test_mpc_rotation_errors()
    test_mpc_rotation_dtype()
