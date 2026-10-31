# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pathlib
import tempfile

import numpy as np
import pytest

import felupe as fem


def pre():
    r = fem.RegionHexahedron(fem.Cube(n=3))

    u = fem.Field(r, dim=3)
    p = fem.Field(r)
    J = fem.Field(r, values=1.0)

    f = fem.FieldContainer((u, p, J))

    return r, f, (u, p, J)


def test_solve(tmp_path):
    r, _, (u, p, J) = pre()
    f = fem.FieldContainer([u])

    W = fem.constitution.NeoHooke(1, 3)

    F = f.extract()

    bounds = fem.dof.symmetry(u)
    dof0, dof1 = fem.dof.partition(f, bounds)

    ext0 = fem.dof.apply(f, bounds, dof0)

    L = fem.IntegralForm(W.gradient(F)[:-1], f, r.dV)
    a = fem.IntegralForm(W.hessian(F), f, r.dV, f)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fem.tools.solve(A, b, f, dof0, dof1, f.offsets, ext0)
    assert dx[0].shape == f[0].values.ravel().shape

    fem.tools.save(r, f, filename=tmp_path / "result.vtu")
    fem.tools.save(r, f, forces=b, filename=tmp_path / "result.vtu")
    fem.tools.save(
        r,
        f,
        forces=b,
        gradient=W.gradient(F),
        filename=tmp_path / "result.vtu",
    )

    for b in [L.assemble(), L.assemble().toarray(), L.assemble().toarray()[:, 0]]:
        force = fem.tools.force(f, b, bounds["symx"])
        moment = fem.tools.moment(f, b, bounds["symx"])

    for a in [2, 3, 4, 5]:
        curve = fem.tools.curve(np.arange(a), np.ones(a) * force[0])

    s = fem.math.dot(W.gradient(F)[0], fem.math.inv(fem.math.cof(F[0])))

    cauchy = fem.tools.project(fem.math.tovoigt(s), region=r, average=True)
    assert cauchy.shape == (r.mesh.npoints, 6)

    cauchy = fem.tools.project(fem.math.tovoigt(s), region=r, average=False)
    assert cauchy.shape == (r.mesh.cells.size, 6)

    cauchy = fem.tools.project(fem.math.tovoigt(s), region=r, mean=True)
    assert cauchy.shape == (r.mesh.npoints, 6)

    cauchy = fem.tools.topoints(s, region=r)
    assert cauchy.shape == (r.mesh.npoints, 3, 3)

    cauchy = fem.tools.topoints(s[:2, :2], region=r)
    assert cauchy.shape == (r.mesh.npoints, 2, 2)

    cauchy = fem.tools.topoints(s[0, 0], region=r)
    assert cauchy.shape == (r.mesh.npoints,)


def test_hello_world():
    fem.hello_world()
    fem.hello_world(pypardiso=True, parallel=True, xdmf=True)
    fem.hello_world(curve=True, axisymmetric=True)
    fem.hello_world(container=True, planestrain=True)


def test_solve_mixed(tmp_path):
    r, f, fields = pre()
    u = fields[0]

    f = fem.FieldContainer(fields)

    F, p, J = f.extract()

    W = fem.NeoHooke(1, 3)
    W_mixed = fem.ThreeFieldVariation(W)

    F = u.extract()

    bounds = fem.dof.symmetry(u)
    dof0, dof1 = fem.dof.partition(f, bounds)

    ext0 = fem.dof.apply(f, bounds, dof0)

    L = fem.IntegralForm(W_mixed.gradient([F, p, J])[:-1], f, r.dV)
    a = fem.IntegralForm(W_mixed.hessian([F, p, J]), f, r.dV, f)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fem.tools.solve(A, b, f, dof0, dof1, f.offsets, ext0)

    assert dx[0].shape == u.values.ravel().shape
    assert dx[1].shape == fields[1].values.ravel().shape
    assert dx[2].shape == fields[2].values.ravel().shape

    fem.tools.save(r, f, filename=tmp_path / "result.vtu")
    fem.tools.save(r, f, forces=b, filename=tmp_path / "result.vtu")
    fem.tools.save(
        r,
        f,
        forces=b,
        gradient=W_mixed.gradient([F, p, J]),
        filename=tmp_path / "result.vtu",
    )

    force = fem.tools.force(f, b, bounds["symx"])
    moment = fem.tools.moment(f, b, bounds["symx"])

    for a in [2, 3, 4, 5]:
        curve = fem.tools.curve(np.arange(a), np.ones(a) * force[0])


def test_newton_update():
    "The updated field container shares the geometry with the old one."

    from felupe.tools._newton import update

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    x = fem.FieldContainer([fem.Field(region, dim=3), fem.Field(region)])
    dx = np.ones(sum(x.fieldsizes))

    y = update(x, dx)

    for field, other in zip(x.fields, y.fields):
        assert other is not field
        assert other.region is field.region
        assert other.indices is field.indices
        assert other.values is not field.values
        assert np.allclose(field.values, 0)
        assert np.allclose(other.values, 1)

    # unchanged update of an array
    assert np.allclose(update(np.zeros(3), np.ones(3)), 1)


def test_newton_simple():
    def fun(x):
        return (x - 3) ** 2

    def jac(x):
        return np.array([2 * (x - 3)])

    x0 = np.array([3.1])

    os.environ["FELUPE_VERBOSE"] = "true"

    # a double root without a reference norm (no prescribed degrees of freedom and no
    # items) converges linearly, an absolute tolerance is required
    res = fem.tools.newtonraphson(
        x0, fun, jac, solve=np.linalg.solve, maxiter=32, atol=1e-11
    )

    os.environ.pop("FELUPE_VERBOSE")

    assert abs(res.fun) < 1e-11
    assert np.isclose(res.x, 3, rtol=1e-2)

    with pytest.raises(ValueError):
        res = fem.tools.newtonraphson(
            x0, fun, jac, solve=np.linalg.solve, maxiter=4, verbose=False, atol=1e-11
        )

    with pytest.raises(ValueError):
        x0 = np.array([np.nan])
        res = fem.tools.newtonraphson(x0, fun, jac, solve=np.linalg.solve)


def test_newton():
    # create a hexahedron-region on a cube
    region = fem.RegionHexahedron(fem.Cube(n=6))

    # add a displacement field and apply a uniaxial elongation on the cube
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )

    # define the constitutive material behavior
    umat = fem.NeoHooke(mu=1.0, bulk=2.0)

    for kwargs in [{"parallel": True, "umat": umat}]:
        # newton-raphson procedure
        res = fem.newtonraphson(
            field,
            verbose=True,
            kwargs=kwargs,
            **loadcase,
        )


def test_newton_plane():
    # create a quad-region on a rectangle
    region = fem.RegionQuad(fem.Rectangle(n=6))

    # add a displacement field and apply a uniaxial elongation on the rectangle
    displacement = fem.Field(region, dim=2)
    field = fem.FieldContainer([displacement])
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )

    # define the constitutive material behavior
    umat = fem.LinearElasticPlaneStress(E=1.0, nu=0.3)

    # newton-raphson procedure
    res = fem.newtonraphson(
        field,
        verbose=True,
        kwargs=dict(umat=umat),
        tqdm="auto",
        **loadcase,
    )

    # define the constitutive material behavior
    umat = fem.constitution.LinearElasticPlaneStrain(E=1.0, nu=0.3)

    with pytest.raises(ValueError):
        res = fem.newtonraphson(
            field,
            verbose=True,
            kwargs=dict(umat=umat),
            tqdm="my_fancy_backend",
            **loadcase,
        )

    # newton-raphson procedure
    res = fem.newtonraphson(
        field,
        verbose=True,
        kwargs=dict(umat=umat),
        tqdm="notebook",
        **loadcase,
    )


def test_newton_linearelastic():
    # create a hexahedron-region on a cube
    region = fem.RegionHexahedron(fem.Cube(n=6))

    # add a displacement field and apply a uniaxial elongation on the cube
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )

    # define the constitutive material behavior
    umat = fem.LinearElastic(E=1.0, nu=0.3)

    # newton-raphson procedure
    res = fem.newtonraphson(
        field,
        verbose=True,
        kwargs={"umat": umat, "grad": True, "sym": True, "add_identity": False},
        **loadcase,
    )


def test_newton_mixed():
    # create a hexahedron-region on a cube
    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    region0 = fem.RegionConstantHexahedron(fem.mesh.convert(mesh, 0))

    # add a displacement field and apply a uniaxial elongation on the cube
    u = fem.Field(region, dim=3)
    p = fem.Field(region0)
    J = fem.Field(region0, values=1.0)
    field = fem.FieldContainer((u, p, J))

    assert len(field) == 3

    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )

    # deformation gradient
    F = field.extract(grad=True, sym=False, add_identity=True)

    # define the constitutive material behavior
    nh = fem.NeoHooke(mu=1.0, bulk=2.0)
    umat = fem.ThreeFieldVariation(nh)

    # newton-raphson procedure
    res = fem.newtonraphson(x0=field, kwargs=dict(umat=umat), **loadcase)


def test_newton_body():
    # create a hexahedron-region on a cube
    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    region0 = fem.RegionConstantHexahedron(fem.mesh.convert(mesh, 0))

    # add a displacement field and apply a uniaxial elongation on the cube
    u = fem.Field(region, dim=3)
    p = fem.Field(region0)
    J = fem.Field(region0, values=1.0)
    field = fem.FieldContainer((u, p, J))

    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )

    # define the constitutive material behavior
    nh = fem.NeoHooke(mu=1.0, bulk=2.0)
    umat = fem.ThreeFieldVariation(nh)
    body = fem.SolidBody(umat, field)

    # create a region, a field and a body for a pressure boundary
    regionp = fem.RegionHexahedronBoundary(mesh, only_surface=True)
    fieldp = fem.FieldContainer([fem.Field(regionp, dim=3)])
    bodyp = fem.SolidBodyPressure(fieldp, pressure=1.0)

    # newton-raphson procedure
    res = fem.newtonraphson(
        items=[body, bodyp],
        kwargs={},
        **loadcase,
    )


def test_newton_check():
    "The convergence criterion with an absolute tolerance and a reference norm."

    from felupe.tools._newton import check

    f = np.array([3.0, 4.0, 1.0, 0.0])
    dx = np.zeros(4)
    dof1 = np.array([0, 1])
    dof0 = np.array([2, 3])

    # the reference norm is the norm on the prescribed degrees of freedom
    xnorm, fnorm, success = check(dx, None, f, np.inf, 0.1, dof1, dof0)
    assert np.isclose(fnorm, 5.0)
    assert not success

    # a given reference norm and an absolute tolerance
    xnorm, fnorm, success = check(dx, None, f, np.inf, 0.1, dof1, dof0, reference=4.0)
    assert np.isclose(fnorm, 5.0 / 4.0)

    xnorm, fnorm, success = check(
        dx, None, f, np.inf, 0.1, dof1, dof0, atol=4.0, reference=20.0
    )
    assert np.isclose(fnorm, 5.0 / (20.0 + 4.0 / 0.1))
    assert success  # 5 < 4 + 0.1 * 20

    # without a reference norm and without an absolute tolerance
    zeros = np.zeros(4)
    assert check(dx, None, zeros, np.inf, 0.1, dof1, dof0)[1:] == (0.0, True)
    xnorm, fnorm, success = check(dx, None, f * [1, 1, 0, 0], np.inf, 0.1, dof1, dof0)
    assert np.isinf(fnorm)
    assert not success

    # all degrees of freedom are active (no prescribed degrees of freedom)
    xnorm, fnorm, success = check(dx, None, f, np.inf, 0.1, atol=1.0)
    assert np.isclose(fnorm, np.sqrt(26) / (1.0 / 0.1))


def test_newton_default_atol():
    "The default absolute tolerance is a bound of the round-off errors."

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import aslinearoperator

    from felupe.tools._newton import default_atol

    eps = np.finfo(float).eps
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    x = np.array([1.0, -1.0])

    atol = default_atol(A, A @ x, x)
    assert np.isclose(atol, 10 * eps * np.linalg.norm([3.0, 3.0]))
    assert np.isclose(default_atol(csr_matrix(A), A @ x, x), atol)
    assert np.isclose(default_atol(A, A @ x, x, dof1=[1]), 10 * eps * 3.0)
    assert default_atol(A, A @ x, np.zeros(2)) == 0.0

    # the lowest precision of the linear equation system and of the unknowns
    eps32 = np.finfo(np.float32).eps
    atol32 = default_atol(A, A @ x, x.astype(np.float32))
    assert np.isclose(atol32, 10 * eps32 * np.linalg.norm([3.0, 3.0]))

    # an upper bound for batched Jacobians (e.g. of local problems)
    B = np.ones((3, 3, 3, 3, 5))
    y = np.ones((3, 3, 5))
    bound = 10 * eps * np.linalg.norm(B) * np.linalg.norm(y)
    assert np.isclose(default_atol(B, y, y), bound)

    # linear operators are not supported
    assert default_atol(aslinearoperator(A), A @ x, x) == 0.0

    # the round-off errors of the deformation gradient F = 1 + du/dX in a force-free
    # state are evaluated in the precision of the linear equation system (of the
    # region), also for fields in single-precision
    region = fem.RegionHexahedron(fem.Cube(n=3))
    for dtype in [np.float64, np.float32]:
        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=dtype)])
        boundaries, loadcase = fem.dof.uniaxial(
            field, clamped=True, return_loadcase=True
        )
        solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)
        K = solid.assemble.matrix()
        f = solid.assemble.vector().toarray()[:, 0]
        dof1 = loadcase["dof1"]

        points = region.mesh.points
        X = (points - points.mean(axis=0)).ravel()
        expected = 10 * eps * np.linalg.norm((K @ X)[dof1])
        assert expected > 0
        assert np.isclose(default_atol(K, f, field, dof1=dof1), expected)


def test_newton_default_atol_location():
    "The default absolute tolerance doesn't depend on the location of the mesh."

    from felupe.tools._newton import default_atol

    atol = []
    iterations = []

    for offset in [0.0, 1e3, 1e9]:
        mesh = fem.Cube(n=3)
        x = mesh.x.copy()
        mesh.points += offset

        region = fem.RegionHexahedron(mesh)
        displacement = fem.Field(region, dim=3)
        field = fem.FieldContainer([displacement])
        solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)

        # a penalty-based support resists a rigid body translation
        support = fem.CartesianConstraint(
            field, points=np.arange(mesh.npoints)[np.isclose(x, 0)], multiplier=1e3
        )

        right = np.isclose(x, 1)
        boundaries = {
            "right": fem.Boundary(displacement, mask=right, skip=(1, 0, 0)),
            "move": fem.Boundary(displacement, mask=right, skip=(0, 1, 1), value=0.2),
        }
        dof0, dof1 = fem.dof.partition(field, boundaries)
        ext0 = fem.dof.apply(field, boundaries, dof0)

        res = fem.newtonraphson(
            items=[solid, support], dof0=dof0, dof1=dof1, ext0=ext0, verbose=0
        )
        iterations.append(res.iterations)

        K = fem.tools.jac([solid, support], field)
        atol.append(default_atol(K, res.fun, field, dof1=dof1))

    assert iterations[0] == iterations[1] == iterations[2]
    assert np.allclose(atol, atol[0], rtol=1e-6)


def test_newton_fun_items_reference():
    "The reference norm of the items is the largest norm without one of the items."

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)

    # a single item
    f, reference = fem.tools.fun([solid], field, return_reference=True)
    assert reference == 0.0
    assert np.allclose(f, fem.tools.fun([solid], field))

    # a pressure on all faces of the cube
    regionp = fem.RegionHexahedronBoundary(mesh)
    fieldp = fem.FieldContainer([fem.Field(regionp, dim=3)])
    pressure = fem.SolidBodyPressure(fieldp, pressure=1.0)

    f, reference = fem.tools.fun([solid, pressure], field, return_reference=True)
    norms = [
        np.linalg.norm(item.assemble.vector(item.field).toarray())
        for item in [solid, pressure]
    ]
    assert np.isclose(reference, max(norms))
    assert reference > 0


def self_equilibrated(mu=8e4, p=100.0, L=10.0, n=4):
    """A cube with a pressure on all faces and statically determinate supports, i.e.
    the reaction forces are zero."""

    mesh = fem.Cube(b=(L, L, L), n=n)
    region = fem.RegionHexahedron(mesh)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])

    regionp = fem.RegionHexahedronBoundary(mesh)
    fieldp = fem.FieldContainer([fem.Field(regionp, dim=3)])

    solid = fem.SolidBody(fem.NeoHooke(mu=mu, bulk=5 * mu), field)
    pressure = fem.SolidBodyPressure(fieldp, pressure=p)

    x, y, z = mesh.points.T
    origin = np.isclose(x, 0) & np.isclose(y, 0) & np.isclose(z, 0)
    a = np.isclose(x, L) & np.isclose(y, 0) & np.isclose(z, 0)
    b = np.isclose(x, 0) & np.isclose(y, L) & np.isclose(z, 0)

    boundaries = {
        "origin": fem.Boundary(displacement, mask=origin),
        "a": fem.Boundary(displacement, mask=a, skip=(1, 0, 0)),
        "b": fem.Boundary(displacement, mask=b, skip=(1, 1, 0)),
    }
    dof0, dof1 = fem.dof.partition(field, boundaries)
    ext0 = fem.dof.apply(field, boundaries, dof0)

    return [solid, pressure], dict(dof0=dof0, dof1=dof1, ext0=ext0)


def test_newton_self_equilibrated():
    "Forces between the items are a reference also without reaction forces."

    items, loadcase = self_equilibrated()

    class Trial(fem.Plugin):
        "Check the convergence of the trial of the full Newton increment."

        def __init__(self):
            self.converged = []
            self.success = []

        def after_linear_solve(self, context, state):
            x = context.update(state.x, state.dx)
            f = context.fun(x)
            self.converged.append(context.check(state.dx, x, f)[2])

        def after_iteration(self, context, state):
            self.success.append(state.success)

    trial = Trial()
    res = fem.newtonraphson(items=items, **loadcase, plugins=[trial], verbose=0)

    f = res.fun
    forces = np.linalg.norm(items[1].results.force.toarray())

    # the reaction forces are zero (up to round-off errors)
    assert np.linalg.norm(f[loadcase["dof0"]]) < 1e-6 * forces
    assert np.linalg.norm(f[loadcase["dof1"]]) < np.sqrt(np.finfo(float).eps) * forces

    # a plugin checks the convergence of a trial with the same reference norm
    assert trial.converged == trial.success
    assert trial.converged[-1]


def test_newton_units():
    "The convergence is independent of the units of the model."

    region = fem.RegionHexahedron(fem.Cube(n=3))
    iterations = []
    fnorms = []

    for scale in [1e-9, 1.0, 1e9]:
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        boundaries, loadcase = fem.dof.uniaxial(
            field, move=0.2, clamped=True, return_loadcase=True
        )
        umat = fem.NeoHooke(mu=scale, bulk=2 * scale)
        solid = fem.SolidBody(umat, field)
        res = fem.newtonraphson(items=[solid], **loadcase, verbose=0)
        iterations.append(res.iterations)
        fnorms.append(res.fnorms)

    assert iterations[0] == iterations[1] == iterations[2]
    assert np.allclose(fnorms[0][:-1], fnorms[1][:-1])
    assert np.allclose(fnorms[2][:-1], fnorms[1][:-1])


def test_newton_rigid_body_motion_contact():
    "A stiff solid body is moved as a rigid body to a rigid plane and compressed."

    L, gap, mu = 10.0, 1.3, 8e4  # units: mm, MPa (a soft steel)
    mesh = fem.Cube(b=(L, L, L), n=4)
    mesh.add_points([L + gap, L / 2, L / 2])

    region = fem.RegionHexahedron(mesh)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])
    solid = fem.SolidBody(fem.NeoHooke(mu=mu, bulk=5 * mu), field)
    contact = fem.ContactRigidPlane(
        field,
        points=np.arange(mesh.npoints)[np.isclose(mesh.x, L)],
        centerpoint=-1,
        normal=[-1.0, 0, 0],
        multiplier=100 * mu * L,
    )
    boundaries = {
        "move": fem.Boundary(displacement, fx=0, skip=(0, 1, 1)),
        "fixed": fem.Boundary(displacement, fx=0, skip=(1, 0, 0)),
        "plane": fem.Boundary(displacement, fx=L + gap),
    }

    # five substeps without contact, then five substeps in contact
    move = np.linspace(0, gap + 0.01 * L, 11)[1:]
    step = fem.Step(
        [solid, contact], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    job = fem.Job([step]).evaluate(verbose=0)

    # the rigid body motions converge in one iteration
    assert [len(fnorms) for fnorms in job.fnorms[:5]] == [1, 1, 1, 1, 1]
    assert np.all([fnorms[-1] < 1.5e-8 for fnorms in job.fnorms])


def test_newton_atol():
    "A given absolute tolerance is available in all hooks."

    region = fem.RegionHexahedron(fem.Cube(n=3))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)

    class Tolerance(fem.Plugin):
        def __init__(self):
            self.atol = []

        def before_newton(self, context, state):
            self.atol.append(state.atol)

        def after_iteration(self, context, state):
            self.atol.append(state.atol)

    # a given absolute tolerance
    tolerance = Tolerance()
    res = fem.newtonraphson(
        items=[solid], **loadcase, atol=1e-3, plugins=[tolerance], verbose=0
    )
    assert np.all(np.array(tolerance.atol) == 1e-3)
    assert res.iterations == 2  # a coarse absolute tolerance

    # the default absolute tolerance is evaluated in each iteration
    field.link(fem.FieldContainer([fem.Field(region, dim=3)]))
    tolerance = Tolerance()
    res = fem.newtonraphson(items=[solid], **loadcase, plugins=[tolerance], verbose=0)
    assert tolerance.atol[0] is None
    assert np.all(np.array(tolerance.atol[1:]) > 0)
    assert res.iterations == 4


def test_project():
    # rectangle (triangle)
    mesh = fem.Rectangle(n=2).triangulate()
    mesh.update(points=np.vstack([mesh.points, [1000, 1000]]))
    region = fem.RegionTriangle(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.project(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.project(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    # rectangle (quadratic triangle)
    mesh = fem.Rectangle(n=2).triangulate().add_midpoints_edges()
    region = fem.RegionQuadraticTriangle(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.project(values, region, average=True)

    # cube (tetra)
    mesh = fem.Cube(n=2).triangulate()
    region = fem.RegionTetra(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.project(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.project(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # cube (quadratic tetra)
    mesh = fem.Cube(n=2).triangulate().add_midpoints_edges()
    region = fem.RegionQuadraticTetra(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.project(values, region, average=True)


def test_project_solver():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    values = fem.Field(region, dim=3).extract()

    calls = []

    def solver(A, b):
        "A custom solver which records its calls."
        calls.append(b.shape)
        return fem.solve.SuperLUSolver()(A, b)

    projected = fem.project(values, region, solver=solver)
    assert len(calls) == 1
    assert calls[0] == (mesh.npoints, 9)
    assert np.allclose(projected, fem.project(values, region))

    from scipy.sparse.linalg import spsolve

    assert np.allclose(projected, fem.project(values, region, solver=spsolve))


def test_topoints():
    mesh = fem.Rectangle(n=2).triangulate()
    region = fem.RegionTriangle(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    # single quadrature-point values
    data = fem.topoints(values, region)
    assert data.shape == (mesh.npoints, 3, 3)

    mesh = fem.Rectangle(n=3).convert(2, 1)
    region = fem.RegionQuadraticQuad(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    # trim values array to number of points-per-cell
    data = fem.topoints(values, region)
    assert data.shape == (mesh.npoints, 3, 3)

    data = fem.topoints(values, region, average=False)
    assert data.shape == (mesh.cells.size, 3, 3)

    data = fem.topoints(values, region, mean=True)
    assert data.shape == (mesh.npoints, 3, 3)

    data = fem.topoints(values, region, average=False, mean=True)
    assert data.shape == (mesh.cells.size, 3, 3)


def test_extrapolate():
    # rectangle (triangle)
    mesh = fem.Rectangle(n=2)
    mesh.update(points=np.vstack([mesh.points, [1000, 1000]]))
    region = fem.RegionQuad(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.tools.extrapolate(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    # rectangle (quadratic triangle)
    mesh = fem.Rectangle(n=2).add_midpoints_edges()
    region = fem.RegionQuadraticQuad(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.tools.extrapolate(values, region, average=True)

    # cube (tetra)
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.tools.extrapolate(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # cube (quadratic tetra)
    mesh = fem.Cube(n=2).add_midpoints_edges()
    region = fem.RegionQuadraticHexahedron(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.tools.extrapolate(values, region, average=True)


def test_resize():
    from scipy.sparse import csr_matrix

    # empty csr matrix with shape 100, 3
    matrix = csr_matrix((100, 3))

    with pytest.raises(ValueError):
        fem.tools._newton.spresize(matrix, shape=(50, 3))


if __name__ == "__main__":
    test_hello_world()

    for test in [test_solve, test_solve_mixed]:
        with tempfile.TemporaryDirectory() as tmp:
            test(tmp_path=pathlib.Path(tmp))

    test_newton_update()
    test_newton_simple()
    test_newton()
    test_newton_mixed()
    test_newton_plane()
    test_newton_linearelastic()
    test_newton_body()
    test_project()
    test_project_solver()
    test_topoints()
    test_extrapolate()
    test_resize()
