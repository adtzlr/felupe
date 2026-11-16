# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import felupe as fem


def test_dtype(dtype=np.float32, tol=1e-3):
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh).astype(dtype, copy=False)
    displacement = fem.Field(region, dim=3, dtype=dtype)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == dtype
    assert field.extract(grad=False)[0].dtype == dtype

    boundaries = fem.dof.uniaxial(field, clamped=True)

    umat = fem.LinearElastic(E=1, nu=0.3)
    solid = fem.SolidBody(umat, field)
    step = fem.Step([solid], boundaries=boundaries)
    job = fem.Job([step]).evaluate(tol=tol)

    assert np.allclose([norm[-1] for norm in job.fnorms], 0, atol=tol)
    assert displacement.values.dtype == dtype


def test_dtype_newton_tol():
    # the default tolerance depends on the data types of the linear equation system
    # and of the unknowns
    class Tolerance(fem.Plugin):
        def before_newton(self, context, state):
            self.tol_before_newton = state.tol

        def before_linear_solve(self, context, state):
            self.tol = state.tol

    # data types of the region and of the field
    dtypes = [
        (np.float32, np.float32),
        (np.float64, np.float32),
        (np.float64, np.float64),
    ]

    for region_dtype, dtype in dtypes:
        region = fem.RegionHexahedron(fem.Cube(n=3)).astype(region_dtype, copy=False)
        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=dtype)])
        boundaries, loadcase = fem.dof.uniaxial(
            field, move=0.2, clamped=True, return_loadcase=True
        )
        solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)

        tolerance = Tolerance()
        res = fem.newtonraphson(
            items=[solid], **loadcase, plugins=[tolerance], verbose=0
        )

        tol = np.sqrt(np.finfo(dtype).eps)

        assert res.success
        assert res.fun.dtype == np.result_type(region_dtype, dtype)
        assert res.x[0].values.dtype == dtype
        assert res.fnorms[-1] < tol
        assert tolerance.tol_before_newton is None
        assert tolerance.tol == tol

        # a given tolerance is not modified
        tolerance = Tolerance()
        res = fem.newtonraphson(
            items=[solid], **loadcase, plugins=[tolerance], tol=1e-2, verbose=0
        )

        assert res.success
        assert tolerance.tol_before_newton == 1e-2
        assert tolerance.tol == 1e-2


def test_dtype_newton_tol_job():
    # a job without a given tolerance converges with fields in single-precision on a
    # region in single- or double-precision
    for region_dtype in [np.float32, np.float64]:
        region = fem.RegionHexahedron(fem.Cube(n=3)).astype(region_dtype, copy=False)
        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=np.float32)])
        boundaries = fem.dof.uniaxial(field, clamped=True)

        solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)
        step = fem.Step(
            [solid], ramp={boundaries["move"]: [0.1, 0.2]}, boundaries=boundaries
        )
        job = fem.Job([step]).evaluate(verbose=0)

        tol = np.sqrt(np.finfo(np.float32).eps)

        assert np.all([fnorms[-1] < tol for fnorms in job.fnorms])
        assert field[0].values.dtype == np.float32


def test_dtype_default_tol():
    from scipy.sparse import csr_matrix

    default_tol = fem.tools._newton.default_tol

    def tol(dtype):
        return float(np.sqrt(np.finfo(dtype).eps))

    A = np.eye(2)
    b = np.ones(2)

    # the common data type of the Jacobian and the objective function
    for dtype in [np.float16, np.float32, np.float64, np.longdouble]:
        assert default_tol(A.astype(dtype), b.astype(dtype)) == tol(dtype)

    assert default_tol(A, b) == np.sqrt(np.finfo(float).eps)
    assert default_tol(A.astype(np.float32), b) == tol(np.float64)
    assert default_tol(A, b.astype(np.float32)) == tol(np.float64)
    assert default_tol(A.astype(np.float16), b.astype(np.float32)) == tol(np.float32)

    # sparse matrices and complex data types
    assert default_tol(csr_matrix(A, dtype=np.float32), b.astype(np.float32)) == tol(
        np.float32
    )
    assert default_tol(A.astype(np.complex64), b.astype(np.complex64)) == tol(
        np.float32
    )

    # data types without a machine epsilon and objects without a data type
    assert default_tol(A.astype(int), b.astype(int)) == tol(np.float64)
    assert default_tol(A.astype(bool), b.astype(np.int8)) == tol(np.float64)
    assert default_tol(1.0, [1.0]) == tol(np.float64)
    assert isinstance(default_tol(A, b), float)

    # the lowest precision of the linear equation system and of the unknowns (arrays)
    assert default_tol(A, b, x=b) == tol(np.float64)
    assert default_tol(A, b, x=b.astype(np.float32)) == tol(np.float32)
    assert default_tol(A, b, x=b.astype(np.float16)) == tol(np.float16)
    assert default_tol(A.astype(np.float32), b.astype(np.float32), x=b) == tol(
        np.float32
    )
    assert default_tol(A, b, x=b.astype(np.complex64)) == tol(np.float32)

    # data types of the unknowns without a machine epsilon are ignored
    assert default_tol(A, b, x=b.astype(int)) == tol(np.float64)
    assert default_tol(A.astype(np.float32), b, x=b.astype(int)) == tol(np.float64)
    assert default_tol(A, b, x=None) == tol(np.float64)

    # the lowest precision of all fields of a field container
    region = fem.RegionHexahedron(fem.Cube(n=2))
    u = fem.Field(region, dim=3, dtype=np.float32)
    p = fem.Field(region, dtype=np.float64)

    assert default_tol(A, b, x=fem.FieldContainer([p])) == tol(np.float64)
    assert default_tol(A, b, x=fem.FieldContainer([u])) == tol(np.float32)
    assert default_tol(A, b, x=fem.FieldContainer([u, p])) == tol(np.float32)
    assert default_tol(A, b, x=fem.FieldContainer([p, u])) == tol(np.float32)


def test_dtype_axi():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh).astype(np.float32)
    displacement = fem.FieldAxisymmetric(region, dtype=np.float32)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == np.float32
    assert field.extract(grad=False)[0].dtype == np.float32


def test_dtype_planestrain():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh, hess=True).astype(np.float32)
    displacement = fem.FieldPlaneStrain(region, dtype=np.float32)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == np.float32
    assert field.extract(grad=False)[0].dtype == np.float32


def test_dtype_dual():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh).astype(np.float32)
    pressure = fem.FieldDual(region, dtype=np.float32)

    assert pressure.values.dtype == np.float32
    assert pressure.region.h.dtype == np.float32


def test_dtype_mixed():
    for region_type, mesh_fn, kwargs in [
        (fem.RegionHexahedron, lambda: fem.Cube(n=3), {}),
        (fem.RegionQuad, lambda: fem.Rectangle(n=3), {"axisymmetric": True}),
        (fem.RegionQuad, lambda: fem.Rectangle(n=3), {"planestrain": True}),
    ]:
        mesh = mesh_fn()
        region = region_type(mesh).astype(np.float32)
        field = fem.FieldsMixed(region, n=3, dtype=np.float32, **kwargs)

        assert all(f.values.dtype == np.float32 for f in field.fields)
        assert all(f.region.h.dtype == np.float32 for f in field.fields)


def test_dtype_item_assembly():
    def get_items(dtype):
        mesh = fem.Cube(n=3)
        mesh.add_points([[2.0, 0.5, 0.5], [2.0, 0.6, 0.6]])
        region = fem.RegionHexahedron(mesh).astype(dtype)
        displacement = fem.Field(region, dim=3, dtype=dtype)
        field = fem.FieldContainer([displacement])
        field[0].values[-2, 0] = -2.0

        load = fem.PointLoad(field, [1, 2], values=[[3], [5]])

        mpc = fem.MultiPointConstraint(
            field, points=np.arange(mesh.npoints - 2)[mesh.x[:-2] == 1], centerpoint=-2
        )

        mpc_rot = fem.MultiPointConstraint(
            field,
            points=np.arange(mesh.npoints - 2)[mesh.x[:-2] == 1],
            centerpoint=-2,
            rotationpoint=-1,
        )

        contact = fem.MultiPointContact(
            field,
            points=np.arange(mesh.npoints - 2)[mesh.x[:-2] == 1],
            centerpoint=-2,
        )

        cartesian = fem.CartesianConstraint(field, points=[0])
        cylindrical = fem.CylindricalConstraint(
            field,
            points=[np.arange(mesh.npoints - 2)[mesh.x[:-2] == 1][0]],
            center=(0.5, 0.5, 0.5),
            axis=(1.0, 0.0, 0.0),
        )

        field_mixed = fem.FieldsMixed(region, n=3, dtype=dtype)
        umat_mixed = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
        solid_mixed = fem.SolidBody(umat_mixed, field_mixed)

        @fem.Form(v=field)
        def linearform():
            return [lambda v: fem.math.ddot(v, v)]

        @fem.Form(v=field, u=field)
        def bilinearform():
            return [lambda v, u: fem.math.ddot(v, u)]

        form_linear = fem.FormItem(linearform=linearform)
        form_bilinear = fem.FormItem(bilinearform=bilinearform)

        boundary = fem.RegionHexahedronBoundary(mesh, mask=mesh.x == 1).astype(dtype)
        temperature = fem.FieldContainer([fem.Field(region, dim=1, dtype=dtype)])
        temperature_boundary = fem.FieldContainer(
            [fem.Field(boundary, dim=1, dtype=dtype)]
        )
        temperature[0].values[:] = 1.0
        temperature_boundary[0].values[:] = 1.0

        thermal = fem.thermal.SolidBodyThermal(
            temperature,
            mass_density=1.0,
            specific_heat_capacity=1.0,
            thermal_conductivity=1.0,
            time_step=1.0,
        )
        thermal.results.statevars = np.zeros_like(temperature[0].values)

        items = [
            load,
            mpc,
            mpc_rot,
            contact,
            cartesian,
            cylindrical,
            solid_mixed,
            form_linear,
            form_bilinear,
            fem.SolidBodyForce(field),
            thermal,
            fem.thermal.SolidBodyHeatFlux(temperature_boundary, heat_flux=1.0),
            fem.thermal.SolidBodySurfaceHeatTransfer(
                temperature_boundary, coefficient=1.0, temperature=0.0
            ),
            fem.thermal.SolidBodySurfaceRadiation(
                temperature_boundary, emissivity=0.5, temperature=0.0
            ),
        ]

        try:
            import tensortrax  # noqa: F401

            umat_hyper = fem.Hyperelastic(fem.mooney_rivlin, C10=1.0, C01=0.2)
            solid_hyper = fem.SolidBody(umat_hyper, field)
            items.append(solid_hyper)
        except ImportError:
            pass

        return items

    items32 = get_items(np.float32)
    items64 = get_items(np.float64)

    for item32, item64 in zip(items32, items64):
        r32 = item32.assemble.vector()
        K32 = item32.assemble.matrix()
        r64 = item64.assemble.vector()
        K64 = item64.assemble.matrix()

        assert r32.dtype == np.float32, type(item32).__name__
        assert K32.dtype == np.float32, type(item32).__name__
        assert r64.dtype == np.float64, type(item64).__name__
        assert K64.dtype == np.float64, type(item64).__name__

        assert np.allclose(K32.toarray(), K64.toarray(), rtol=1e-5, atol=1e-6)
        assert np.allclose(r32.toarray(), r64.toarray(), rtol=1e-5, atol=1e-8)


def test_dtype_boundary():
    "Items on a boundary region assemble in the precision of the field."

    for dtype in [np.float32, np.float64]:
        mesh = fem.Cube(n=3)
        region = fem.RegionHexahedronBoundary(mesh, mask=mesh.points[:, 0] == 1)
        region = region.astype(dtype)

        assert region.dA.dtype == dtype
        assert region.dV.dtype == dtype
        assert region.normals.dtype == dtype
        assert all(tangent.dtype == dtype for tangent in region.tangents)

        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=dtype)])
        pressure = fem.SolidBodyPressure(field, pressure=1.0)
        stress = fem.SolidBodyCauchyStress(field)

        for item in [pressure, stress]:
            assert item.assemble.vector().dtype == dtype
            assert item.assemble.matrix().dtype == dtype

        # a pressure of one on the face x=1 of a unit cube integrates to -1 in x
        vector = pressure.assemble.vector().toarray().reshape(-1, 3)

        assert np.allclose(vector.sum(axis=0), [-1.0, 0.0, 0.0])


def test_dtype_form():
    "The Form API integrates and assembles in the precision of the fields."

    def assemble(dtype):
        mesh = fem.Cube(n=3)
        region = fem.RegionHexahedron(mesh).astype(dtype)
        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=dtype)])

        np.random.seed(0)
        field[0].values = (np.random.rand(*field[0].values.shape) / 20).astype(dtype)

        f = np.array([1.0, 0.5, 0.0])

        @fem.Form(v=field, u=field)
        def bilinearform():
            return [lambda v, u: fem.math.ddot(v, u)]

        @fem.Form(v=field)
        def linearform():
            return [lambda v: np.einsum("i...,i...->...", f, v)]

        values = [
            *bilinearform.integrate(v=field, u=field),
            *linearform.integrate(v=field),
        ]
        item = fem.FormItem(bilinearform=bilinearform, linearform=linearform)

        return values, item.assemble.vector(), item.assemble.matrix()

    values32, vector32, matrix32 = assemble(np.float32)
    values64, vector64, matrix64 = assemble(np.float64)

    assert all(value.dtype == np.float32 for value in values32)
    assert all(value.dtype == np.float64 for value in values64)

    assert vector32.dtype == np.float32
    assert matrix32.dtype == np.float32
    assert vector64.dtype == np.float64
    assert matrix64.dtype == np.float64

    assert np.allclose(matrix32.toarray(), matrix64.toarray(), rtol=1e-5, atol=1e-6)
    assert np.allclose(vector32.toarray(), vector64.toarray(), rtol=1e-5, atol=1e-8)


def test_dtype_contact():
    "The contact items assemble in the precision of the field."

    def assemble(dtype):
        mesh = fem.Cube(n=3)
        region = fem.RegionHexahedron(mesh).astype(dtype)
        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=dtype)])

        np.random.seed(0)
        field[0].values = (np.random.rand(*field[0].values.shape) / 20).astype(dtype)

        points = np.arange(mesh.npoints)[mesh.points[:, 0] == 1]
        center = np.argmin(np.linalg.norm(mesh.points - [0.5, 0.5, 0.5], axis=1))

        items = [
            fem.ContactRigidPlane(field, points, center, normal=[-1, 0, 0]),
            fem.ContactRigidCylinder(field, points, center, radius=1.0, axis=[0, 0, 1]),
        ]

        return [(item.assemble.vector(), item.assemble.matrix()) for item in items]

    for (r32, K32), (r64, K64) in zip(assemble(np.float32), assemble(np.float64)):
        assert r32.dtype == np.float32
        assert K32.dtype == np.float32
        assert r64.dtype == np.float64
        assert K64.dtype == np.float64

        assert np.allclose(r32.toarray(), r64.toarray(), rtol=1e-4, atol=1e-6)
        assert np.allclose(K32.toarray(), K64.toarray(), rtol=1e-4, atol=1e-3)


def test_dtype_truss():
    "The truss body assembles in the precision of the field."

    def assemble(dtype):
        mesh = fem.Mesh(
            points=[[0, 0], [1, 1], [2.0, 0]], cells=[[0, 1], [1, 2]], cell_type="line"
        )
        region = fem.RegionTruss(mesh).astype(dtype)
        field = fem.FieldContainer([fem.Field(region, dim=2, dtype=dtype)])

        np.random.seed(0)
        field[0].values = (np.random.rand(*field[0].values.shape) / 20).astype(dtype)

        truss = fem.TrussBody(fem.LinearElastic1D(E=[1.0, 1.0]), field, area=[1.0, 1.0])
        truss.evaluate.gradient(field)  # initial kinematics

        vector = truss.assemble.vector()
        matrix = truss.assemble.matrix()

        assert truss.results.stress.dtype == dtype
        assert truss.results.elasticity.dtype == dtype

        return vector, matrix

    r32, K32 = assemble(np.float32)
    r64, K64 = assemble(np.float64)

    assert r32.dtype == np.float32
    assert K32.dtype == np.float32
    assert r64.dtype == np.float64
    assert K64.dtype == np.float64

    assert np.allclose(r32.toarray(), r64.toarray(), rtol=1e-5, atol=1e-8)
    assert np.allclose(K32.toarray(), K64.toarray(), rtol=1e-5, atol=1e-6)


def test_dtype_ogden_roxburgh():
    "The pseudo-elastic material assembles in the precision of the field."

    def assemble(dtype):
        mesh = fem.Cube(n=3)
        region = fem.RegionHexahedron(mesh).astype(dtype)
        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=dtype)])

        np.random.seed(0)
        field[0].values = (np.random.rand(*field[0].values.shape) / 20).astype(dtype)

        umat = fem.OgdenRoxburgh(fem.NeoHooke(mu=1), r=3, m=1, beta=0)
        solid = fem.SolidBody(umat, field)

        return solid.assemble.vector(), solid.assemble.matrix()

    r32, K32 = assemble(np.float32)
    r64, K64 = assemble(np.float64)

    assert r32.dtype == np.float32
    assert K32.dtype == np.float32
    assert r64.dtype == np.float64
    assert K64.dtype == np.float64

    assert np.allclose(r32.toarray(), r64.toarray(), rtol=1e-5, atol=1e-8)
    assert np.allclose(K32.toarray(), K64.toarray(), rtol=1e-5, atol=1e-6)


def test_dtype_free_vibration():
    "The free vibration solves in the precision of the fields."

    for dtype in [np.float32, np.float64]:
        mesh = fem.Cube(n=3)
        region = fem.RegionHexahedron(mesh).astype(dtype)
        field = fem.FieldContainer([fem.Field(region, dim=3, dtype=dtype)])
        solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field, density=1.5e-9)

        matrices = {}

        def solver(A, M, **kwargs):
            matrices["A"] = A.dtype
            matrices["M"] = M.dtype

            return np.array([1.0]), np.ones((A.shape[0], 1))

        fem.FreeVibration([solid]).evaluate(solver=solver)

        assert matrices["A"] == dtype
        assert matrices["M"] == dtype
        assert solid.assemble.mass().dtype == dtype


if __name__ == "__main__":
    test_dtype()
    test_dtype_newton_tol()
    test_dtype_newton_tol_job()
    test_dtype_default_tol()
    test_dtype_axi()
    test_dtype_planestrain()
    test_dtype_dual()
    test_dtype_mixed()
    test_dtype_item_assembly()
    test_dtype_boundary()
    test_dtype_form()
    test_dtype_contact()
    test_dtype_truss()
    test_dtype_ogden_roxburgh()
    test_dtype_free_vibration()
