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
    for dtype in [np.float32, np.float64]:
        mesh = fem.Cube(n=3)
        mesh.add_points([[2.0, 0.5, 0.5], [2.0, 0.5, 0.5]])
        region = fem.RegionHexahedron(mesh).astype(dtype)
        displacement = fem.Field(region, dim=3, dtype=dtype)
        field = fem.FieldContainer([displacement])

        # PointLoad
        load = fem.PointLoad(field, [1, 2], values=[[3], [5]])
        assert load.assemble.vector().dtype == dtype
        assert load.assemble.matrix().dtype == dtype

        # MultiPointConstraint without rotation point
        mpc = fem.MultiPointConstraint(
            field, points=np.arange(mesh.npoints - 2)[mesh.x[:-2] == 1], centerpoint=-2
        )
        assert mpc.assemble.vector().dtype == dtype
        assert mpc.assemble.matrix().dtype == dtype

        # MultiPointConstraint with rotation point
        mpc_rot = fem.MultiPointConstraint(
            field,
            points=np.arange(mesh.npoints - 2)[mesh.x[:-2] == 1],
            centerpoint=-2,
            rotationpoint=-1,
        )
        assert mpc_rot.assemble.vector().dtype == dtype
        assert mpc_rot.assemble.matrix().dtype == dtype

        # SolidBody with NearlyIncompressible
        field_mixed = fem.FieldsMixed(region, n=3, dtype=dtype)
        umat_mixed = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
        solid_mixed = fem.SolidBody(umat_mixed, field_mixed)
        assert solid_mixed.assemble.vector().dtype == dtype
        assert solid_mixed.assemble.matrix().dtype == dtype

        # SolidBody with Hyperelastic (tensortrax, if installed)
        try:
            import tensortrax  # noqa: F401

            umat_hyper = fem.Hyperelastic(fem.mooney_rivlin, C10=1.0, C01=0.2)
            solid_hyper = fem.SolidBody(umat_hyper, field)
            assert solid_hyper.assemble.vector().dtype == dtype
            assert solid_hyper.assemble.matrix().dtype == dtype
        except ImportError:
            pass


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
