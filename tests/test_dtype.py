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

    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

    umat = fem.LinearElastic(E=1, nu=0.3)
    solid = fem.SolidBody(umat, field)
    step = fem.Step([solid], boundaries=boundaries)
    job = fem.Job([step]).evaluate(tol=tol)

    assert np.allclose([norm[-1] for norm in job.fnorms], 0, atol=tol)
    assert displacement.values.dtype == dtype


def test_dtype_newton_tol():
    # the default tolerance depends on the data types of the linear equation system
    class Tolerance(fem.Plugin):
        def before_newton(self, context, state):
            self.tol_before_newton = state.tol

        def before_linear_solve(self, context, state):
            self.tol = state.tol

    for dtype in [np.float32, np.float64]:
        region = fem.RegionHexahedron(fem.Cube(n=3)).astype(dtype, copy=False)
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
        assert res.fun.dtype == dtype
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
    # a job without a given tolerance converges in single-precision
    region = fem.RegionHexahedron(fem.Cube(n=3)).astype(np.float32, copy=False)
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


if __name__ == "__main__":
    test_dtype()
    test_dtype_newton_tol()
    test_dtype_newton_tol_job()
    test_dtype_default_tol()
    test_dtype_axi()
    test_dtype_planestrain()
