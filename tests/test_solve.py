# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import MatrixRankWarning, spsolve

import felupe as fem


def test_solve():
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3)
    v = fem.FieldContainer([u])

    W = fem.constitution.NeoHooke(1, 3)

    F = v.extract()
    P = W.gradient(F)[:-1]
    A = W.hessian(F)

    b = fem.dof.symmetry(u)
    dof0, dof1 = fem.dof.partition(v, b)

    ext0 = fem.dof.apply(v, b, dof0)

    L = fem.IntegralForm(P, v, r.dV)
    a = fem.IntegralForm(A, v, r.dV, v)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    system = fem.solve.partition(v, A, dof1, dof0)

    du = fem.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fem.solve.solve(*system, ext0)
    assert np.allclose(du, 0)

    system = fem.solve.partition(v, A, dof1, dof0, b)

    du = fem.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fem.solve.solve(*system, ext0)
    assert np.allclose(du, 0)


def stiffness():
    "Return the partitioned stiffness matrix of a deformed cube and a right-hand side."

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), field)
    step = fem.Step([solid], ramp={boundaries["move"]: [0.2]}, boundaries=boundaries)
    fem.Job([step]).evaluate()

    dof0, dof1 = fem.dof.partition(field, boundaries)
    K = solid.assemble.matrix().tocsr()[dof1][:, dof1]
    b = np.random.default_rng(seed=56).uniform(-1, 1, K.shape[0])

    return K, b


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")  # np.matrix
def test_superlu():
    K, b = stiffness()
    solver = fem.solve.SuperLUSolver()

    x = solver(K, b)
    assert x.shape == b.shape
    assert np.allclose(K @ x, b)
    assert np.allclose(x, spsolve(K.tocsc(), b))

    # a single right-hand side with the shape (n, 1) returns a 1d-array, like spsolve
    for rhs in [b.reshape(-1, 1), np.asmatrix(b).T, sp.csr_array(b.reshape(-1, 1))]:
        y = solver(K, rhs)
        assert y.shape == b.shape
        assert np.allclose(y, x)

    # multiple right-hand sides
    B = np.vstack([b, 2 * b, -b]).T
    X = solver(K, B)
    assert X.shape == B.shape
    assert np.allclose(X, np.vstack([x, 2 * x, -x]).T)

    # input formats and a dense matrix
    for A in [K.tocsc(), K.tocoo(), sp.csr_matrix(K), K.toarray()]:
        assert np.allclose(solver(A, b), x)

    # single-precision
    x32 = solver(K.astype(np.float32), b.astype(np.float32))
    assert x32.dtype == np.float32
    assert np.allclose(x32, x, rtol=1e-3, atol=1e-4)


def test_superlu_options():
    K, b = stiffness()
    x = spsolve(K.tocsc(), b)

    solvers = [
        fem.solve.SuperLUSolver(),
        fem.solve.SuperLUSolver(diag_pivot_thresh=1.0),
        fem.solve.SuperLUSolver(diag_pivot_thresh=0.0),
        fem.solve.SuperLUSolver(permc_spec="COLAMD", symmetric_mode=False),
        fem.solve.SuperLUSolver(relax=None, panel_size=None),
        fem.solve.SuperLUSolver(refine=False, tol=1e-12, fallback_pivot_thresh=None),
        fem.solve.SuperLUSolver(diag_perturbation=0),
        fem.solve.SuperLUSolver(options=dict(Equil=False)),
    ]

    for solver in solvers:
        assert np.allclose(solver(K, b), x)
        assert repr(solver).startswith("SuperLUSolver(permc_spec=")
        assert repr(solver).endswith(
            f"fallback_pivot_thresh={solver.fallback_pivot_thresh!r})"
        )

    assert "options={'Equil': False}" in repr(solvers[-1])
    assert "diag_perturbation=None" in repr(solvers[0])
    assert "diag_perturbation=0" in repr(solvers[-2])
    assert isinstance(solvers[0].factorize(K), sp.linalg.SuperLU)
    assert isinstance(solvers[0].factorize(K, diag_pivot_thresh=1.0), sp.linalg.SuperLU)


def test_superlu_nonsymmetric():
    K, b = stiffness()

    # non-symmetric values with a symmetric structure, e.g. of a frictional contact
    A = (K + 0.2 * sp.triu(K, k=1) - 0.1 * sp.tril(K, k=-1)).tocsr()
    assert abs(A - A.T).max() > 0.1 * abs(A).max()

    x = fem.solve.SuperLUSolver()(A, b)
    assert np.allclose(A @ x, b)
    assert np.allclose(x, spsolve(A.tocsc(), b))


def test_superlu_singular():
    # an exactly singular matrix (zero row and column)
    A = sp.diags_array([1.0, 0.0, 2.0], format="csc")
    b = np.ones(3)

    with pytest.warns(MatrixRankWarning):
        x = fem.solve.SuperLUSolver()(A, b)

    assert x.shape == b.shape
    assert np.all(np.isnan(x))

    # zero diagonal entries are perturbed, but the refinement does not converge (the
    # equations of the last two rows are inconsistent)
    A = sp.csc_array(np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    b = np.array([1.0, 1.0, 0.0])
    solver = CountingSolver()

    assert (solver.perturb(A) != A).nnz == 2

    with pytest.warns(MatrixRankWarning):
        x = solver(A, b)

    # the factorization of the fallback (not recorded) fails, i.e. exactly singular
    assert np.all(np.isnan(x))
    assert solver.thresholds == [0.0]

    # other errors of SuperLU are raised
    class FailingSolver(fem.solve.SuperLUSolver):
        def factorize(self, A, diag_pivot_thresh=None):
            raise RuntimeError("Not enough memory to perform factorization.")

    with pytest.raises(RuntimeError, match="memory"):
        FailingSolver()(A, b)


def test_superlu_job():
    "The default solver and spsolve give the same results."

    results = []

    for solver in [None, spsolve]:
        mesh = fem.Rectangle(n=6)
        region = fem.RegionQuad(mesh)
        field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        boundaries = fem.dof.uniaxial(field, clamped=True)
        solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), field)
        ramp = {boundaries["move"]: fem.math.linsteps([0, 0.5], num=3)}
        step = fem.Step([solid], ramp=ramp, boundaries=boundaries)
        kwargs = {} if solver is None else dict(solver=solver)
        fem.Job([step]).evaluate(**kwargs)
        results.append(field[0].values.copy())

    assert np.allclose(*results)


class CountingSolver(fem.solve.SuperLUSolver):
    "A solver which records the pivot thresholds of its factorizations."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = []

    def factorize(self, A, diag_pivot_thresh=None):
        lu = super().factorize(A, diag_pivot_thresh=diag_pivot_thresh)
        thresh = (
            self.diag_pivot_thresh if diag_pivot_thresh is None else diag_pivot_thresh
        )
        self.thresholds.append(thresh)
        return lu


def unstable():
    "Return a matrix whose factorization without pivoting is not stable."

    A = sp.csc_array(
        np.array(
            [
                [1e-16, 1.0, 1.0],
                [1.0, 1.0 + 1e-3, 2.0],
                [1.0, 2.0, 1.0 + 2e-3],
            ]
        )
    )
    b = np.array([1.0, 3.0, 7.0])

    return A, b


def test_superlu_fallback():
    A, b = unstable()
    x = np.linalg.solve(A.toarray(), b)

    # without pivoting (and without the fallback), the solution is wrong
    solver = CountingSolver(
        permc_spec="NATURAL", refine=False, fallback_pivot_thresh=None
    )
    assert solver.backward_error(A, solver(A, b), b) > 1e-3
    assert solver.thresholds == [0.0]

    # the backward error triggers a second factorization with pivoting
    for refine in [False, True]:
        solver = CountingSolver(permc_spec="NATURAL", refine=refine)
        y = solver(A, b)
        assert solver.thresholds == [0.0, 0.1]
        assert np.allclose(y, x)
        assert solver.backward_error(A, y, b) < 1e-15

    # a custom tolerance of the backward error
    solver = CountingSolver(permc_spec="NATURAL", tol=1.0)
    solver(A, b)
    assert solver.thresholds == [0.0]

    # multiple right-hand sides
    solver = CountingSolver(permc_spec="NATURAL")
    Y = solver(A, np.vstack([b, -2 * b]).T)
    assert solver.thresholds == [0.0, 0.1]
    assert np.allclose(Y, np.vstack([x, -2 * x]).T)


def test_superlu_backward_error():
    K, b = stiffness()
    solver = fem.solve.SuperLUSolver()
    x = solver(K, b)

    # a backward-stable solution and a perturbed solution
    assert solver.backward_error(K, x, b) < 1e-14
    assert solver.backward_error(K, x, b, r=b - K @ x) < 1e-14
    assert solver.backward_error(K, 1.01 * x, b) > 1e-6

    # multiple right-hand sides: the largest backward error
    X = np.vstack([x, 1.01 * x]).T
    B = np.vstack([b, b]).T
    assert np.isclose(
        solver.backward_error(K, X, B), solver.backward_error(K, 1.01 * x, b)
    )

    # a NaN-solution has a NaN backward error
    assert np.isnan(solver.backward_error(K, np.full_like(x, np.nan), b))

    # the zero solution of a right-hand side of zeros
    z = np.zeros_like(b)
    assert solver.backward_error(K, z, z) == 0
    assert solver.backward_error(sp.csc_array(K.shape), z, z) == 0


def test_superlu_zeros():
    "A right-hand side of zeros does not trigger the fallback, e.g. a first substep."

    K, b = stiffness()
    solver = CountingSolver()
    x = solver(K, np.zeros_like(b))

    assert np.all(x == 0)
    assert solver.thresholds == [0.0]


def test_superlu_tolerance():
    K, b = stiffness()

    tol = fem.solve.SuperLUSolver()._tolerance
    assert np.isclose(tol(K, b), np.sqrt(np.finfo(np.float64).eps))
    assert np.isclose(tol(K, b.astype(np.float32)), np.sqrt(np.finfo(np.float32).eps))
    assert np.isclose(tol(K.astype(np.float32), b), np.sqrt(np.finfo(np.float32).eps))
    assert np.isclose(tol(K, np.ones(len(b), dtype=int)), np.sqrt(np.finfo(float).eps))
    assert fem.solve.SuperLUSolver(tol=1e-3)._tolerance(K, b) == 1e-3


def contact_model(mixed):
    "A block with mixed fields or displacements, pressed and dragged by a plane."

    mesh = fem.Cube(n=4)
    mesh.add_points([0.5, 0.5, 1.0])
    region = fem.RegionHexahedron(mesh)

    if mixed:
        field = fem.FieldsMixed(region, n=3)
        umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1.0, bulk=5000.0))
    else:
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        umat = fem.NeoHooke(mu=1.0, bulk=5.0)

    solid = fem.SolidBody(umat, field)
    top = np.arange(mesh.npoints - 1)[np.isclose(mesh.points[:-1, 2], 1.0)]
    contact = fem.ContactRigidPlane(
        field,
        points=top,
        centerpoint=-1,
        normal=[0, 0, -1],
        friction=0.3,
        multiplier=10.0,
        items=[solid],
    )

    center = np.zeros(mesh.npoints, dtype=bool)
    center[-1] = True
    boundaries = {
        "bottom": fem.Boundary(field[0], fz=0),
        "move-z": fem.Boundary(field[0], mask=center, skip=(1, 1, 0)),
        "move-xy": fem.Boundary(field[0], mask=center, skip=(0, 0, 1)),
    }
    ramp = {
        boundaries["move-z"]: fem.math.linsteps([0, -0.1, -0.1], num=[2, 2]),
        boundaries["move-xy"]: fem.math.linsteps([0, 0, 0.1], num=[2, 2]),
    }
    step = fem.Step([solid, contact], ramp=ramp, boundaries=boundaries)

    return field, contact, step


@pytest.mark.parametrize("mixed", [True, False])
def test_superlu_contact(mixed):
    """A saddle point problem with zero diagonal entries (three-field variation) and a
    frictional contact are solved without pivoting, i.e. without the fallback."""

    results = []
    solvers = [CountingSolver(), spsolve]

    for solver in solvers:
        field, contact, step = contact_model(mixed)
        fem.Job([step]).evaluate(solver=solver, verbose=0)
        assert np.any(contact.results.active)
        results.append(field[0].values.copy())

    assert len(solvers[0].thresholds) > 0
    assert set(solvers[0].thresholds) == {0.0}
    assert np.allclose(*results)


def test_superlu_third_medium():
    "A third medium contact (mixed field formulation) is solved without the fallback."

    results = []
    solvers = [CountingSolver(), spsolve]

    for solver in solvers:
        region = fem.RegionQuad(fem.Rectangle(n=4))
        field = fem.FieldContainer(
            [fem.FieldPlaneStrain(region, dim=2), fem.Field(region, dim=9)],
            take=[0, 1, 1],
        )
        umat = fem.ThirdMediumContactMixed(
            fem.NeoHooke(mu=1, bulk=20), gamma=1e-5, alpha_r=1e-4, p_r=1e-2
        )
        solid = fem.SolidBody(umat, field, grad=[True, False, True])
        boundaries = fem.dof.uniaxial(field, clamped=True)
        ramp = {boundaries["move"]: fem.math.linsteps([0, -0.5], num=4)}
        step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
        fem.Job([step]).evaluate(solver=solver, verbose=0)
        results.append(field[0].values.copy())

    assert len(solvers[0].thresholds) > 0
    assert set(solvers[0].thresholds) == {0.0}
    assert np.allclose(*results)


def saddle_point(n=3):
    """Return the partitioned matrix and a right-hand side of a saddle point problem,
    i.e. an incompressible mixed u/p-formulation by a mixed weak form on quadratic
    hexahedrons with a constant pressure per cell (with a zero block of the pressure)."""

    mesh = fem.Cube(n=n).add_midpoints_edges()
    region = fem.RegionQuadraticHexahedron(mesh)
    field = fem.FieldsMixed(region, n=2)
    boundaries = fem.dof.uniaxial(field, clamped=True, move=0.1)
    dof0, dof1 = fem.dof.partition(field, boundaries)
    ext0 = fem.dof.apply(field, boundaries, dof0)

    @fem.Form(v=field, u=field)
    def bilinearform():
        return [
            lambda v, u: 2 * fem.math.ddot(fem.math.sym(v.grad), fem.math.sym(u.grad)),
            lambda v, q: -fem.math.trace(v.grad) * q,
            lambda q, p: 0 * q * p,
        ]

    K = bilinearform.assemble(v=field, u=field).tocsr()

    return K[dof1][:, dof1], -(K[dof1][:, dof0] @ ext0)


class RecordingSolver(CountingSolver):
    "A solver which records the pivot thresholds and the factors of its matrices."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factors = []

    def factorize(self, A, diag_pivot_thresh=None):
        lu = super().factorize(A, diag_pivot_thresh=diag_pivot_thresh)
        self.factors.append(lu)
        return lu


def test_superlu_perturbation():
    "The zero diagonal entries of a saddle point problem are perturbed."

    A, b = saddle_point()
    zeros = A.diagonal() == 0
    assert np.any(zeros)

    solver = fem.solve.SuperLUSolver()
    P = solver.perturb(A)
    perturbation = (P - A).tocsc()

    # only the zero diagonal entries are perturbed (with a negative sign)
    assert abs(perturbation - sp.diags(perturbation.diagonal())).max() == 0
    assert np.all(perturbation.diagonal()[zeros] < 0)
    assert np.all(perturbation.diagonal()[~zeros] == 0)

    # by the square root of the machine epsilon times the estimated pivots
    assert np.all(-P.diagonal()[zeros] < 1e-7 * abs(A).max())

    # the perturbation is invariant to the scaling of the unknowns, e.g. of the pressure
    scale = sp.diags(np.where(zeros, 1e3, 1.0))
    Q = solver.perturb(scale @ A @ scale)
    assert np.allclose(Q.diagonal()[zeros], 1e6 * P.diagonal()[zeros])

    # the matrix is factorized without row exchanges and without the fallback
    solver = RecordingSolver()
    x = solver(A, b)
    assert solver.thresholds == [0.0]
    assert np.all(solver.factors[0].perm_r == solver.factors[0].perm_c)
    assert np.allclose(x, spsolve(A.tocsc(), b))
    assert solver.backward_error(A, x, b) < 1e-14

    # multiple right-hand sides and single-precision
    X = solver(A, np.vstack([b, -2 * b]).T)
    assert np.allclose(X, np.vstack([x, -2 * x]).T)

    x32 = solver(A.astype(np.float32), b.astype(np.float32))
    assert x32.dtype == np.float32
    assert np.allclose(x32, x, rtol=1e-3, atol=1e-4 * np.abs(x).max())

    # without the perturbation, the zero diagonal entries lead to row exchanges
    for kwargs in [dict(diag_perturbation=0), dict(refine=False)]:
        solver = RecordingSolver(**kwargs)
        y = solver(A, b)
        assert solver.thresholds == [0.0]
        assert np.any(solver.factors[0].perm_r != solver.factors[0].perm_c)
        assert np.allclose(y, x)


def test_superlu_perturbation_unchanged():
    "Matrices without perturbed zero diagonal entries are not changed."

    K, b = stiffness()
    K = K.tocsc()
    assert fem.solve.SuperLUSolver().perturb(K) is K

    # an empty row and column remains (singular)
    A = sp.diags_array([1.0, 0.0, 2.0], format="csc")
    assert fem.solve.SuperLUSolver().perturb(A) is A

    # zero diagonal entries without non-zero diagonal entries of their neighbours
    A = sp.csc_array(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert fem.solve.SuperLUSolver().perturb(A) is A
    assert np.allclose(fem.solve.SuperLUSolver()(A, np.array([1.0, 2.0])), [2.0, 1.0])

    # a custom relative perturbation
    A, b = saddle_point()
    zeros = A.diagonal() == 0
    P = fem.solve.SuperLUSolver().perturb(A)
    Q = fem.solve.SuperLUSolver(diag_perturbation=1e-4).perturb(A)
    eps = np.sqrt(np.finfo(float).eps)
    assert np.allclose(Q.diagonal()[zeros], 1e-4 / eps * P.diagonal()[zeros])


def test_superlu_perturbation_fallback():
    "A (too large) perturbation without convergence of the refinement is rejected."

    A, b = saddle_point()
    x = spsolve(A.tocsc(), b)

    solver = RecordingSolver(diag_perturbation=1e3)
    y = solver(A, b)
    assert solver.thresholds == [0.0, 0.1]
    assert np.allclose(y, x)

    # without a fallback, the solution of the perturbed matrix is returned
    solver = RecordingSolver(diag_perturbation=1e3, fallback_pivot_thresh=None)
    z = solver(A, b)
    assert solver.thresholds == [0.0]
    assert solver.backward_error(A, z, b) > 1e-3


if __name__ == "__main__":
    test_solve()
    test_superlu()
    test_superlu_options()
    test_superlu_nonsymmetric()
    test_superlu_singular()
    test_superlu_job()
    test_superlu_fallback()
    test_superlu_backward_error()
    test_superlu_zeros()
    test_superlu_tolerance()
    test_superlu_contact(mixed=True)
    test_superlu_contact(mixed=False)
    test_superlu_third_medium()
    test_superlu_perturbation()
    test_superlu_perturbation_unchanged()
    test_superlu_perturbation_fallback()
