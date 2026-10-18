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
        fem.solve.SuperLUSolver(options=dict(Equil=False)),
    ]

    for solver in solvers:
        assert np.allclose(solver(K, b), x)
        assert repr(solver).startswith("SuperLUSolver(permc_spec=")

    assert "options={'Equil': False}" in repr(solvers[-1])
    assert isinstance(solvers[0].factorize(K), sp.linalg.SuperLU)


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


if __name__ == "__main__":
    test_solve()
    test_superlu()
    test_superlu_options()
    test_superlu_nonsymmetric()
    test_superlu_singular()
    test_superlu_job()
