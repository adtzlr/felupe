# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

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


if __name__ == "__main__":
    test_solve()
