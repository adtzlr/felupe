# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import felupe as fem


def pre(dim):
    m = fem.Cube(n=3)
    r = fem.RegionHexahedron(m)
    u = fem.FieldContainer([fem.Field(r, dim=dim)])
    return r.reload(grad=True, hess=True), u


def pre_constant(dim):
    m = fem.Cube(n=3)
    r = fem.RegionConstantHexahedron(m)
    u = fem.FieldContainer([fem.Field(r, dim=dim)])
    return r, u


def test_basis():
    for parallel in [False, True]:
        r, u = pre(dim=3)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert not np.any(np.equal(b[0].basis.grad, None))

        r, u = pre(dim=1)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert not np.any(np.equal(b[0].basis.grad, None))

        r, u = pre(dim=1)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert not np.any(np.equal(b[0].basis.hess, None))

        r, u = pre_constant(dim=3)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert np.all(np.equal(b[0].basis.grad, None))
        assert np.all(np.equal(b[0].basis.hess, None))

        r, u = pre_constant(dim=1)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert np.all(np.equal(b[0].basis.grad, None))
        assert np.all(np.equal(b[0].basis.hess, None))


if __name__ == "__main__":
    test_basis()
