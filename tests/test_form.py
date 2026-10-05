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


def pre():
    m = fem.Cube(n=3)
    r = fem.RegionHexahedron(m)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)

    v = fem.FieldContainer([u])
    q = fem.FieldContainer([p])

    W = fem.constitution.NeoHooke(1, 3)

    F = v.extract(grad=True, add_identity=True)
    P = W.gradient(F)[:-1]
    A = W.hessian(F)

    return r, v, q, P, A


def pre_broadcast():
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)

    v = fem.FieldContainer([u])
    q = fem.FieldContainer([p])

    W = fem.constitution.LinearElastic(E=1.0, nu=0.3)

    F = v.extract(grad=True, add_identity=True)
    P = W.gradient(F)[:-1]
    A = W.hessian()

    P = [P[0][:, :, 0, 0].reshape(3, 3, 1, 1)]

    return r, v, q, P, A


def pre_axi():
    m = fem.Rectangle(n=3)
    r = fem.RegionQuad(m)

    u = fem.FieldAxisymmetric(r, dim=2)
    v = fem.FieldContainer([u])

    W = fem.constitution.NeoHooke(1, 3)

    F = v.extract(grad=True, add_identity=True)
    P = W.gradient(F)[:-1]
    A = W.hessian(F)

    return r, v, P, A


def pre_mixed():
    m = fem.mesh.Cube(n=3)
    e = fem.element.Hexahedron()
    q = fem.quadrature.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)
    J = fem.Field(r, values=1.0)

    f = fem.FieldContainer((u, p, J))

    nh = fem.NeoHooke(1, 3)
    W = fem.ThreeFieldVariation(nh)

    return r, f, W.gradient(f.extract())[:-1], W.hessian(f.extract())


def pre_axi_mixed():
    m = fem.mesh.Rectangle(n=3)
    e = fem.element.Quad()
    q = fem.quadrature.GaussLegendre(1, 2)
    r = fem.Region(m, e, q)

    u = fem.FieldAxisymmetric(r, dim=2)
    p = fem.Field(r)
    J = fem.Field(r, values=1.0)

    f = fem.FieldContainer((u, p, J))

    nh = fem.NeoHooke(1, 3)
    W = fem.ThreeFieldVariation(nh)

    return r, f, W.gradient(f.extract())[:-1], W.hessian(f.extract())


def test_axi():
    r, u, P, A = pre_axi()

    for parallel in [False, True]:
        L = fem.IntegralForm(P, u, r.dV)
        x = L.integrate(parallel=parallel)

        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        a = fem.IntegralForm(A, u, r.dV, u, grad_v=[True], grad_u=[True])
        y = a.integrate(parallel=parallel)

        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fem.IntegralForm(
            [A[0][:, 0, :, :]], u, r.dV, u, grad_v=[False], grad_u=[True]
        )
        y = a.integrate(parallel=parallel)

        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fem.IntegralForm(
            [A[0][:, :, :, 0]], u, r.dV, u, grad_v=[True], grad_u=[False]
        )
        y = a.integrate(parallel=parallel)

        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fem.IntegralForm(
            [A[0][:, 0, :, 0]], u, r.dV, u, grad_v=[False], grad_u=[False]
        )
        y = a.integrate(parallel=parallel)

        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)


def test_linearform():
    r, u, p, P, A = pre()

    for parallel in [False, True]:
        L = fem.IntegralForm(P, u, r.dV, grad_v=[True])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        L = fem.IntegralForm(p.extract(grad=False), p, r.dV, grad_v=[False])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)


def test_linearform_broadcast():
    r, u, p, P, A = pre_broadcast()

    for parallel in [False, True]:
        L = fem.IntegralForm(P, u, r.dV, grad_v=[True])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        L = fem.IntegralForm(p.extract(grad=False), p, r.dV, grad_v=[False])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)


def test_bilinearform():
    r, u, p, P, A = pre()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, u, r.dV, u)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fem.IntegralForm(P, u, r.dV, p, [True], [False])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)

    nc = r.mesh.ncells
    nq = r.quadrature.npoints

    fun = np.ones((3, 3, nq, nc))
    form = fem.IntegralForm([fun], u, r.dV, u, grad_v=[False], grad_u=[False])
    form.assemble()

    fun = np.ones((3, 1, 3, nq, nc))
    form = fem.IntegralForm([fun], u, r.dV, u, grad_v=[False], grad_u=[False])
    with pytest.raises(ValueError):
        form.assemble()

    fun = np.ones((3, 3, 1, nq, nc))
    form = fem.IntegralForm([fun], u, r.dV, u, grad_v=[False], grad_u=[False])
    with pytest.raises(ValueError):
        form.assemble()

    fun = np.ones((3, 1, 3, 1, nq, nc))
    form = fem.IntegralForm([fun], u, r.dV, u, grad_v=[False], grad_u=[False])
    form.assemble()

    fun = np.ones((3, 1, 1, 3, 1, nq, nc))
    form = fem.IntegralForm([fun], u, r.dV, u, grad_v=[False], grad_u=[False])
    with pytest.raises(ValueError):
        form.assemble()


def test_bilinearform_broadcast():
    r, u, p, P, A = pre_broadcast()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, u, r.dV, u, [True], [True])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fem.IntegralForm(P, u, r.dV, p, [True], [False])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)

        q = p.extract(grad=False)
        f = fem.math.dya(q[0], q[0], mode=1)
        a = fem.IntegralForm(f, p, r.dV, p, [False], [False])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.npoints, r.mesh.npoints)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.npoints, r.mesh.npoints)


def test_bilinearform_grad_grad_chunks():
    "The chunked evaluation of the gradient-gradient form matches a single einsum."

    import felupe.assembly._cartesian as cartesian

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    field[0].values[:] = 0.1 * mesh.points**2

    F = field.extract()
    hessians = [
        fem.NeoHooke(mu=1.0, bulk=2.0).hessian(F)[0],  # one tensor per point
        fem.LinearElastic(E=1.0, nu=0.3).hessian()[0],  # broadcasted tensor
    ]

    chunksize_bytes = cartesian.CHUNKSIZE_BYTES

    try:
        for A in hessians:
            expected = np.einsum(
                "aJqc,iJkLqc,bLqc,qc->aibkc", region.dhdX, A, region.dhdX, region.dV
            )

            # chunks of one cell, chunks of 7 of 27 cells and a single chunk
            for nbytes in [1, 200_000, 2**30]:
                cartesian.CHUNKSIZE_BYTES = nbytes

                form = fem.IntegralForm([A], field, region.dV, field)
                assert np.allclose(form.integrate()[0], expected)

                out = np.zeros_like(expected)
                values = form.integrate(out=[out])[0]
                assert values is out
                assert np.allclose(out, expected)

    finally:
        cartesian.CHUNKSIZE_BYTES = chunksize_bytes


def test_sparsity_pattern():
    "The assembly with a cached sparsity pattern is equal to the COO-assembly."

    import gc
    import weakref

    from scipy.sparse import coo_matrix

    from felupe.assembly._sparsity import _patterns, sparsity_pattern

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    rng = np.random.default_rng(0)

    def assemble_coo(values, v, u):
        "Assemble the (duplicate) values of shape (a, i, b, k, c) in COO-format."
        rows = v.indices.cai.transpose(1, 2, 0)[:, :, None, None, :]
        cols = u.indices.cai.transpose(1, 2, 0)[None, None, :, :, :]
        rows, cols = [np.broadcast_to(x, values.shape).ravel() for x in [rows, cols]]
        shape = (v.indices.shape[0], u.indices.shape[0])
        matrix = coo_matrix((values.ravel(), (rows, cols)), shape=shape).tocsr()
        matrix.sort_indices()
        return matrix

    displacement = fem.Field(region, dim=3)
    pressure = fem.Field(region, dim=1)

    for v, u in [
        (displacement, displacement),
        (displacement, pressure),
        (pressure, displacement),
        (pressure, pressure),
    ]:
        shape = (8, v.dim, 8, u.dim, mesh.ncells)
        values = rng.normal(size=shape)
        form = fem.assembly.IntegralFormCartesian(values, v, region.dV, u=u)

        matrix = form.assemble(values=values)
        expected = assemble_coo(values, v, u)

        assert matrix.has_canonical_format
        assert np.array_equal(matrix.indptr, expected.indptr)
        assert np.array_equal(matrix.indices, expected.indices)
        assert np.allclose(matrix.data, expected.data)

        # the cached pattern is not modified by in-place changes of the matrix
        matrix.indices[:] = 0
        matrix = form.assemble(values=values)
        assert np.array_equal(matrix.indices, expected.indices)

        # broadcasted values of a uniform grid mesh
        matrix = form.assemble(values=values[..., :1])
        expected = assemble_coo(np.broadcast_to(values[..., :1], shape), v, u)
        assert np.allclose(matrix.toarray(), expected.toarray())

    # the pattern is cached and not copied with the field
    field = fem.Field(region, dim=3)
    assert sparsity_pattern(field, field) is sparsity_pattern(field, field)
    assert field.indices in _patterns
    assert field.copy().indices not in _patterns

    # the cache does not keep the indices alive, the pattern is released with them
    indices = weakref.ref(field.indices)
    del field
    gc.collect()
    assert indices() is None

    # fall back to the COO-assembly for degrees of freedom which are not point-wise
    field = fem.Field(region, dim=3)
    field.indices.cai = field.indices.cai[..., ::-1]
    assert sparsity_pattern(field, field) is None

    shape = (8, 3, 8, 3, mesh.ncells)
    values = rng.normal(size=shape)
    form = fem.assembly.IntegralFormCartesian(values, field, region.dV, u=field)
    matrix = form.assemble(values=values)
    assert np.allclose(matrix.toarray(), assemble_coo(values, field, field).toarray())


def test_bilinearform_lazy_indices():
    "The COO-indices of a bilinear form are only evaluated on demand."

    r, u, p, P, A = pre()

    form = fem.IntegralForm(A, u, r.dV, u).forms[0]
    form.assemble()
    assert "indices" not in vars(form)  # not required with a sparsity pattern

    rows, cols = form.indices
    assert rows.size == cols.size == form.integrate().size
    assert form.indices is form.indices  # evaluated only once


def test_mixed():
    r, v, f, A = pre_mixed()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, v, r.dV, v)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        K = a.assemble(parallel=parallel).toarray()

        z = r.mesh.ndof + 2 * r.mesh.npoints
        assert K.shape == (z, z)

        L = fem.IntegralForm(f, v, r.dV)
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        b = L.assemble(parallel=parallel).toarray()

        assert b.shape == (z, 1)

    L = fem.IntegralForm([f[0], None, f[2]], v, r.dV)
    x = L.integrate()
    b = L.assemble(x)

    assert b.shape == (z, 1)

    a = fem.IntegralForm(
        [A[0], A[1], A[2], A[1], A[3], A[4], A[2], A[4], A[5]], v, r.dV, v
    )
    y = a.integrate()
    K = a.assemble(y)

    assert K.shape == (z, z)

    r, v, f, A = pre_axi_mixed()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, v, r.dV, v)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        K = a.assemble(parallel=parallel).toarray()

        z = r.mesh.ndof + 2 * r.mesh.npoints
        assert K.shape == (z, z)

        L = fem.IntegralForm(f, v, r.dV)
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        b = L.assemble(parallel=parallel).toarray()

        assert b.shape == (z, 1)


if __name__ == "__main__":
    test_linearform()
    test_linearform_broadcast()
    test_bilinearform()
    test_bilinearform_broadcast()
    test_bilinearform_grad_grad_chunks()
    test_sparsity_pattern()
    test_bilinearform_lazy_indices()
    test_axi()
    test_mixed()
