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


def test_axi_reference():
    """The axisymmetric integral forms of all combinations of gradients are equal to a
    reference with the three-dimensional values and gradients of the basis functions."""

    rng = np.random.default_rng(1234)
    region = fem.RegionQuad(fem.Rectangle(a=(0, 1), b=(2, 3), n=(3, 4)))
    u = fem.FieldAxisymmetric(region, dim=2)
    p = fem.Field(region, dim=1)
    w = fem.Field(region, dim=3)

    dhdX = region.dhdX  # (a, J, q, c)
    h = np.broadcast_to(region.h, dhdX[:, 0].shape)  # (a, q, c)
    R = u.radius
    dV = 2 * np.pi * R * region.dV
    a, q, c = h.shape

    def basis(field, grad):
        "Basis (a, i, k, [L], q, c): the values (v_x, v_r, v_r / R) or the gradients."
        if field.dim == 1:
            return (dhdX if grad else h)[:, None]

        axisymmetric = isinstance(field, fem.FieldAxisymmetric)
        dim = 3 if axisymmetric else field.dim

        if grad:
            # the gradients of a field on a 2d-mesh (and the out-of-plane gradient)
            b = np.zeros((a, field.dim, dim, 3 if axisymmetric else 2, q, c))
            for i in range(field.dim):
                b[:, i, i, :2] = dhdX
            if axisymmetric:
                b[:, 1, 2, 2] = h / R
        else:
            b = np.zeros((a, field.dim, dim, q, c))
            for i in range(field.dim):
                b[:, i, i] = h
            if axisymmetric:
                b[:, 1, 2] = h / R

        return b

    def reference(fun, v, grad_v, u=None, grad_u=None):
        bv = basis(v, grad_v)
        sv = "klmn"[: bv.ndim - 4]
        if u is None:
            return np.einsum(f"ai{sv}qc,{sv}qc,qc->aic", bv, fun, dV)
        bu = basis(u, grad_u)
        su = "rstu"[: bu.ndim - 4]
        return np.einsum(f"ai{sv}qc,{sv}{su}qc,bj{su}qc,qc->aibjc", bv, fun, bu, dV)

    for shape, grad_v, field, grad_u in [
        ((3, 3), True, None, None),
        ((3,), False, None, None),
        ((3, 3, 3, 3), True, u, True),
        ((3, 3, 3), False, u, True),
        ((3, 3, 3), True, u, False),
        ((3, 3), False, u, False),
        ((3, 3), True, p, False),
        ((3, 3, 3), True, w, False),
        ((3,), False, p, False),
        ((3, 3, 3, 2), True, w, True),
    ]:
        fun = rng.standard_normal((*shape, q, c))
        form = fem.assembly.IntegralFormAxisymmetric(
            fun, u, region.dV, field, grad_v=grad_v, grad_u=grad_u
        )
        values = form.integrate()
        expected = reference(fun, u, grad_v, field, grad_u)

        assert np.allclose(values, expected.reshape(values.shape))

    # a symmetric function gives a symmetric matrix
    fun = rng.standard_normal((3, 3, q, c))
    fun = fun + np.einsum("ij...->ji...", fun)
    form = fem.IntegralForm(
        [fun], u.as_container(), region.dV, u.as_container(), [False], [False]
    )
    matrix = form.assemble().toarray()

    assert np.allclose(matrix, matrix.T)


def test_linearform():
    r, u, p, P, A = pre()

    for parallel in [False, True]:
        L = fem.IntegralForm(P, u, r.dV, grad_v=[True])
        x = L.integrate(parallel=parallel)
        y = np.einsum("aJqc,iJqc,qc->aic", r.dhdX, P[0], r.dV, optimize=True)
        assert np.allclose(x[0], y)
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

                for parallel in [False, True]:
                    values = form.integrate(parallel=parallel)[0]
                    assert np.allclose(values, expected)

                    out = np.zeros_like(expected)
                    values = form.integrate(parallel=parallel, out=[out])[0]
                    assert values is out
                    assert np.allclose(out, expected)

    finally:
        cartesian.CHUNKSIZE_BYTES = chunksize_bytes


def test_bilinearform_grad_grad_threads(monkeypatch):
    "The chunks are evaluated by threads (one per usable CPU, at most one per chunk)."

    import felupe.assembly._cartesian as cartesian
    import felupe.math._threads as threads

    monkeypatch.setattr(cartesian, "CHUNKSIZE_BYTES", 200_000)  # 4 chunks of 27 cells

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    A = fem.NeoHooke(mu=1.0, bulk=2.0).hessian(field.extract())[0]

    form = fem.IntegralForm([A], field, region.dV, field)
    expected = form.integrate(parallel=False)[0]

    # one thread (serial), less threads than chunks and more CPUs than chunks
    for cpus in [1, 2, 64]:
        monkeypatch.setattr(threads, "cpu_count", lambda: cpus)
        assert np.allclose(form.integrate(parallel=True)[0], expected)


def test_integrate_parallel(monkeypatch):
    "All branches of the integration are equal in serial and in parallel."

    import felupe.math._threads as threads

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    u = fem.Field(region, dim=3)
    p = fem.Field(region)
    dV = region.dV

    rng = np.random.default_rng(0)
    q, c = dV.shape

    # function shapes (without the quadrature points and cells), the test and trial
    # fields and the gradient flags, for all branches of the linear and bilinear forms
    cases = [
        ((3,), u, None, False, False),
        ((), p, None, False, False),
        ((3, 3), u, None, True, False),
        ((), p, p, False, False),
        ((3,), u, p, False, False),
        ((3, 3), u, u, False, False),
        ((3, 3, 3, 3), u, u, False, False),
        ((3, 3), u, p, True, False),
        ((3, 3, 3), u, u, True, False),
        ((3, 3), p, u, False, True),
        ((3, 3, 3), u, u, False, True),
        ((3, 3, 3, 3), u, u, True, True),
    ]

    for shape, v, w, grad_v, grad_u in cases:
        # one function per quadrature point and cell or broadcasted
        for size in [(q, c), (1, 1)]:
            fun = rng.random((*shape, *size))
            form = fem.assembly.IntegralFormCartesian(
                fun=fun, v=v, dV=dV, u=w, grad_v=grad_v, grad_u=grad_u
            )
            expected = form.integrate(parallel=False)

            # less threads than cells and more threads than cells
            for cpus in [2, 64]:
                monkeypatch.setattr(threads, "cpu_count", lambda: cpus)
                assert np.allclose(form.integrate(parallel=True), expected)

                out = np.zeros_like(expected)
                values = form.integrate(parallel=True, out=out)
                assert values is out
                assert np.allclose(out, expected)


def test_basis_parallel(monkeypatch):
    "The bases are equal in serial and in parallel, also for three cells in 3d."

    import felupe.math._threads as threads

    # the identity matrix of the basis has the same length as the cells
    mesh = fem.Cube(n=(4, 2, 2))
    region = fem.RegionHexahedron(mesh, hess=True)
    field = fem.Field(region, dim=3)
    assert mesh.ncells == field.dim

    expected = fem.assembly.expression.BasisField(field, parallel=False).basis

    for cpus in [2, 3, 64]:
        monkeypatch.setattr(threads, "cpu_count", lambda: cpus)
        basis = fem.assembly.expression.BasisField(field, parallel=True).basis

        assert np.allclose(basis, expected)
        assert np.allclose(basis.grad, expected.grad)
        assert np.allclose(basis.hess, expected.hess)


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


def test_single_block():
    "A single block is returned without the block-assembly."

    from scipy.sparse import bmat, vstack

    r, u, p, P, A = pre()

    a = fem.IntegralForm(A, u, r.dV, u)
    K = a.assemble()
    assert K.format == "csr"
    assert np.allclose(K.toarray(), bmat([[a.forms[0].assemble()]]).toarray())

    L = fem.IntegralForm(P, u, r.dV)
    b = L.assemble()
    assert b.format == "csr"
    assert np.allclose(b.toarray(), vstack([L.forms[0].assemble()]).toarray())


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
    test_single_block()
    test_bilinearform_lazy_indices()
    test_axi()
    test_axi_reference()
    test_mixed()
