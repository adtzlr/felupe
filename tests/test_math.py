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


def test_math_field():
    m = fem.mesh.Cube()
    e = fem.element.Hexahedron()
    q = fem.quadrature.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3)
    v = fem.FieldContainer((u, u))

    fem.math.values(v)
    fem.math.deformation_gradient(v)

    fem.math.norm([u.values, u.values])
    fem.math.norm(u.values)
    fem.math.interpolate(u)
    fem.math.grad(u)
    fem.math.grad(u, sym=True)
    fem.math.tovoigt(fem.math.strain(v)[:1, :1])
    fem.math.tovoigt(fem.math.strain(v), strain=False)
    fem.math.tovoigt(fem.math.strain(v), strain=True)
    fem.math.strain(v)
    fem.math.extract(u)
    fem.math.extract(u, grad=False)
    fem.math.extract(u, sym=True)
    fem.math.extract(u, grad=True, sym=False, add_identity=False)


def test_math():
    H = (np.random.rand(3, 3, 5, 7) - 0.5) / 10
    F = fem.math.identity(H) + H
    C = fem.math.dot(fem.math.transpose(F), F)
    A = np.random.rand(3, 3, 3, 3, 5, 7)
    B = np.random.rand(3, 3, 3, 5, 7)
    D = np.random.rand(4, 4, 5, 7)

    a = np.random.rand(3, 5, 7)

    fem.math.identity(A=None, dim=3, shape=(8, 20))

    fem.math.cross(a, a)
    fem.math.dya(a, a, mode=1)

    with pytest.raises(ValueError):
        fem.math.dya(a, a, mode=3)

    fem.math.sym(H)

    fem.math.dot(C, C)
    fem.math.dot(C, A, mode=(2, 4))
    fem.math.dot(A, C, mode=(4, 2))

    fem.math.transpose(F, mode=1)
    fem.math.transpose(A, mode=2)

    with pytest.raises(ValueError):
        fem.math.transpose(F, mode=3)

    with pytest.raises(TypeError):
        fem.math.dot(A, B, mode=(4, 3))

    with pytest.raises(TypeError):
        fem.math.dot(B, B, mode=(3, 3))

    fem.math.dot(C, B, mode=(2, 3))
    fem.math.dot(B, C, mode=(3, 2))

    assert fem.math.dot(C, a, mode=(2, 1)).shape == (3, 5, 7)
    assert fem.math.dot(a, C, mode=(1, 2)).shape == (3, 5, 7)
    assert fem.math.dot(a, a, mode=(1, 1)).shape == (5, 7)

    assert fem.math.dot(a, A, mode=(1, 4)).shape == (3, 3, 3, 5, 7)
    assert fem.math.dot(A, a, mode=(4, 1)).shape == (3, 3, 3, 5, 7)
    assert fem.math.dot(A, A, mode=(4, 4)).shape == (3, 3, 3, 3, 3, 3, 5, 7)

    assert fem.math.ddot(C, C, mode=(2, 2)).shape == (5, 7)
    assert fem.math.ddot(C, A, mode=(2, 4)).shape == (3, 3, 5, 7)
    assert fem.math.ddot(A, C, mode=(4, 2)).shape == (3, 3, 5, 7)

    assert fem.math.dddot(B, B, mode=(3, 3)).shape == (5, 7)
    assert fem.math.dddot(B, B, mode=(3, 3), parallel=True).shape == (5, 7)

    assert fem.math.ddot(A, A, mode=(4, 4)).shape == (3, 3, 3, 3, 5, 7)
    assert fem.math.equivalent_von_mises(C).shape == (5, 7)
    assert fem.math.equivalent_von_mises(C[:2, :2, ...]).shape == (5, 7)

    with pytest.raises(TypeError):
        fem.math.ddot(A, B, mode=(4, 3))

    with pytest.raises(TypeError):
        fem.math.ddot(B, B, mode=(3, 3))

    with pytest.raises(TypeError):
        fem.math.dddot(B, C, mode=(3, 2))

    fem.math.ddot(C, B, mode=(2, 3))
    fem.math.ddot(B, C, mode=(3, 2))

    detC = fem.math.det(C)
    fem.math.det(C[:2, :2])
    fem.math.det(C[:1, :1])

    fem.math.inv(C)
    fem.math.inv(C[:2, :2])
    fem.math.inv(C, determinant=detC)
    fem.math.inv(C, full_output=True)
    fem.math.inv(C, sym=True)
    assert np.allclose(fem.math.inv(C[:1, :1]), 1 / C[:1, :1])

    with pytest.raises(ValueError):
        fem.math.inv(D)
        fem.math.inv(D, determinant=1)

    fem.math.dev(C)
    fem.math.cof(C)
    fem.math.dya(C, C)
    fem.math.cdya_ik(F, F)
    fem.math.cdya_il(F, F)
    fem.math.cdya(F, F)

    fem.math.tovoigt(C)
    fem.math.tovoigt(C[:2, :2])
    with pytest.raises(TypeError):
        fem.math.tovoigt(C[:, :2])

    fem.math.eigvals(C)
    fem.math.eigvals(C[:2, :2])
    fem.math.eigvals(C, shear=True)
    fem.math.eigvals(C[:2, :2], shear=True)
    fem.math.eigvalsh(C)
    fem.math.eigh(C)
    fem.math.eig(C)
    fem.math.svd(C)

    fem.math.majortranspose(A)
    fem.math.trace(C)

    for dim in [1, 2]:
        for phi in [180, [0, 180, 360]]:
            points_new = fem.math.revolve_points(np.eye(dim), n=11, phi=phi, axis=0)
            assert points_new.shape[1] == dim + 1

    fem.math.rotate_points(np.eye(2), angle_deg=45, axis=0)
    fem.math.rotate_points(np.eye(2), angle_deg=45, axis=1, center=[0.5, 0.5])


def test_math_linsteps():
    steps = fem.math.linsteps([0, 1], num=10)
    assert len(steps) == 11

    steps = fem.math.linsteps([0, 1, 0], num=(10, 100))
    assert len(steps) == 111

    steps = fem.math.linsteps([1], num=0)
    assert len(steps) == 1
    assert steps[-1] == 1

    steps = fem.math.linsteps([1], num=(0, 1))
    assert len(steps) == 1
    assert steps[-1] == 1

    steps = fem.math.linsteps([0, 1, 5], num=(10, 100), axis=1, axes=None)
    assert len(steps) == 111
    assert np.allclose(steps[-1], (0, 5))


def test_inplane():
    import felupe as fem

    A = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]], dtype=float)
    vectors = [[1, 0, 0], [0, 1, 0]]

    A_inplane = fem.math.inplane(A, vectors)
    assert np.allclose(A_inplane.ravel(), [0, 3, 3, 1])


def test_math_parallel(monkeypatch):
    "The math helpers are equal in serial and in parallel."

    import felupe.math._threads as threads

    rng = np.random.default_rng(0)
    batch = (4, 5)  # quadrature points and cells

    a = rng.random((3, *batch))
    A = rng.random((3, 3, *batch))
    B = rng.random((3, 3, *batch))
    A3 = rng.random((3, 3, 3, *batch))
    A4 = rng.random((3, 3, 3, 3, *batch))
    A4b = rng.random((3, 3, 3, 3, 1, 1))  # broadcasted along the batch axes
    P = rng.random((3, 3))  # without batch axes

    m = fem.math
    cases = [
        (m.dot, (A, B), dict(mode=(2, 2))),
        (m.dot, (a, a), dict(mode=(1, 1))),
        (m.dot, (A4, A4), dict(mode=(4, 4))),
        (m.dot, (A, a), dict(mode=(2, 1))),
        (m.dot, (a, A), dict(mode=(1, 2))),
        (m.dot, (A, A3), dict(mode=(2, 3))),
        (m.dot, (A3, A), dict(mode=(3, 2))),
        (m.dot, (A4, a), dict(mode=(4, 1))),
        (m.dot, (a, A4), dict(mode=(1, 4))),
        (m.dot, (A, A4), dict(mode=(2, 4))),
        (m.dot, (A4, A), dict(mode=(4, 2))),
        (m.dot, (P, A), dict(mode=(2, 2))),
        (m.dot, (P, P), dict(mode=(2, 2))),
        (m.ddot, (A, B), dict(mode=(2, 2))),
        (m.ddot, (A, A4), dict(mode=(2, 4))),
        (m.ddot, (A4, A), dict(mode=(4, 2))),
        (m.ddot, (A, A3), dict(mode=(2, 3))),
        (m.ddot, (A3, A), dict(mode=(3, 2))),
        (m.ddot, (A4, A4), dict(mode=(4, 4))),
        (m.ddot, (A4b, A), dict(mode=(4, 2))),
        (m.ddot, (P, P), dict(mode=(2, 2))),
        (m.dddot, (A3, A3), dict(mode=(3, 3))),
        (m.cdya_ik, (A, B), dict()),
        (m.cdya_il, (A, B), dict()),
        (m.cdya, (A, B), dict()),
    ]

    for fun, args, kwargs in cases:
        expected = fun(*args, parallel=False, **kwargs)

        # a single thread, less threads than cells and more threads than cells
        for cpus in [1, 2, 64]:
            monkeypatch.setattr(threads, "cpu_count", lambda: cpus)
            res = fun(*args, parallel=True, **kwargs)
            assert np.allclose(res, expected)

            out = np.zeros_like(expected)
            res = fun(*args, parallel=True, out=out, **kwargs)
            assert res is out
            assert np.allclose(out, expected)

    # the area change with and without normal vectors
    F = np.eye(3).reshape(3, 3, 1, 1) + 0.1 * A
    for N in [None, a]:
        area = fem.constitution.AreaChange()
        for method in [area.function, area.gradient]:
            expected = method([F], N=N, parallel=False)[0]

            for cpus in [1, 2, 64]:
                monkeypatch.setattr(threads, "cpu_count", lambda: cpus)
                assert np.allclose(method([F], N=N, parallel=True)[0], expected)


def test_einsum_chunks(monkeypatch):
    "Contractions evaluated in chunks are equal to numpy.einsum()."

    import felupe.math._einsum as einsum
    import felupe.math._threads as threads

    rng = np.random.default_rng(0)

    # subscripts, shapes of the operands, chunked index and contraction path
    cases = [
        # the bilinear form of the gradients of the test and the trial field
        (
            "aJqc,iJkLqc,bLqc,qc->aibkc",
            [(8, 3, 4, 5), (3, 3, 3, 3, 4, 5), (8, 3, 4, 5), (4, 5)],
            None,
            ["einsum_path", (0, 3), (0, 1), (0, 1)],
        ),
        # an operand without the axis of the cells, with the same length as the cells
        ("ik...,kj...->ij...", [(3, 3), (3, 3, 3)], None, False),
        # an operand which is broadcasted (length one) along the cells
        ("ik...,kj...->ij...", [(3, 3, 1, 1), (3, 3, 4, 5)], None, False),
        # right-aligned axes of the ellipsis with different numbers of axes
        ("ij...,j...->i...", [(3, 3, 4, 5), (3, 5)], None, False),
        # chunks of the basis functions of a single cell
        ("aJqc,bJqc->abc", [(10, 3, 4, 1), (10, 3, 4, 1)], "a", False),
    ]

    for subscripts, shapes, idx, optimize in cases:
        operands = [rng.random(shape) for shape in shapes]
        expected = np.einsum(subscripts, *operands)

        for chunksize in [None, 1, 2, 1000]:
            # serial, a single thread, less threads than chunks and more CPUs
            for parallel, cpus in [(False, 1), (True, 1), (True, 2), (True, 64)]:
                monkeypatch.setattr(threads, "cpu_count", lambda: cpus)
                kwargs = dict(
                    idx=idx, chunksize=chunksize, optimize=optimize, parallel=parallel
                )

                res = einsum.einsum_chunks(subscripts, *operands, **kwargs)
                assert res.shape == expected.shape
                assert np.allclose(res, expected)

                out = np.zeros_like(expected)
                res = einsum.einsum_chunks(subscripts, *operands, out=out, **kwargs)
                assert res is out
                assert np.allclose(out, expected)


def test_einsum_chunks_layout():
    "A single chunk has the memory layout of NumPy, more chunks are in C-order."

    from felupe.math._einsum import einsum_chunks

    rng = np.random.default_rng(0)
    A = rng.random((3, 4, 50))

    # NumPy returns a (transposed) view of a single operand, which is not in C-order
    subscripts = "ijc->jic"
    expected = np.einsum(subscripts, A)
    assert not expected.flags.c_contiguous

    res = einsum_chunks(subscripts, A)
    assert res.strides == expected.strides
    assert np.allclose(res, expected)

    res = einsum_chunks(subscripts, A, chunksize=10)
    assert res.flags.c_contiguous
    assert np.allclose(res, expected)


def test_einsum_chunks_errors():
    from felupe.math._einsum import einsum_chunks

    A = np.ones((3, 3, 5))

    # the checks are skipped for a single chunk in serial (passed to numpy.einsum)
    for kwargs in [dict(chunksize=2), dict(parallel=True)]:
        with pytest.raises(ValueError):  # implicit output
            einsum_chunks("ij...,jk...", A, A, **kwargs)

        with pytest.raises(ValueError):  # wrong number of operands
            einsum_chunks("ij...,jk...->ik...", A, **kwargs)

        with pytest.raises(ValueError):  # summed index
            einsum_chunks("ij...,jk...->ik...", A, A, idx="j", **kwargs)

        with pytest.raises(ValueError):  # more than one index
            einsum_chunks("ij...,jk...->ik...", A, A, idx="ik", **kwargs)

        with pytest.raises(ValueError):  # repeated index within an operand
            einsum_chunks("iic->ic", A, idx="i", **kwargs)

        with pytest.raises(ValueError):  # index of the output not given in the inputs
            einsum_chunks("ijc->ijkc", A, **kwargs)


def test_einsum_chunks_single_chunk():
    "Contractions without a free last index of the output are a single chunk."

    from felupe.math._einsum import einsum_chunks

    rng = np.random.default_rng(0)
    A = rng.random((3, 3, 5))

    for kwargs in [dict(chunksize=2), dict(parallel=True)]:
        # a full contraction (scalar output) and a repeated last index of the output
        for subscripts, operands in [("ijc,ijc->", (A, A)), ("iic->ci", (A[..., :3],))]:
            expected = np.einsum(subscripts, *operands)
            assert np.allclose(einsum_chunks(subscripts, *operands, **kwargs), expected)

            out = np.zeros_like(expected)
            res = einsum_chunks(subscripts, *operands, out=out, **kwargs)
            assert res is out
            assert np.allclose(out, expected)


def test_einsum_chunks_kwargs():
    "Optional keyword-arguments are passed to numpy.einsum()."

    from felupe.math._einsum import einsum_chunks

    A = np.ones((3, 3, 5), dtype=np.float32)

    for kwargs in [dict(), dict(chunksize=2), dict(parallel=True)]:
        res = einsum_chunks("ij...,jk...->ik...", A, A, dtype=np.float64, **kwargs)
        assert res.dtype == np.float64
        assert np.allclose(res, 3)

        with pytest.raises(TypeError):  # the casting is passed to numpy.einsum
            out = np.zeros((3, 3, 5), dtype=np.int64)
            einsum_chunks("ij...,jk...->ik...", A, A, out=out, casting="safe", **kwargs)


def test_thread_pool(monkeypatch):
    "The pool of threads is re-used."

    import felupe.math._einsum as einsum
    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)
    monkeypatch.setattr(threads, "_pool", None)
    A = np.ones((3, 3, 10))

    einsum.einsum_chunks("ij...,jk...->ik...", A, A, parallel=True)
    pool = threads.thread_pool()

    einsum.einsum_chunks("ij...,jk...->ik...", A, A, parallel=True)
    assert threads.thread_pool() is pool

    pool.shutdown()


def test_map_threaded_nested(monkeypatch):
    "Calls from a thread of the pool are evaluated in serial (instead of blocking)."

    from threading import Thread

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)
    monkeypatch.setattr(threads, "_pool", None)

    F = np.ones((3, 3, 8, 400))
    slices = [slice(start, start + 50) for start in range(0, 400, 50)]

    def chunk(cells):
        # a math-function with parallel=True within a task of the pool
        return fem.math.ddot(F[..., cells], F[..., cells], parallel=True)

    results = []
    thread = Thread(
        target=lambda: results.append(threads.map_threaded(chunk, slices)), daemon=True
    )
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "The nested evaluation did not finish."
    assert np.allclose(np.concatenate(results[0], axis=-1), 9)

    threads.thread_pool().shutdown()


def test_thread_pool_reset(monkeypatch):
    "After a reset (in a forked child process), a new pool of threads is created."

    import felupe.math._einsum as einsum
    import felupe.math._threads as threads

    # the pool and its lock of the module are restored after the test
    monkeypatch.setattr(threads, "cpu_count", lambda: 2)
    monkeypatch.setattr(threads, "_pool", None)
    monkeypatch.setattr(threads, "_pool_lock", threads._pool_lock)

    pool = threads.thread_pool()
    lock = threads._pool_lock

    threads._reset_thread_pool()
    assert threads._pool is None
    assert threads._pool_lock is not lock

    A = np.ones((3, 3, 10))
    res = einsum.einsum_chunks("ij...,jk...->ik...", A, A, parallel=True)
    assert np.allclose(res, 3)
    assert threads.thread_pool() is not pool

    pool.shutdown()
    threads.thread_pool().shutdown()


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
@pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called")  # by JAX
def test_thread_pool_fork(monkeypatch):
    "A forked child process creates its own pool of threads."

    import multiprocessing
    import os
    from threading import Barrier

    import felupe.math._einsum as einsum
    import felupe.math._threads as threads

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("The fork start method is not available.")

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)
    monkeypatch.setattr(threads, "_pool", None)

    # start all threads of the pool in the parent process (idle afterwards)
    pool = threads.thread_pool()
    barrier = Barrier(2)
    list(pool.map(lambda i: barrier.wait(), range(2)))

    A = np.ones((3, 3, 10))
    expected = np.einsum("ij...,jk...->ik...", A, A)

    def child():
        res = einsum.einsum_chunks("ij...,jk...->ik...", A, A, parallel=True)
        os._exit(0 if np.allclose(res, expected) else 1)

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(timeout=60)

    if process.is_alive():
        process.kill()
        pytest.fail("The evaluation in the child process did not finish.")

    assert process.exitcode == 0

    pool.shutdown()


def test_cpu_count(monkeypatch):
    "The number of usable CPUs, also without the newer functions of os."

    import os

    from felupe.math._threads import cpu_count

    assert cpu_count() >= 1
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    assert cpu_count() >= 1
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert cpu_count() >= 1


if __name__ == "__main__":
    test_math()
    test_math_field()
    test_math_linsteps()
    test_inplane()
    test_einsum_chunks_errors()
