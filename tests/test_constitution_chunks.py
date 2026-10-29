# SPDX-License-Identifier: GPL-3.0-or-later

from threading import Thread

import numpy as np
import pytest

import felupe as fem


class ChunkedMaterial(fem.ConstitutiveMaterial):
    "A material which is evaluated in chunks of cells."

    def __init__(self, chunksize=8192, parallel=False):
        self.x = [np.eye(3), np.zeros(2)]
        self.kwargs = {}
        self.chunksize = chunksize
        self.parallel = parallel

    def _gradient(self, x, factor=1.0):
        F, statevars = x[0], x[-1]

        # math-functions with parallel=True are evaluated in serial within a chunk
        P = factor * fem.math.dot(F, F, parallel=self.parallel)
        statevars_new = statevars + fem.math.det(F) if statevars is not None else None

        return [P, statevars_new]

    def _hessian(self, x, factor=1.0):
        F = x[0]

        # a hessian per cell and a constant (broadcasted) hessian
        A = factor * fem.math.cdya(F, F, parallel=self.parallel)
        B = np.ones((3, 3, 3, 3, 1, 1))

        return [A, B]


def deformation_gradient(ncells, npoints=4):
    rng = np.random.default_rng(ncells)
    eye = np.eye(3).reshape(3, 3, 1, 1)
    return eye + 0.1 * rng.random((3, 3, npoints, ncells))


def test_chunks(monkeypatch):
    "Chunks of cells are equal to the evaluation of all cells at once."

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)

    for ncells in [1, 2, 7, 100]:
        F = deformation_gradient(ncells)
        statevars = np.ones((2, *F.shape[-2:]))

        whole = ChunkedMaterial(chunksize=None)
        P, statevars_new = whole.gradient([F, statevars], factor=2.0)
        A, B = whole.hessian([F, statevars], factor=2.0)

        assert np.allclose(P, 2 * fem.math.dot(F, F))
        assert np.allclose(statevars_new, 1 + fem.math.det(F))
        assert np.allclose(A, 2 * fem.math.cdya(F, F))

        # chunks of two cells, three cells (with a remainder) and a single chunk
        for chunksize in [8, 12, 10000]:
            for parallel in [False, True]:
                umat = ChunkedMaterial(chunksize=chunksize, parallel=parallel)

                res = umat.gradient([F, statevars], factor=2.0)
                assert np.allclose(res[0], P)
                assert np.allclose(res[1], statevars_new)

                res = umat.hessian([F, statevars], factor=2.0)
                assert np.allclose(res[0], A)
                assert res[1].shape == (3, 3, 3, 3, 1, 1)  # taken from the first chunk
                assert np.allclose(res[1], B)

                # the first results are stored in the given output arrays
                out = np.zeros_like(P)
                res = umat.gradient([F, statevars], out=out, factor=2.0)
                assert res[0] is out
                assert np.allclose(out, P)

                out = np.zeros_like(A)
                res = umat.hessian([F, statevars], out=out, factor=2.0)
                assert res[0] is out
                assert np.allclose(out, A)


def test_chunks_inputs():
    "Inputs without the axis of the cells are passed unchanged to all chunks."

    F = deformation_gradient(ncells=10)
    umat = ChunkedMaterial(chunksize=8)

    # state variables of None, e.g. by the views on a material
    P, statevars_new = umat.gradient([F, None])
    assert np.allclose(P, fem.math.dot(F, F))
    assert statevars_new is None

    # broadcasted state variables
    P, statevars_new = umat.gradient([F, np.ones((2, 1, 1))])
    assert np.allclose(statevars_new, 1 + fem.math.det(F))

    # inputs which are all broadcasted along the cells are evaluated at once
    P, statevars_new = umat.gradient([F[..., :1], np.ones((2, 1, 1))])
    assert P.shape == (3, 3, 4, 1)


def test_chunks_nested(monkeypatch):
    "Math-functions with parallel=True within the threads of the chunks don't block."

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)
    monkeypatch.setattr(threads, "_pool", None)

    F = deformation_gradient(ncells=400)
    umat = ChunkedMaterial(chunksize=200, parallel=True)

    results = []
    thread = Thread(target=lambda: results.append(umat.hessian([F, None])), daemon=True)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "The evaluation in chunks did not finish."
    assert np.allclose(results[0][0], fem.math.cdya(F, F))

    threads.thread_pool().shutdown()


def test_chunks_solidbody():
    "A solid body with a material evaluated in chunks re-uses the output arrays."

    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    field[0].values[:] = 0.1 * mesh.points**2

    solid = fem.SolidBody(ChunkedMaterial(chunksize=64), field)
    expected = fem.SolidBody(ChunkedMaterial(chunksize=None), field)

    P = solid.evaluate.gradient(field)
    assert np.allclose(P[0], expected.evaluate.gradient(field)[0])
    assert solid.results.statevars.shape == (2, 8, mesh.ncells)

    # the stress of the first evaluation is the output array of the second evaluation
    stress = solid.results.gradient
    solid.evaluate.gradient(field)
    assert solid.results.gradient is stress

    r = solid.assemble.vector(field)
    assert np.allclose(r.toarray(), expected.assemble.vector(field).toarray())

    A = solid.evaluate.hessian(field)
    assert np.allclose(A[0], expected.evaluate.hessian(field)[0])

    elasticity = solid.results.hessian
    solid.evaluate.hessian(field)
    assert solid.results.hessian is elasticity


def composed_materials(parallel):
    "Return a list of composed materials (with inputs) for the evaluation in chunks."

    mesh = fem.Cube(n=4)  # 27 cells
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldsMixed(region, n=3)

    rng = np.random.default_rng(6)
    field[0].values[:] = rng.random(mesh.points.shape) / 20
    F, p, J = field.extract()
    p = rng.random(p.shape) / 10
    J = 1 + rng.random(J.shape) / 10
    q, c = F.shape[-2:]

    nh = dict(mu=1.0, bulk=2.0)
    composite = fem.NeoHooke(mu=1.0) & fem.Volumetric(bulk=2.0)
    composite.parallel = parallel
    strain = fem.MaterialStrain(
        material=fem.constitution.linear_elastic_plastic_isotropic_hardening,
        λ=1.0,
        μ=1.0,
        σy=0.01,
        K=0.1,
        statevars=(1, (3, 3)),
    )
    strain.parallel = parallel

    return [
        (
            fem.ThreeFieldVariation(fem.NeoHooke(**nh), parallel=parallel),
            [F, p, J, np.zeros((0, q, c))],
        ),
        (
            fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), bulk=2.0, parallel=parallel),
            [F, p, J, np.zeros((0, q, c))],
        ),
        (
            fem.ThirdMediumContactMixed(
                fem.NeoHooke(**nh), gamma=0.1, alpha_r=0.2, p_r=0.3, parallel=parallel
            ),
            [F, rng.random((9, q, c)), rng.random((9, 3, q, c)), np.zeros((0, q, c))],
        ),
        (
            fem.OgdenRoxburgh(
                fem.NeoHooke(**nh), r=3, m=1, beta=0.1, parallel=parallel
            ),
            [F, rng.random((1, q, c)) / 100],
        ),
        (composite, [F, np.zeros((0, q, c))]),
        (strain, [F, rng.random((28, q, c)) / 100]),
    ]


def test_chunks_composed_materials(monkeypatch):
    "Composed materials in chunks are equal to the evaluation of all cells at once."

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)

    for parallel in [False, True]:
        wholes = composed_materials(parallel=False)
        chunked = composed_materials(parallel=parallel)

        for (whole, x), (umat, _) in zip(wholes, chunked):
            whole.chunksize = None
            umat.chunksize = 2 * x[0].shape[-2]  # chunks of two cells

            for method in ["gradient", "hessian"]:
                expected = getattr(whole, method)(x)
                res = getattr(umat, method)(x)

                assert len(res) == len(expected)

                for r, e in zip(res, expected):
                    if e is None:
                        assert r is None
                    else:
                        assert r.shape == e.shape
                        assert np.allclose(r, e)

    # the keyword-arguments of a strain-based material are not changed
    strain, x = composed_materials(parallel=False)[-1]
    kwargs = dict(strain.kwargs)
    strain.gradient(x)
    strain.hessian(x)
    assert strain.kwargs == kwargs
    assert "tangent" not in strain.kwargs


def test_parallel_from_material():
    "Composed materials take the flag for threads from the material by default."

    for parallel in [False, True]:
        nh = fem.NeoHooke(mu=1.0, parallel=parallel)

        for umat in [
            fem.ThreeFieldVariation(nh),
            fem.NearlyIncompressible(nh, bulk=2.0),
            fem.ThirdMediumContactMixed(nh, gamma=0.1, alpha_r=0.2, p_r=0.3),
            fem.OgdenRoxburgh(nh, r=3, m=1, beta=0.1),
            nh & fem.Volumetric(bulk=2.0),
            fem.Volumetric(bulk=2.0) & nh,
        ]:
            assert umat.parallel is parallel

    # the flag is given explicitly
    nh = fem.NeoHooke(mu=1.0, parallel=True)
    assert fem.ThreeFieldVariation(nh, parallel=False).parallel is False
    assert fem.ThreeFieldVariation(fem.NeoHooke(mu=1.0), parallel=True).parallel


def test_chunks_tensortrax(monkeypatch):
    "The tensortrax-materials are evaluated in chunks of cells by FElupe."

    import tensortrax.math as tm

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)

    @fem.total_lagrange
    def neo_hooke_total_lagrange(F, mu=1.0):
        C = F.T @ F
        return mu * tm.special.dev(tm.linalg.det(C) ** (-1 / 3) * C) @ tm.linalg.inv(C)

    mesh = fem.Cube(n=4)  # 27 cells
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    field[0].values[:] = np.random.default_rng(7).random(mesh.points.shape) / 20
    F = field.extract()[0]
    q, c = F.shape[-2:]

    # the old state variables, the inelastic right Cauchy-Green deformation tensor is the
    # identity (all-zero deviation from the identity)
    Ci = np.zeros((6, q, c))

    materials = [
        (
            lambda parallel: fem.Hyperelastic(fem.neo_hooke, mu=1.0, parallel=parallel),
            [F, np.zeros((0, q, c))],
            {},
        ),
        (
            lambda parallel: fem.Hyperelastic(
                fem.constitution.finite_strain_viscoelastic,
                nstatevars=6,
                mu=1.0,
                eta=1.0,
                parallel=parallel,
            ),
            [F, Ci.copy()],
            {"dtime": 1.0},  # the time increment of the time-dependent material
        ),
        (
            lambda parallel: fem.MaterialAD(
                neo_hooke_total_lagrange, mu=1.0, parallel=parallel
            ),
            [F, np.zeros((0, q, c))],
            {},
        ),
    ]

    for Material, x, kwargs in materials:
        whole = Material(parallel=False)
        whole.chunksize = None

        for parallel in [False, True]:
            umat = Material(parallel=parallel)
            umat.chunksize = 2 * q  # chunks of two cells

            for method in ["gradient", "hessian"]:
                expected = getattr(whole, method)(x, **kwargs)
                res = getattr(umat, method)(x, **kwargs)

                for r, e in zip(res, expected):
                    if e is None:
                        assert r is None
                    else:
                        assert np.allclose(r, e)


def test_chunks_not_implemented():
    "A material must implement either the public or the chunked methods."

    F = deformation_gradient(ncells=3)
    umat = fem.ConstitutiveMaterial()

    with pytest.raises(NotImplementedError):
        umat.gradient([F, None])

    with pytest.raises(NotImplementedError):
        umat.hessian([F, None])


def test_constitutive_material_decorator():
    "The methods of a decorated material take precedence over the base class."

    class MyMaterial:
        def __init__(self, a=2.0):
            self.x = [np.eye(3), np.zeros(0)]
            self.kwargs = {"a": a}

        def gradient(self, x):
            return [self.kwargs["a"] * x[0], x[-1]]

        def hessian(self, x):
            return [self.kwargs["a"] * np.ones((3, 3, 3, 3, 1, 1))]

    F = deformation_gradient(ncells=3)
    umat = fem.constitutive_material(MyMaterial)(a=3.0)

    assert isinstance(umat, fem.ConstitutiveMaterial)
    assert np.allclose(umat.gradient([F, None])[0], 3 * F)
    assert np.allclose(umat.hessian([F, None])[0], 3)

    # a class which is already derived from the base class
    umat = fem.constitutive_material(fem.NeoHooke)(mu=1.0)
    assert np.allclose(
        umat.gradient([F, None])[0], fem.NeoHooke(mu=1.0).gradient([F])[0]
    )


if __name__ == "__main__":
    test_chunks_inputs()
    test_chunks_solidbody()
    test_chunks_not_implemented()
    test_constitutive_material_decorator()
