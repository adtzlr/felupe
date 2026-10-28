# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import felupe as fem
import felupe.constitution.tensortrax.models as models


def pre(sym, add_identity, add_random=False):
    m = fem.mesh.Cube()
    e = fem.element.Hexahedron()
    q = fem.quadrature.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3)

    if add_random:
        np.random.seed(55601)
        u.values += np.random.rand(*u.values.shape) / 20

    v = fem.FieldContainer([u])
    return r, v.extract(grad=True, sym=sym, add_identity=add_identity)


def pre_mixed(sym, add_identity):
    m = fem.mesh.Cube()
    e = fem.element.Hexahedron()
    q = fem.quadrature.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3)
    v = fem.Field(r, dim=1)
    z = fem.Field(r, dim=1, values=1.0)
    w = fem.FieldContainer([u, v, z])
    return r, w.extract(grad=True, sym=sym, add_identity=add_identity), m


def test_nh():
    r, F = pre(sym=False, add_identity=True)

    for parallel in [False, True]:
        for nh in [
            fem.constitution.NeoHooke(mu=1.0, bulk=2.0, parallel=parallel),
            fem.constitution.NeoHookeCompressible(mu=1.0, lmbda=2.0, parallel=parallel),
            fem.constitution.LinearElasticLargeStrain(E=1.0, nu=0.3, parallel=parallel),
        ]:
            W = nh.function(F)
            P = nh.gradient(F)[:-1]
            A = nh.hessian(F)

            Wx = nh.energy(F)
            Px = nh.stress(F)[:-1]
            Ax = nh.elasticity(F)

            ax = nh.plot(incompressible=False)
            plt.close(ax.get_figure())

            assert np.allclose(W, Wx)
            assert np.allclose(P, Px)
            assert np.allclose(A, Ax)

            assert W[0].shape == F[0].shape[-2:]
            assert P[0].shape == (3, 3, *F[0].shape[-2:])
            assert A[0].shape == (3, 3, 3, 3, *F[0].shape[-2:])

            nh = fem.constitution.NeoHooke(mu=None, bulk=2.0, parallel=parallel)

            assert np.allclose(P, 0)

            nh = fem.constitution.NeoHooke(mu=1.0, parallel=parallel)

            W = nh.function(F)
            P = nh.gradient(F)[:-1]
            A = nh.hessian(F)

            assert W[0].shape == F[0].shape[-2:]
            assert P[0].shape == (3, 3, *F[0].shape[-2:])
            assert A[0].shape == (3, 3, 3, 3, *F[0].shape[-2:])

    assert np.all(nh.is_stable(F))


def test_nh_out():
    r, F = pre(sym=False, add_identity=True, add_random=True)

    for parallel in [False, True]:
        for kwargs in [
            dict(mu=None, bulk=1.0),
            dict(mu=1.0, bulk=None),
            dict(mu=1.0, bulk=2.0),
        ]:
            nh = fem.constitution.NeoHooke(**kwargs, parallel=parallel)

            P = nh.gradient(F)[0]
            A = nh.hessian(F)[0]

            # re-used output arrays with non-zero, NaN and inf values must be reset
            for value in [1.0, np.nan, np.inf]:
                Pout = np.full_like(P, value)
                Aout = np.full_like(A, value)

                Px = nh.gradient(F, out=Pout)[0]
                Ax = nh.hessian(F, out=Aout)[0]

                assert Px is Pout
                assert Ax is Aout

                assert np.allclose(Px, P)
                assert np.allclose(Ax, A)


def test_nh_chunks(monkeypatch):
    "The Neo-Hookean materials are evaluated in chunks of cells."

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)

    mesh = fem.Cube(n=4)  # 27 cells
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    field[0].values[:] = np.random.default_rng(55601).random(mesh.points.shape) / 20
    F = field.extract()[0]
    x = [F, np.zeros((0, *F.shape[-2:]))]

    for Material, kwargs in [
        (fem.NeoHooke, dict(mu=None, bulk=1.0)),
        (fem.NeoHooke, dict(mu=1.0, bulk=None)),
        (fem.NeoHooke, dict(mu=1.0, bulk=2.0)),
        (fem.NeoHookeCompressible, dict(mu=1.0, lmbda=None)),
        (fem.NeoHookeCompressible, dict(mu=1.0, lmbda=2.0)),
    ]:
        whole = Material(**kwargs)
        whole.chunksize = None
        P = whole.gradient(x)[0]
        A = whole.hessian(x)[0]

        for parallel in [False, True]:
            umat = Material(**kwargs, parallel=parallel)
            umat.chunksize = 2 * F.shape[-2]  # chunks of two cells

            assert np.allclose(umat.gradient(x)[0], P)
            assert np.allclose(umat.hessian(x)[0], A)


def test_linear_tensor_notation_chunks(monkeypatch):
    "The stress of the linear-elastic material is evaluated in chunks of cells."

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)

    mesh = fem.Cube(n=4)  # 27 cells
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    field[0].values[:] = np.random.default_rng(8).random(mesh.points.shape) / 20
    F = field.extract()[0]
    x = [F, np.zeros((0, *F.shape[-2:]))]

    whole = fem.constitution.LinearElasticTensorNotation(E=1.0, nu=0.3)
    whole.chunksize = None
    P = whole.gradient(x)[0]

    for parallel in [False, True]:
        umat = fem.constitution.LinearElasticTensorNotation(
            E=1.0, nu=0.3, parallel=parallel
        )
        umat.chunksize = 2 * F.shape[-2]  # chunks of two cells

        assert np.allclose(umat.gradient(x)[0], P)
        assert np.allclose(umat.stress(x)[0], P)

        # the elasticity tensor is constant
        A = umat.hessian(x)[0]
        assert A.shape == (3, 3, 3, 3, 1, 1)
        assert np.allclose(
            fem.math.ddot(A, F - np.eye(3).reshape(3, 3, 1, 1), mode=(4, 2)), P
        )


def test_linear():
    r, F = pre(sym=False, add_identity=True)

    check_stress = []
    check_dsde = []

    for Material in [
        (fem.constitution.LinearElastic, {}),
        (fem.constitution.LinearElasticTensorNotation, dict(parallel=False)),
        (fem.constitution.LinearElasticTensorNotation, dict(parallel=True)),
    ]:
        LinearElastic, kwargs = Material

        le = LinearElastic(E=1.0, nu=0.3, **kwargs)

        stress = le.gradient(F)[:-1]
        dsde = le.hessian(F)
        dsde2 = le.hessian(shape=F[0].shape[-2:])

        assert le.elasticity()[0].shape[-2:] == (1, 1)

        check_stress.append(stress)
        check_dsde.append([dsde[0][..., 0, 0], dsde2[0][..., 0, 0]])

        assert dsde[0].shape[:-2] == dsde2[0].shape[:-2]
        assert np.allclose(stress, 0)

    assert np.allclose(*check_stress)
    assert np.allclose(*check_dsde)


def test_linear_orthotropic():
    r, F = pre(sym=False, add_identity=True)

    lmbda, mu = fem.constitution.lame_converter_orthotropic(
        E=[6919, 271, 450],
        nu=[0.388, 0.278, 0.375],
        G=[262, 34, 354],
    )

    for Material in [
        (fem.constitution.LinearElasticOrthotropic, {}),
    ]:
        LinearElasticOrtho, kwargs = Material

        # doi.org/10.2478/ace-2018-0027 (pine wood)
        le = LinearElasticOrtho(
            E=[6919, 271, 450],
            nu=[0.388, 0.278, 0.375],
            G=[262, 34, 354],
            **kwargs,
        )

        stress = le.gradient(F)[:-1]
        dsde = le.hessian(F)

        assert le.elasticity()[0].shape[-2:] == (1, 1)

        assert stress[0].shape == (3, 3, *F[0].shape[-2:])
        assert dsde[0].shape == (3, 3, 3, 3, 1, 1)

        assert np.allclose(stress, 0)


def test_linear_planestress():
    r, F = pre(sym=False, add_identity=True)
    F = [F[0][:2][:, :2]]

    le = fem.constitutive_material(fem.constitution.LinearElasticPlaneStress)(
        E=1.0, nu=0.3
    )

    stress = le.gradient(F)[:-1]
    dsde = le.hessian(F)
    dsde = le.hessian(F)
    dsde2 = le.hessian(shape=F[0].shape[-2:])

    assert le.elasticity()[0].shape[-2:] == (1, 1)

    check_dsde = [dsde[0][..., 0, 0], dsde2[0][..., 0, 0]]

    assert np.allclose(*check_dsde)

    stress_full = le.stress(F)
    strain_full = le.strain(F)

    assert stress_full[0].shape == (3, 3, *F[0].shape[-2:])
    assert strain_full[0].shape == (3, 3, *F[0].shape[-2:])

    assert stress[0].shape == (2, 2, *F[0].shape[-2:])
    assert dsde[0].shape == (2, 2, 2, 2, 1, 1)

    assert np.allclose(stress, 0)


def test_linear_planestrain():
    r, F = pre(sym=False, add_identity=True)
    F = [F[0][:2][:, :2]]

    le = fem.constitution.LinearElasticPlaneStrain(E=1.0, nu=0.3)

    stress = le.gradient(F)[:-1]
    dsde = le.hessian(F)
    dsde = le.hessian(F)

    stress_full = le.stress(F)
    strain_full = le.strain(F)

    assert stress_full[0].shape == (3, 3, *F[0].shape[-2:])
    assert strain_full[0].shape == (3, 3, *F[0].shape[-2:])

    assert stress[0].shape == (2, 2, *F[0].shape[-2:])
    assert dsde[0].shape == (2, 2, 2, 2, 1, 1)

    assert np.allclose(stress, 0)


def test_kinematics():
    r, F = pre(sym=False, add_identity=True)

    N = F[0][:, 0]

    for parallel in [False, True]:
        lc = fem.constitution.LineChange(parallel=parallel)
        ac = fem.constitution.AreaChange(parallel=parallel)
        vc = fem.constitution.VolumeChange(parallel=parallel)

        xf = lc.function(F)
        xg = lc.gradient(F)
        xg = lc.gradient(F, parallel=parallel)

        Yf = ac.function(F, N)
        Yf = ac.function(F, N, parallel=parallel)
        Yg = ac.gradient(F, N)
        Yg = ac.gradient(F, N, parallel=parallel)

        yf = ac.function(F)
        yg = ac.gradient(F)

        zf = vc.function(F)
        zg = vc.gradient(F)
        zh = vc.hessian(F)
        zh = vc.hessian(F, parallel=parallel)

        assert np.allclose(xf, F)

        assert xf[0].shape == (3, 3, *F[0].shape[-2:])
        assert xg[0].shape == (3, 3, 3, 3, 1, 1)

        assert yf[0].shape == (3, 3, *F[0].shape[-2:])
        assert yg[0].shape == (3, 3, 3, 3, *F[0].shape[-2:])

        assert Yf[0].shape == (3, *F[0].shape[-2:])
        assert Yg[0].shape == (3, 3, 3, *F[0].shape[-2:])

        assert zf[0].shape == F[0].shape[-2:]
        assert zg[0].shape == (3, 3, *F[0].shape[-2:])
        assert zh[0].shape == (3, 3, 3, 3, *F[0].shape[-2:])


def test_umat():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    from felupe.math import cdya, dya, identity, sym, trace

    def stress(x, mu, lmbda):
        "Evaluate the user-defined linear-elastic stress tensor."

        # extract variables
        F, statevars = x[0], x[-1]

        # user code for linear-elastic stress tensor
        e = sym(F)
        s = 2 * mu * e + lmbda * trace(e) * identity(e)

        # update state variables
        statevars_new = None

        return [s, statevars_new]

    def elasticity(x, mu, lmbda):
        """Evaluate the user-defined fourth-order elasticity tensor according to
        the first Piola-Kirchhoff stress tensor."""

        # extract variables
        F, statevars = x[0], x[-1]

        # user code for fourth-order elasticity tensor
        d = identity(F)
        dsde = 2 * mu * cdya(d, d) + lmbda * dya(d, d)

        return [dsde]

    linear_elastic = fem.Material(stress, elasticity, mu=1, lmbda=2)

    s, statevars_new = linear_elastic.gradient([F, None])
    dsde = linear_elastic.hessian([F, None])


def test_umat_hyperelastic_statevars(close_figs=True):
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    for model, kwargs, incompressible in [
        (
            fem.constitution.ogden_roxburgh,
            dict(r=3, m=1, beta=0, material=fem.neo_hooke, mu=1, nstatevars=1),
            True,
        ),
        (
            models.hyperelastic.morph_representative_directions,
            dict(
                p=[0.011, 0.408, 0.421, 6.85, 0.0056, 5.54, 5.84, 0.117], nstatevars=84
            ),
            True,
        ),
    ]:
        umat = fem.Hyperelastic(model, **kwargs)

        statevars = np.zeros((kwargs["nstatevars"], 8, 1))
        s, statevars_new = umat.gradient([F, statevars])
        dsde = umat.hessian([F, statevars])

        ux = fem.math.linsteps([1, 2, 1], num=10)
        ax = umat.plot(ux=ux, bx=None, ps=None, incompressible=True)

        if close_figs:
            plt.close(ax.get_figure())


def test_umat_hyperelastic(tmp_path, close_figs=True):
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    import tensortrax.math as tm

    def neo_hooke(C, mu=1):
        return mu / 2 * (tm.linalg.det(C) ** (-1 / 3) * tm.trace(C) - 3)

    for model, kwargs, incompressible in [
        (neo_hooke, {"mu": 1}, True),
        (fem.saint_venant_kirchhoff, {"mu": 1, "lmbda": 20.0}, False),
        (fem.saint_venant_kirchhoff, {"mu": 1, "lmbda": 20.0, "k": 0}, False),
        (fem.saint_venant_kirchhoff, {"mu": 1, "lmbda": 20.0, "k": 1}, False),
        (
            fem.saint_venant_kirchhoff_orthotropic,
            {
                "mu": [1, 1, 1],
                "lmbda": [20, 20, 20, 20, 20, 20],
                "r1": np.eye(3)[:, 0],
                "r2": np.eye(3)[:, 1],
                "r3": np.eye(3)[:, 2],
            },
            False,
        ),
        (
            fem.saint_venant_kirchhoff_orthotropic,
            {
                "mu": [1, 1, 1],
                "lmbda": [20, 20, 20, 20, 20, 20],
                "r1": np.eye(3)[:, 0],
                "r2": np.eye(3)[:, 1],
                "r3": None,
                "k": 0,
            },
            False,
        ),
        (
            fem.saint_venant_kirchhoff_orthotropic,
            {
                "mu": [1, 1, 1],
                "lmbda": [20, 20, 20, 20, 20, 20],
                "r1": np.eye(3)[:, 0],
                "r2": np.eye(3)[:, 1],
                "r3": np.eye(3)[:, 2],
                "k": 1,
            },
            False,
        ),
        (fem.neo_hooke, {"mu": 1}, True),
        (fem.mooney_rivlin, {"C10": 0.3, "C01": 0.8}, True),
        (fem.yeoh, {"C10": 0.5, "C20": -0.1, "C30": 0.02}, True),
        (
            fem.third_order_deformation,
            {"C10": 0.5, "C01": 0.1, "C11": 0.01, "C20": -0.1, "C30": 0.02},
            True,
        ),
        (fem.ogden, {"mu": [1, 0.2], "alpha": [1.7, -1.5]}, True),
        (fem.arruda_boyce, {"C1": 1.0, "limit": 3.2}, True),
        (
            fem.extended_tube,
            {"Gc": 0.1867, "Ge": 0.2169, "beta": 0.2, "delta": 0.09693},
            True,
        ),
        (
            fem.van_der_waals,
            {"mu": 1.0, "beta": 0.1, "a": 0.5, "limit": 5.0},
            True,
        ),
        (
            fem.alexander,
            dict(C1=0.117, C2=0.137, C3=0.00690, gamma=0.735, k=0.00015),
            True,
        ),
        (
            fem.anssari_benam_bucchi,
            dict(mu=0.29, N=26.8),
            True,
        ),
        (
            fem.lopez_pamies,
            dict(mu=[0.2699, 0.00001771], alpha=[1.08, 4.40]),
            True,
        ),
        (
            fem.storakers,
            dict(
                mu=[104.869e-3],
                alpha=[7.10874],
                beta=[0.106469],
            ),
            False,
        ),
        (fem.blatz_ko, dict(mu=1.0), False),
    ]:
        umat = fem.Hyperelastic(model, **kwargs)

        statevars = None
        s, statevars_new = umat.gradient([F, statevars])
        dsde = umat.hessian([F, statevars])

    for incompressible in [False, True]:
        ax = umat.plot(incompressible=incompressible)

        if close_figs:
            plt.close(ax.get_figure())

        ax = umat.screenshot(
            filename=tmp_path / "umat.png", incompressible=incompressible
        )

        if close_figs:
            plt.close(ax.get_figure())

    umat = fem.Hyperelastic(fem.neo_hooke, mu=np.nan)

    with pytest.raises(ValueError):
        ax = umat.plot(bx=None, ps=None)

        if close_figs:
            plt.close(ax.get_figure())

    with pytest.raises(ValueError):
        ax = umat.plot(ux=None, bx=None)

        if close_figs:
            plt.close(ax.get_figure())

    with pytest.raises(ValueError):
        ax = umat.plot(ux=None, ps=None)

        if close_figs:
            plt.close(ax.get_figure())


def test_umat_lagrange(tmp_path, close_figs=True):
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    import felupe.constitution.tensortrax as mat

    for model, kwargs, incompressible in [
        (mat.models.lagrange.becker, {"mu": 1, "lmbda": 20.0}, True),
        (mat.models.lagrange.becker, {"mu": 1, "lmbda": 20.0}, False),
    ]:
        umat = mat.Material(model, **kwargs)

        statevars = None
        s, statevars_new = umat.gradient([F, statevars])
        dsde = umat.hessian([F, statevars])

    for incompressible in [False, True]:
        ax = umat.plot(incompressible=incompressible)

        if close_figs:
            plt.close(ax.get_figure())

        ax = umat.screenshot(
            filename=tmp_path / "umat.png", incompressible=incompressible
        )

        if close_figs:
            plt.close(ax.get_figure())


def test_umat_hyperelastic2():
    r, x = pre(sym=False, add_identity=True, add_random=True)
    F = x[0]

    import tensortrax.math as tm

    def neo_hooke(F, mu=1):
        "First Piola-Kirchhoff stress of the Neo-Hookean material formulation."

        C = tm.dot(tm.transpose(F), F)
        Cu = tm.linalg.det(C) ** (-1 / 3) * C

        return mu * F @ tm.special.dev(Cu) @ tm.linalg.inv(C)

    kwargs = {"mu": 1}
    umat = fem.MaterialAD(neo_hooke, **kwargs)

    s, statevars_new = umat.gradient([F, None])
    dsde = umat.hessian([F, None])

    umat = fem.Hyperelastic(fem.constitution.neo_hooke, **kwargs)

    s2, statevars_new = umat.gradient([F, None])
    dsde2 = umat.hessian([F, None])

    assert np.allclose(s, s2)
    assert np.allclose(dsde, dsde2)


def test_umat_viscoelastic(close_figs=True):
    r, x = pre(sym=False, add_identity=True, add_random=True)
    F = x[0]

    import tensortrax.math as tm

    def viscoelastic(C, Cin, mu, eta, dtime):
        "Finite strain viscoelastic material formulation."
        eye = tm.base.eye(C)
        Ci = (
            eye
            + tm.special.from_triu_1d(Cin, like=C)
            + mu / eta * dtime * tm.linalg.det(C) ** (-1 / 3) * C
        )
        Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci
        I1 = tm.linalg.det(C) ** (-1 / 3) * tm.trace(C @ tm.linalg.inv(Ci))

        return mu / 2 * (I1 - 3), tm.special.triu_1d(Ci - eye)

    kwargs = {"mu": 1, "eta": 1, "dtime": 1}
    umat = fem.Hyperelastic(viscoelastic, nstatevars=6, **kwargs)

    statevars = np.zeros((6, *F.shape[-2:]))
    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])

    umat = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, nstatevars=6, **kwargs
    )
    ax = umat.plot(
        ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
        ps=None,
        bx=None,
        incompressible=True,
    )

    if close_figs:
        plt.close(ax.get_figure())

    s2, statevars_new = umat.gradient([F, statevars])
    dsde2 = umat.hessian([F, statevars])

    assert np.allclose(s, s2)
    assert np.allclose(
        fem.math.ddot(s, dsde, mode=(2, 4)),
        fem.math.ddot(s2, dsde2, mode=(2, 4)),
    )
    assert np.allclose(
        fem.math.ddot(dsde, s, mode=(4, 2)),
        fem.math.ddot(dsde2, s2, mode=(4, 2)),
    )


def test_umat_viscoelastic2():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    import tensortrax.math as tm

    def viscoelastic(F, statevars, mu, eta, dtime):
        "Finite strain viscoelastic material formulation."
        C = tm.dot(tm.transpose(F), F)
        Cu = tm.linalg.det(C) ** (-1 / 3) * C
        eye = tm.base.eye(C)
        Cin = eye + tm.special.from_triu_1d(statevars[:6], like=C)
        Ci = Cin + mu / eta * dtime * Cu
        Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci
        S = mu * tm.special.dev(Cu @ tm.linalg.inv(Ci)) @ tm.linalg.inv(C)
        return F @ S, tm.special.triu_1d(Ci - eye)

    kwargs = {"mu": 1, "eta": 1, "dtime": 1}
    umat = fem.MaterialAD(viscoelastic, nstatevars=6, **kwargs)

    statevars = np.zeros((6, *F.shape[-2:]))
    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])

    umat = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, nstatevars=6, **kwargs
    )

    s2, statevars_new = umat.gradient([F, statevars])
    dsde2 = umat.hessian([F, statevars])

    assert np.allclose(s, s2)
    assert np.allclose(
        fem.math.ddot(s, dsde, mode=(2, 4)),
        fem.math.ddot(s2, dsde2, mode=(2, 4)),
    )
    assert np.allclose(
        fem.math.ddot(dsde, s, mode=(4, 2)),
        fem.math.ddot(dsde2, s2, mode=(4, 2)),
    )


def test_umat_strain():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]
    statevars = np.zeros((18 + 0, *F.shape[-2:]))

    for framework in ["small-strain", "total-lagrange", "co-rotational"]:
        umat = fem.MaterialStrain(
            material=fem.constitution.linear_elastic,
            λ=1,
            μ=2,
            framework=framework,
        )

        s, statevars_new = umat.gradient([F, statevars])
        dsde = umat.hessian([F, statevars])

    with pytest.raises(NotImplementedError):
        umat = fem.MaterialStrain(
            material=fem.constitution.linear_elastic,
            λ=1,
            μ=1,
            framework="my-fance-framework",
        )

        s, statevars_new = umat.gradient([F, statevars])
        dsde = umat.hessian([F, statevars])


def test_umat_strain_plasticity():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    statevars = np.ones((18 + 10, *F.shape[-2:]))

    umat = fem.MaterialStrain(
        material=fem.constitution.linear_elastic_plastic_isotropic_hardening,
        λ=1,
        μ=1,
        σy=0,
        K=0.1,
        statevars=(1, (3, 3)),
    )

    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])


def test_umat_strain_viscoelasticity():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    statevars = np.ones((18 + 18, *F.shape[-2:]))

    umat = fem.MaterialStrain(
        material=fem.constitution.linear_elastic_viscoelastic,
        λ=2.0,
        μ=1.0,
        G=[3.0, 15.0],
        τ=[10.0, 100.0],
        Δt=5.0,
        statevars=((2, 3, 3),),
    )

    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])


def test_elpliso():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]
    statevars = np.zeros((28, *F.shape[-2:]))

    umat = fem.LinearElasticPlasticIsotropicHardening(E=3, nu=0.3, sy=1, K=0.1)

    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])


def test_umat_strain_optimize():
    "The parameters of a strain-based material are optimized, but not `tangent`."

    # uniaxial and biaxial tension are required to identify both lamé parameters
    stretches = np.linspace(1, 1.01, 6)
    umat = fem.MaterialStrain(material=fem.linear_elastic, λ=2.0, μ=1.0)
    view = umat.view(incompressible=False, ux=stretches, bx=stretches, ps=None)
    uniaxial, biaxial = view.evaluate()

    # a given flag `tangent` is not a parameter of the material
    initial = fem.MaterialStrain(
        material=fem.linear_elastic, λ=1.0, μ=0.5, tangent=True
    )
    assert list(initial.kwargs.keys()) == ["λ", "μ"]

    # also after an evaluation of the material
    initial.gradient([np.eye(3).reshape(3, 3, 1, 1), np.zeros((18, 1, 1))])
    umat_new, res = initial.optimize(
        ux=[stretches, uniaxial[1]],
        bx=[stretches, biaxial[1]],
        incompressible=False,
    )

    assert res.success
    assert list(umat_new.kwargs.keys()) == ["λ", "μ"]
    assert np.isclose(umat_new.kwargs["λ"], 2.0)
    assert np.isclose(umat_new.kwargs["μ"], 1.0)


def test_composite(close_figs=True):
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    nh = fem.NeoHooke(mu=1.0)
    vol = fem.Volumetric(bulk=2.0)
    umat = nh & vol

    ax = umat.plot()

    if close_figs:
        plt.close(ax.get_figure())

    dWdF, statevars_new = umat.gradient([F, None])
    (d2WdFdF,) = umat.hessian([F, None])


def test_optimize(close_figs=True):
    stretches, stresses = (
        np.array(
            [
                [1.000, 0.00],
                [1.020, 0.26],
                [1.125, 1.37],
                [1.240, 2.30],
                [1.390, 3.23],
                [1.585, 4.16],
                [1.900, 5.10],
                [2.180, 6.00],
                [2.420, 6.90],
                [3.020, 8.80],
                [3.570, 10.7],
                [4.030, 12.5],
                [4.760, 16.2],
                [5.360, 19.9],
                [5.750, 23.6],
                [6.150, 27.4],
                [6.400, 31.0],
                [6.600, 34.8],
                [6.850, 38.5],
                [7.050, 42.1],
                [7.150, 45.8],
                [7.250, 49.6],
                [7.400, 53.3],
                [7.500, 57.0],
                [7.600, 64.4],
            ]
        )
        * np.array([1.0, 0.0980665])
    ).T

    for model in [
        fem.extended_tube,
    ]:
        umat = fem.Hyperelastic(model)
        with pytest.warns():
            umat_new, res = umat.optimize(ux=[stretches, stresses], incompressible=True)

        ux = np.linspace(stretches.min(), stretches.max(), num=200)
        ax = umat_new.plot(incompressible=True, ux=ux, bx=None, ps=None)
        ax.plot(stretches, stresses, "C0x")

        if close_figs:
            plt.close(ax.get_figure())

    # models with scalar and with array-valued material parameters (ogden), each fit
    # takes less than a second
    for model in [
        fem.neo_hooke,
        fem.ogden,
        fem.third_order_deformation,
    ]:
        umat = fem.Hyperelastic(model)
        umat_new, res = umat.optimize(ux=[stretches, stresses], incompressible=True)

        ux = np.linspace(stretches.min(), stretches.max(), num=200)
        ax = umat_new.plot(incompressible=True, ux=ux, bx=None, ps=None)
        ax.plot(stretches, stresses, "C0x")

        if close_figs:
            plt.close(ax.get_figure())

    for model in [
        fem.neo_hooke,
        fem.ogden,
    ]:
        umat = fem.Hyperelastic(model)
        umat_new, res = umat.optimize(
            ux=[stretches, stresses], incompressible=True, relative=True
        )

        ux = np.linspace(stretches.min(), stretches.max(), num=200)
        ax = umat_new.plot(incompressible=True, ux=ux, bx=None, ps=None)
        ax.plot(stretches, stresses, "C0x")

        if close_figs:
            plt.close(ax.get_figure())


def test_lagrange():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    import tensortrax.math as tm

    @fem.total_lagrange
    def neo_hooke_total_lagrange(F, mu=1):
        C = F.T @ F
        S = mu * tm.special.dev(tm.linalg.det(C) ** (-1 / 3) * C) @ tm.linalg.inv(C)
        return S

    @fem.updated_lagrange
    def neo_hooke_updated_lagrange(F, mu=1):
        b = F @ F.T
        J = tm.linalg.det(F)
        σ = mu * tm.special.dev(J ** (-2 / 3) * b) / J
        return σ

    for fun, kwargs in zip(
        [neo_hooke_total_lagrange, neo_hooke_updated_lagrange],
        [{"mu": 1}, {"mu": 1}],
    ):
        umat = fem.MaterialAD(fun, **kwargs)
        nh = fem.NeoHooke(**kwargs)

        P = umat.gradient([F])
        A4 = umat.hessian([F])
        assert np.allclose(P[0], nh.gradient([F])[0])
        assert np.allclose(A4[0], nh.hessian([F])[0])

    for fun, kwargs in zip(
        [fem.becker],
        [{"mu": 1, "lmbda": 2}],
    ):
        umat = fem.MaterialAD(fun, **kwargs)

        P = umat.gradient([F])
        A4 = umat.hessian([F])
        assert not np.isnan(P[0]).any()
        assert not np.isnan(A4[0]).any()


def test_lagrange_statevars():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    import tensortrax.math as tm

    @fem.total_lagrange
    def neo_hooke_total_lagrange(F, statevars, mu=1):
        C = F.T @ F
        S = mu * tm.special.dev(tm.linalg.det(C) ** (-1 / 3) * C) @ tm.linalg.inv(C)
        return S, statevars

    @fem.updated_lagrange
    def neo_hooke_updated_lagrange(F, statevars, mu=1):
        b = F @ F.T
        J = tm.linalg.det(F)
        σ = mu * tm.special.dev(J ** (-2 / 3) * b) / J
        return σ, statevars

    for fun in [neo_hooke_total_lagrange, neo_hooke_updated_lagrange]:
        umat = fem.MaterialAD(fun, mu=1, nstatevars=1)
        nh = fem.NeoHooke(mu=1)

        statevars = np.zeros((1, 8, 1))
        P = umat.gradient([F, statevars])
        A4 = umat.hessian([F, statevars])
        assert np.allclose(P[0], nh.gradient([F])[0])
        assert np.allclose(A4[0], nh.hessian([F])[0])

    p = [0.039, 0.371, 0.174, 2.41, 0.0094, 6.84, 5.65, 0.244]
    umat = fem.MaterialAD(fem.morph, p=p, nstatevars=13)

    statevars = np.zeros((13, 8, 1))
    P = umat.gradient([F, statevars])
    A4 = umat.hessian([F, statevars])

    assert not np.isnan(P[0]).any()
    assert not np.isnan(A4[0]).any()

    p = [0.011, 0.408, 0.421, 6.85, 0.0056, 5.54, 5.84, 0.117]
    umat = fem.MaterialAD(fem.morph_representative_directions, p=p, nstatevars=84)

    statevars = np.zeros((84, 8, 1))
    P = umat.gradient([F, statevars])
    A4 = umat.hessian([F, statevars])

    assert not np.isnan(P[0]).any()
    assert not np.isnan(A4[0]).any()


def test_laplace():
    r, F = pre(sym=False, add_identity=True)

    umat = fem.Laplace()
    W = umat.function(F)
    P = umat.gradient(F)[:-1]
    A = umat.hessian(F)

    assert W[0].shape == F[0].shape[-2:]
    assert P[0].shape == (3, 3, *F[0].shape[-2:])
    assert A[0].shape == (3, 3, 3, 3, 1, 1)


def test_plot_negative_stretches(close_figs=True):
    stretches = np.linspace(-0.5, 1, 16)
    umat = fem.NeoHooke(mu=1.0, bulk=2.0)

    for incompressible in [False, True]:
        with pytest.raises(ValueError):
            ax = umat.plot(ux=stretches, incompressible=incompressible)

            if close_figs:
                plt.close(ax.get_figure())

        with pytest.raises(ValueError):
            ax = umat.plot(bx=stretches, incompressible=incompressible)

            if close_figs:
                plt.close(ax.get_figure())

        with pytest.raises(ValueError):
            ax = umat.plot(ps=stretches, incompressible=incompressible)

            if close_figs:
                plt.close(ax.get_figure())


def test_plot_title_kwargs(close_figs=True):
    "The parameters in the title of the plot may be None, 0-d arrays or strings."

    umats = [
        # a strain-based material
        fem.MaterialStrain(material=fem.linear_elastic, λ=2.0, μ=1.0),
        # the elastic modulus is a 0-d array
        fem.LinearElastic1D(E=1),
    ]

    for umat in umats:
        ax = umat.plot()

        if close_figs:
            plt.close(ax.get_figure())

    umat = fem.NeoHooke(mu=1.0, bulk=2.0)
    umat.kwargs["name"] = "a string"
    umat.kwargs["unset"] = None
    ax = umat.plot()

    assert '"a string"' in ax.get_title()
    assert "unset" not in ax.get_title()

    if close_figs:
        plt.close(ax.get_figure())


def test_volumetric_strain_energy():
    "The volumetric strain energies and their derivatives w.r.t. the volume ratio."

    bulk = 5.0
    J = np.linspace(0.5, 3.0, 11)
    eps = 1e-6

    for kind in ["volume-ratio", "stretch", "log"]:
        U = fem.VolumetricStrainEnergy(bulk=bulk, kind=kind)
        assert U.kind == kind

        # stress-free undeformed configuration with the bulk modulus as stiffness
        assert np.isclose(U(1.0), 0)
        assert np.isclose(U.dUdJ(1.0), 0)
        assert np.isclose(U.d2UdJdJ(1.0), bulk)

        # derivatives by central differences
        dUdJ = (U(J + eps) - U(J - eps)) / (2 * eps)
        d2UdJdJ = (U.dUdJ(J + eps) - U.dUdJ(J - eps)) / (2 * eps)

        assert np.allclose(U.dUdJ(J), dUdJ)
        assert np.allclose(U.d2UdJdJ(J), d2UdJdJ)

    # the kinds are quadratic in the Seth-Hill strains of the volumetric stretch
    stretch = np.cbrt(J)
    strains = {
        "volume-ratio": (stretch**3 - 1) / 3,
        "stretch": stretch - 1,
        "log": np.log(stretch),
    }

    for kind, strain in strains.items():
        U = fem.VolumetricStrainEnergy(bulk=bulk, kind=kind)
        assert np.allclose(U(J), 9 * bulk / 2 * strain**2)

    # maximum of the first derivative in tension
    U = fem.VolumetricStrainEnergy(bulk=bulk, kind="stretch")
    assert np.isclose(U.dUdJ(8.0), 3 * bulk / 4)
    assert np.isclose(U.d2UdJdJ(8.0), 0)

    U = fem.VolumetricStrainEnergy(bulk=bulk, kind="log")
    assert np.isclose(U.dUdJ(np.e), bulk / np.e)
    assert np.isclose(U.d2UdJdJ(np.e), 0)

    # the cubic root is real-valued for (linearized) negative volume ratios
    U = fem.VolumetricStrainEnergy(bulk=bulk, kind="stretch")
    for fun in [U, U.dUdJ, U.d2UdJdJ]:
        assert np.all(np.isfinite(fun(np.array([-0.5, -0.1]))))

    # the default kind and missing or unknown arguments
    assert fem.VolumetricStrainEnergy(bulk=bulk).kind == "volume-ratio"

    with pytest.raises(ValueError, match="bulk"):
        fem.VolumetricStrainEnergy(bulk=None, kind="stretch")

    with pytest.raises(ValueError, match="Unknown"):
        fem.VolumetricStrainEnergy(bulk=bulk, kind="linear")


def test_nh_volumetric():
    "The volumetric parts of NeoHooke and Volumetric are equal to the ones by AD."

    tm = pytest.importorskip("tensortrax.math")

    # the volumetric parts for AD (tensortrax) and for the energy (numpy)
    volumetric_ad = {
        "volume-ratio": lambda J, bulk, math=tm: bulk / 2 * (J - 1) ** 2,
        "stretch": lambda J, bulk, math=tm: 9 * bulk / 2 * (J ** (1 / 3) - 1) ** 2,
        "log": lambda J, bulk, math=tm: bulk / 2 * math.log(J) ** 2,
    }

    def neo_hooke(C, mu, bulk, U):
        J = tm.sqrt(tm.linalg.det(C))
        return mu / 2 * (J ** (-2 / 3) * tm.trace(C) - 3) + U(J, bulk)

    def volumetric(C, bulk, U):
        return U(tm.sqrt(tm.linalg.det(C)), bulk)

    class Log(fem.VolumetricStrainEnergy):
        "A custom volumetric part (equal to the kind log)."

        def __init__(self, bulk):
            self.bulk = bulk

        def __call__(self, J):
            return self.bulk / 2 * np.log(J) ** 2

        def dUdJ(self, J):
            return self.bulk * np.log(J) / J

        def d2UdJdJ(self, J):
            return self.bulk * (1 - np.log(J)) / J**2

    rng = np.random.default_rng(1198)
    F = np.eye(3).reshape(3, 3, 1, 1) + 0.4 * (rng.random((3, 3, 8, 20)) - 0.5)
    assert np.all(fem.math.det(F) > 0)
    x = [F, np.zeros((0, 8, 20))]

    def evaluate(umat, function=True):
        W = umat.function(x)[0] if function else None
        return W, umat.gradient(x)[0], umat.hessian(x)[0]

    def energy(mu, bulk, U):
        "The strain energy density function, evaluated by numpy."
        J = fem.math.det(F)
        W = U(J, bulk, math=np)
        if mu is not None:
            W += mu / 2 * (J ** (-2 / 3) * fem.math.ddot(F, F) - 3)
        return W

    def assert_allclose(results, results_ref):
        for value, value_ref in zip(results, results_ref):
            if value is not None:
                assert np.allclose(value, value_ref)

    mu, bulk = 1.0, 5.0

    for kind, U in volumetric_ad.items():
        umat_ad = fem.Hyperelastic(neo_hooke, mu=mu, bulk=bulk, U=U)
        results = energy(mu, bulk, U), *evaluate(umat_ad, function=False)[1:]

        for umat in [
            fem.NeoHooke(mu=mu, bulk=bulk, volumetric=kind),
            fem.NeoHooke(mu=mu, bulk=bulk, volumetric=kind, parallel=True),
            fem.NeoHooke(
                mu=mu, volumetric=fem.VolumetricStrainEnergy(bulk=bulk, kind=kind)
            ),
        ]:
            assert_allclose(evaluate(umat), results)

        composite = fem.NeoHooke(mu=mu) & fem.Volumetric(bulk=bulk, volumetric=kind)
        assert_allclose(evaluate(composite, function=False), results)

        umat_ad = fem.Hyperelastic(volumetric, bulk=bulk, U=U)
        results = energy(None, bulk, U), *evaluate(umat_ad, function=False)[1:]
        assert_allclose(evaluate(fem.Volumetric(bulk, volumetric=kind)), results)

    # a custom volumetric part
    results = evaluate(fem.NeoHooke(mu=mu, bulk=bulk, volumetric="log"))
    assert_allclose(evaluate(fem.NeoHooke(mu=mu, volumetric=Log(bulk))), results)
    assert_allclose(
        evaluate(fem.Volumetric(volumetric=Log(bulk))),
        evaluate(fem.Volumetric(bulk, volumetric="log")),
    )

    # without the bulk modulus, there is no volumetric part
    umat = fem.NeoHooke(mu=mu, volumetric="log")
    assert umat.volumetric is None
    assert_allclose(evaluate(umat), evaluate(fem.NeoHooke(mu=mu)))

    # the volumetric part of a kind uses the current bulk modulus
    umat.bulk = bulk
    assert umat.volumetric.kind == "log"
    assert_allclose(evaluate(umat), evaluate(fem.NeoHooke(mu, bulk, volumetric="log")))

    # the default volumetric part is unchanged
    for value, value_ref in zip(
        evaluate(fem.NeoHooke(mu=mu, bulk=bulk)),
        evaluate(
            fem.NeoHooke(
                mu=mu, volumetric=fem.VolumetricStrainEnergy(bulk, kind="volume-ratio")
            )
        ),
    ):
        assert np.array_equal(value, value_ref)

    # missing, unknown or ambiguous arguments
    with pytest.raises(ValueError, match="Unknown"):
        fem.NeoHooke(mu=mu, volumetric="linear")

    with pytest.raises(ValueError, match="bulk modulus must be None"):
        fem.NeoHooke(mu=mu, bulk=bulk, volumetric=Log(bulk))

    with pytest.raises(ValueError, match="bulk"):
        fem.Volumetric(volumetric="stretch")

    with pytest.raises(ValueError, match="Unknown"):
        fem.Volumetric(bulk, volumetric="linear")


def test_nearly_incompressible_volumetric():
    "The volumetric part of NearlyIncompressible is given by a kind or an instance."

    rng = np.random.default_rng(1199)
    F = np.eye(3).reshape(3, 3, 1, 1) + 0.2 * (rng.random((3, 3, 8, 20)) - 0.5)
    p = rng.random((1, 8, 20))
    J = 1 + 0.2 * (rng.random((1, 8, 20)) - 0.5)
    x = [F, p, J, np.zeros((0, 8, 20))]

    bulk = 5.0
    umat_ref = fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), bulk=bulk)
    gradient_ref = umat_ref.gradient(x)
    hessian_ref = umat_ref.hessian(x)

    for kind in ["volume-ratio", "stretch", "log"]:
        U = fem.VolumetricStrainEnergy(bulk=bulk, kind=kind)

        for umat in [
            fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), bulk=bulk, volumetric=kind),
            fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), volumetric=U),
        ]:
            gradient = umat.gradient(x)
            hessian = umat.hessian(x)

            # the volumetric part
            assert np.allclose(gradient[2], U.dUdJ(J) - p)
            assert np.allclose(hessian[-1], U.d2UdJdJ(J))

            # the other parts are independent of the volumetric part
            for i in [0, 1]:
                assert np.allclose(gradient[i], gradient_ref[i])
            for i in [0, 1, 4]:
                assert np.allclose(hessian[i], hessian_ref[i])

    # the volumetric part of a kind uses the current bulk modulus
    umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), bulk=bulk, volumetric="log")
    umat.bulk = 2 * bulk
    assert umat.volumetric.kind == "log"
    assert np.allclose(
        umat.gradient(x)[2],
        fem.VolumetricStrainEnergy(bulk=2 * bulk, kind="log").dUdJ(J) - p,
    )

    # missing, unknown or ambiguous arguments
    with pytest.raises(ValueError, match="bulk"):
        fem.NearlyIncompressible(fem.NeoHooke(mu=1.0))

    with pytest.raises(ValueError, match="Unknown"):
        fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), bulk=bulk, volumetric="linear")

    with pytest.raises(ValueError, match="bulk modulus must be None"):
        fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), bulk=bulk, volumetric=U)


if __name__ == "__main__":
    close_figs = True

    test_nh()
    test_nh_out()
    test_linear()
    test_linear_orthotropic()
    test_linear_planestress()
    test_linear_planestrain()
    test_kinematics()
    test_umat()
    with tempfile.TemporaryDirectory() as tmp:
        test_umat_hyperelastic(tmp_path=pathlib.Path(tmp), close_figs=close_figs)
    test_umat_hyperelastic2()
    test_umat_hyperelastic_statevars(close_figs=close_figs)
    with tempfile.TemporaryDirectory() as tmp:
        test_umat_lagrange(tmp_path=pathlib.Path(tmp), close_figs=close_figs)
    test_umat_viscoelastic(close_figs=close_figs)
    test_umat_viscoelastic2()
    test_umat_strain()
    test_umat_strain_plasticity()
    test_elpliso()
    test_umat_strain_optimize()
    test_composite(close_figs=close_figs)
    test_optimize(close_figs=close_figs)
    test_lagrange()
    test_lagrange_statevars()
    test_laplace()
    test_plot_negative_stretches(close_figs=True)
    test_plot_title_kwargs(close_figs=True)
    test_volumetric_strain_energy()
    test_nh_volumetric()
    test_nearly_incompressible_volumetric()
