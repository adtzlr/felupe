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


def test_umat_hyperelastic(close_figs=True):
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

        ax = umat.screenshot(incompressible=incompressible)

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
        Ci = (
            tm.special.from_triu_1d(Cin, like=C)
            + mu / eta * dtime * tm.linalg.det(C) ** (-1 / 3) * C
        )
        Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci
        I1 = tm.linalg.det(C) ** (-1 / 3) * tm.trace(C @ tm.linalg.inv(Ci))

        return mu / 2 * (I1 - 3), tm.special.triu_1d(Ci)

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
    assert np.allclose(dsde, dsde2)


def test_umat_viscoelastic2():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    import tensortrax.math as tm

    def viscoelastic(F, Cin, mu, eta, dtime):
        "Finite strain viscoelastic material formulation."
        C = tm.dot(tm.transpose(F), F)
        Cu = tm.linalg.det(C) ** (-1 / 3) * C
        Ci = (
            tm.special.from_triu_1d(Cin, like=C)
            + mu / eta * dtime * tm.linalg.det(C) ** (-1 / 3) * C
        )
        Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci
        S = mu * tm.special.dev(Cu @ tm.linalg.inv(Ci)) @ tm.linalg.inv(C)
        return F @ S, tm.special.triu_1d(Ci)

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
    assert np.allclose(dsde, dsde2)


def test_umat_strain():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]
    statevars = np.zeros((18, *F.shape[-2:]))

    umat = fem.MaterialStrain(
        material=fem.constitution.linear_elastic,
        λ=1,
        μ=1,
    )

    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])


def test_umat_strain_plasticity():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    statevars = np.ones((28, *F.shape[-2:]))

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


def test_elpliso():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]
    statevars = np.zeros((28, *F.shape[-2:]))

    umat = fem.LinearElasticPlasticIsotropicHardening(E=3, nu=0.3, sy=1, K=0.1)

    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])


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

    for model in [
        fem.neo_hooke,
        fem.ogden,
        fem.third_order_deformation,
        fem.alexander,
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
        fem.miehe_goektepe_lulei,
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

    for fun in [neo_hooke_total_lagrange, neo_hooke_updated_lagrange]:
        umat = fem.MaterialAD(fun, mu=1)
        nh = fem.NeoHooke(mu=1)

        P = umat.gradient([F])
        A4 = umat.hessian([F])
        assert np.allclose(P[0], nh.gradient([F])[0])
        assert np.allclose(A4[0], nh.hessian([F])[0])


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


if __name__ == "__main__":

    close_figs = True

    test_nh()
    test_linear()
    test_linear_orthotropic()
    test_linear_planestress()
    test_linear_planestrain()
    test_kinematics()
    test_umat()
    test_umat_hyperelastic(close_figs=close_figs)
    test_umat_hyperelastic2()
    test_umat_hyperelastic_statevars(close_figs=close_figs)
    test_umat_viscoelastic(close_figs=close_figs)
    test_umat_viscoelastic2()
    test_umat_strain()
    test_umat_strain_plasticity()
    test_elpliso()
    test_composite(close_figs=close_figs)
    test_optimize(close_figs=close_figs)
    test_lagrange()
    test_lagrange_statevars()
    test_laplace()
    test_plot_negative_stretches(close_figs=True)
