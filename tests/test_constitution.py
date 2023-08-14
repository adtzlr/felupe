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

import felupe as fe


def pre(sym, add_identity, add_random=False):
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)

    if add_random:
        np.random.seed(55601)
        u.values += np.random.rand(*u.values.shape) / 20

    v = fe.FieldContainer([u])
    return r, v.extract(grad=True, sym=sym, add_identity=add_identity)


def pre_mixed(sym, add_identity):
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    v = fe.Field(r, dim=1)
    z = fe.Field(r, dim=1, values=1)
    w = fe.FieldContainer([u, v, z])
    return r, w.extract(grad=True, sym=sym, add_identity=add_identity), m


def test_nh():
    r, F = pre(sym=False, add_identity=True)

    for parallel in [False, True]:
        for nh in [
            fe.constitution.NeoHooke(mu=1.0, bulk=2.0, parallel=parallel),
            fe.constitution.LinearElasticLargeStrain(E=1.0, nu=0.3, parallel=parallel),
        ]:
            W = nh.function(F)
            P = nh.gradient(F)[:-1]
            A = nh.hessian(F)

            Wx = nh.energy(F)
            Px = nh.stress(F)[:-1]
            Ax = nh.elasticity(F)

            assert np.allclose(W, Wx)
            assert np.allclose(P, Px)
            assert np.allclose(A, Ax)

            assert W[0].shape == F[0].shape[-2:]
            assert P[0].shape == (3, 3, *F[0].shape[-2:])
            assert A[0].shape == (3, 3, 3, 3, *F[0].shape[-2:])

            nh = fe.constitution.NeoHooke(mu=None, bulk=2.0, parallel=parallel)

            W = nh.function(F, mu=2.0)
            P = nh.gradient(F, mu=2.0)[:-1]
            A = nh.hessian(F, mu=2.0)

            assert W[0].shape == F[0].shape[-2:]
            assert P[0].shape == (3, 3, *F[0].shape[-2:])
            assert A[0].shape == (3, 3, 3, 3, *F[0].shape[-2:])

            assert np.allclose(P, 0)

            nh = fe.constitution.NeoHooke(mu=1.0, parallel=parallel)

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
        (fe.constitution.LinearElastic, {}),
        (fe.constitution.LinearElasticTensorNotation, dict(parallel=False)),
        (fe.constitution.LinearElasticTensorNotation, dict(parallel=True)),
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

        le = LinearElastic(E=None, nu=0.3, **kwargs)
        stress = le.gradient(F, E=2.0)[:-1]
        stress = le.gradient(F, E=0.5, nu=0.2)[:-1]
        dsde = le.hessian(F, E=2.0)
        dsde = le.hessian(F, E=3.0)

        assert stress[0].shape == (3, 3, *F[0].shape[-2:])
        assert dsde[0].shape == (3, 3, 3, 3, 1, 1)

        assert np.allclose(stress, 0)

    assert np.allclose(*check_stress)
    assert np.allclose(*check_dsde)


def test_linear_planestress():
    r, F = pre(sym=False, add_identity=True)
    F = [F[0][:2][:, :2]]

    le = fe.constitution.LinearElasticPlaneStress(E=1.0, nu=0.3)

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

    le = fe.constitution.LinearElasticPlaneStress(E=None, nu=0.3)
    stress = le.gradient(F, E=2.0)[:-1]
    stress = le.gradient(F, E=0.5, nu=0.2)[:-1]
    dsde = le.hessian(F, E=2.0)
    dsde = le.hessian(F, E=3.0)

    assert stress[0].shape == (2, 2, *F[0].shape[-2:])
    assert dsde[0].shape == (2, 2, 2, 2, 1, 1)

    assert np.allclose(stress, 0)


def test_linear_planestrain():
    r, F = pre(sym=False, add_identity=True)
    F = [F[0][:2][:, :2]]

    le = fe.constitution.LinearElasticPlaneStrain(E=1.0, nu=0.3)

    stress = le.gradient(F)[:-1]
    dsde = le.hessian(F)
    dsde = le.hessian(F)

    stress_full = le.stress(F)
    strain_full = le.strain(F)

    assert stress_full[0].shape == (3, 3, *F[0].shape[-2:])
    assert strain_full[0].shape == (3, 3, *F[0].shape[-2:])

    le = fe.constitution.LinearElasticPlaneStrain(E=None, nu=None)
    le = fe.constitution.LinearElasticPlaneStrain(E=None, nu=0.3)
    stress = le.gradient(F, E=2.0)[:-1]
    stress = le.gradient(F, E=0.5, nu=0.2)[:-1]
    dsde = le.hessian(F, E=2.0)
    dsde = le.hessian(F, E=3.0)

    assert stress[0].shape == (2, 2, *F[0].shape[-2:])
    assert dsde[0].shape == (2, 2, 2, 2, 1, 1)

    assert np.allclose(stress, 0)


def test_kinematics():
    r, F = pre(sym=False, add_identity=True)

    N = F[0][:, 0]

    for parallel in [False, True]:
        lc = fe.constitution.LineChange(parallel=parallel)
        ac = fe.constitution.AreaChange(parallel=parallel)
        vc = fe.constitution.VolumeChange(parallel=parallel)

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
        assert xg[0].shape == (3, 3, 3, 3, *F[0].shape[-2:])

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

    linear_elastic = fe.Material(stress, elasticity, mu=1, lmbda=2)

    s, statevars_new = linear_elastic.gradient([F, None])
    dsde = linear_elastic.hessian([F, None])


def test_umat_hyperelastic():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    import tensortrax.math as tm

    def neo_hooke(C, mu=1):
        return mu / 2 * (tm.linalg.det(C) ** (-1 / 3) * tm.trace(C) - 3)

    for model, kwargs in [
        (neo_hooke, {"mu": 1}),
        (fe.constitution.saint_venant_kirchhoff, {"mu": 1, "lmbda": 20.0}),
        (fe.constitution.neo_hooke, {"mu": 1}),
        (fe.constitution.mooney_rivlin, {"C10": 0.3, "C01": 0.8}),
        (fe.constitution.yeoh, {"C10": 0.5, "C20": -0.1, "C30": 0.02}),
        (
            fe.constitution.third_order_deformation,
            {"C10": 0.5, "C01": 0.1, "C11": 0.01, "C20": -0.1, "C30": 0.02},
        ),
        (fe.constitution.ogden, {"mu": [1, 0.2], "alpha": [1.7, -1.5]}),
        (fe.constitution.arruda_boyce, {"C1": 1.0, "limit": 3.2}),
        (
            fe.constitution.extended_tube,
            {"Gc": 0.1867, "Ge": 0.2169, "beta": 0.2, "delta": 0.09693},
        ),
        (
            fe.constitution.van_der_waals,
            {"mu": 1.0, "beta": 0.1, "a": 0.5, "limit": 5.0},
        ),
    ]:
        umat = fe.Hyperelastic(model, **kwargs)

        s, statevars_new = umat.gradient([F, None])
        dsde = umat.hessian([F, None])


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
    umat = fe.MaterialAD(neo_hooke, **kwargs)

    s, statevars_new = umat.gradient([F, None])
    dsde = umat.hessian([F, None])

    umat = fe.Hyperelastic(fe.constitution.neo_hooke, **kwargs)

    s2, statevars_new = umat.gradient([F, None])
    dsde2 = umat.hessian([F, None])

    assert np.allclose(s, s2)
    assert np.allclose(dsde, dsde2)


def test_umat_viscoelastic():
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
    umat = fe.Hyperelastic(viscoelastic, nstatevars=6, **kwargs)

    statevars = np.zeros((6, *F.shape[-2:]))
    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])

    umat = fe.Hyperelastic(
        fe.constitution.finite_strain_viscoelastic, nstatevars=6, **kwargs
    )

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
    umat = fe.MaterialAD(viscoelastic, nstatevars=6, **kwargs)

    statevars = np.zeros((6, *F.shape[-2:]))
    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])

    umat = fe.Hyperelastic(
        fe.constitution.finite_strain_viscoelastic, nstatevars=6, **kwargs
    )

    s2, statevars_new = umat.gradient([F, statevars])
    dsde2 = umat.hessian([F, statevars])

    assert np.allclose(s, s2)
    assert np.allclose(dsde, dsde2)


def test_umat_strain():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]
    statevars = np.zeros((18, *F.shape[-2:]))

    umat = fe.MaterialStrain(
        material=fe.constitution.linear_elastic,
        λ=1,
        μ=1,
    )

    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])


def test_umat_strain_plasticity():
    r, x = pre(sym=False, add_identity=True)
    F = x[0]

    statevars = np.ones((28, *F.shape[-2:]))

    umat = fe.MaterialStrain(
        material=fe.constitution.linear_elastic_plastic_isotropic_hardening,
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

    umat = fe.LinearElasticPlasticIsotropicHardening(E=3, nu=0.3, sy=1, K=0.1)

    s, statevars_new = umat.gradient([F, statevars])
    dsde = umat.hessian([F, statevars])


if __name__ == "__main__":
    test_nh()
    test_linear()
    test_linear_planestress()
    test_linear_planestrain()
    test_kinematics()
    test_umat()
    test_umat_hyperelastic()
    test_umat_hyperelastic2()
    test_umat_viscoelastic()
    test_umat_viscoelastic2()
    test_umat_strain()
    test_umat_strain_plasticity()
    test_elpliso()
