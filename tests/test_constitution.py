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
import pytest
import numpy as np
import felupe as fe


def pre(sym, add_identity):
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
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

        nh = fe.constitution.NeoHooke(mu=1.0, bulk=2.0, parallel=parallel)

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

    from felupe.math import dya, cdya, sym, trace, identity

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

    linear_elastic = fe.UserMaterial(stress, elasticity, mu=1, lmbda=2)

    s, statevars_new = linear_elastic.gradient([F, None])
    dsde = linear_elastic.hessian([F, None])


def test_umat_strain():

    r, x = pre(sym=False, add_identity=True)
    F = x[0]
    statevars = np.zeros((18, *F.shape[-2:]))

    umat = fe.UserMaterialStrain(
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

    umat = fe.UserMaterialStrain(
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
    test_umat_strain()
    test_umat_strain_plasticity()
    test_elpliso()
