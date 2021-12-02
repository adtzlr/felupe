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
    return r, u.extract(grad=True, sym=sym, add_identity=add_identity)


def pre_mixed(sym, add_identity):
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    v = fe.Field(r, dim=1)
    z = fe.Field(r, dim=1, values=1)
    w = fe.FieldMixed((u, v, z))
    return r, w.extract(grad=True, sym=sym, add_identity=add_identity)


def test_nh():
    r, F = pre(sym=False, add_identity=True)

    nh = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    W = nh.function(F)
    P = nh.gradient(F)
    A = nh.hessian(F)

    Wx = nh.energy(F)
    Px = nh.stress(F)
    Ax = nh.elasticity(F)

    assert np.allclose(W, Wx)
    assert np.allclose(P, Px)
    assert np.allclose(A, Ax)

    assert W.shape == F.shape[-2:]
    assert P.shape == (3, 3, *F.shape[-2:])
    assert A.shape == (3, 3, 3, 3, *F.shape[-2:])

    nh = fe.constitution.NeoHooke(mu=None, bulk=2.0)

    W = nh.function(F, mu=2.0)
    P = nh.gradient(F, mu=2.0)
    A = nh.hessian(F, mu=2.0)

    assert W.shape == F.shape[-2:]
    assert P.shape == (3, 3, *F.shape[-2:])
    assert A.shape == (3, 3, 3, 3, *F.shape[-2:])

    assert np.allclose(P, 0)


def test_linear():
    r, F = pre(sym=False, add_identity=True)

    check_stress = []
    check_dsde = []

    for LinearElastic in [
        fe.constitution.LinearElastic,
        fe.constitution.LinearElasticTensorNotation,
    ]:

        le = LinearElastic(E=1.0, nu=0.3)

        stress = le.gradient(F)
        dsde = le.hessian(F)
        dsde2 = le.hessian(shape=F.shape[-2:])
        dsde3 = le.hessian(region=r)

        with pytest.raises(TypeError):
            le.hessian()

        check_stress.append(stress)
        check_dsde.append([dsde, dsde2, dsde3])

        assert dsde.shape == dsde2.shape

        le = LinearElastic(E=None, nu=0.3)
        stress = le.gradient(F, E=2.0)
        stress = le.gradient(F, E=0.5, nu=0.2)
        dsde = le.hessian(F, E=2.0)
        dsde = le.hessian(F, E=3.0)

        assert stress.shape == (3, 3, *F.shape[-2:])
        assert dsde.shape == (3, 3, 3, 3, *F.shape[-2:])

        assert np.allclose(stress, 0)

    assert np.allclose(*check_stress)
    assert np.allclose(*check_dsde)


def test_linear_planestress():
    r, F = pre(sym=False, add_identity=True)
    F = F[:2][:, :2]

    le = fe.constitution.LinearElasticPlaneStress(E=1.0, nu=0.3)

    stress = le.gradient(F)
    dsde = le.hessian(F)
    dsde = le.hessian(F)
    dsde2 = le.hessian(shape=F.shape[-2:])
    dsde3 = le.hessian(region=r)

    with pytest.raises(TypeError):
        le.hessian()

    check_dsde = [dsde, dsde2, dsde3]

    assert np.allclose(*check_dsde)

    stress_full = le.stress(F)
    strain_full = le.strain(F)

    assert stress_full.shape == (3, 3, *F.shape[-2:])
    assert strain_full.shape == (3, 3, *F.shape[-2:])

    le = fe.constitution.LinearElasticPlaneStress(E=None, nu=0.3)
    stress = le.gradient(F, E=2.0)
    stress = le.gradient(F, E=0.5, nu=0.2)
    dsde = le.hessian(F, E=2.0)
    dsde = le.hessian(F, E=3.0)

    assert stress.shape == (2, 2, *F.shape[-2:])
    assert dsde.shape == (2, 2, 2, 2, *F.shape[-2:])

    assert np.allclose(stress, 0)


def test_linear_planestrain():
    r, F = pre(sym=False, add_identity=True)
    F = F[:2][:, :2]

    le = fe.constitution.LinearElasticPlaneStrain(E=1.0, nu=0.3)

    stress = le.gradient(F)
    dsde = le.hessian(F)
    dsde = le.hessian(F)

    stress_full = le.stress(F)
    strain_full = le.strain(F)

    assert stress_full.shape == (3, 3, *F.shape[-2:])
    assert strain_full.shape == (3, 3, *F.shape[-2:])

    le = fe.constitution.LinearElasticPlaneStrain(E=None, nu=None)
    le = fe.constitution.LinearElasticPlaneStrain(E=None, nu=0.3)
    stress = le.gradient(F, E=2.0)
    stress = le.gradient(F, E=0.5, nu=0.2)
    dsde = le.hessian(F, E=2.0)
    dsde = le.hessian(F, E=3.0)

    assert stress.shape == (2, 2, *F.shape[-2:])
    assert dsde.shape == (2, 2, 2, 2, *F.shape[-2:])

    assert np.allclose(stress, 0)


def test_kinematics():
    r, F = pre(sym=False, add_identity=True)

    lc = fe.constitution.LineChange()
    ac = fe.constitution.AreaChange()
    vc = fe.constitution.VolumeChange()

    xf = lc.function(F)
    xg = lc.gradient(F)

    yf = ac.function(F)
    yg = ac.gradient(F)

    zf = vc.function(F)
    zg = vc.gradient(F)
    zh = vc.hessian(F)

    assert np.allclose(xf, F)

    assert xf.shape == (3, 3, *F.shape[-2:])
    assert xg.shape == (3, 3, 3, 3, *F.shape[-2:])

    assert yf.shape == (3, 3, *F.shape[-2:])
    assert yg.shape == (3, 3, 3, 3, *F.shape[-2:])

    assert zf.shape == F.shape[-2:]
    assert zg.shape == (3, 3, *F.shape[-2:])
    assert zh.shape == (3, 3, 3, 3, *F.shape[-2:])


def test_wrappers():
    r, F = pre(sym=False, add_identity=True)

    nh = fe.NeoHooke(mu=1.0, bulk=2.0)

    class AsMatadi:
        def __init__(self, material):
            self.material = material

        def function(self, x, threads=1):
            if len(x) == 1:
                return [self.material.function(*x)]
            else:
                return self.material.function(*x)

        def gradient(self, x, threads=1):
            if len(x) == 1:
                return [self.material.gradient(*x)]
            else:
                return self.material.gradient(*x)

        def hessian(self, x, threads=1):
            if len(x) == 1:
                return [self.material.hessian(*x)]
            else:
                return self.material.hessian(*x)

    umat = fe.MatadiMaterial(AsMatadi(nh))

    W = umat.function(F)
    P = umat.gradient(F)
    A = umat.hessian(F)

    assert W.shape == F.shape[-2:]
    assert P.shape == (3, 3, *F.shape[-2:])
    assert A.shape == (3, 3, 3, 3, *F.shape[-2:])

    r, FpJ = pre_mixed(sym=False, add_identity=True)

    umat = fe.MatadiMaterial(AsMatadi(fe.ThreeFieldVariation(nh)))

    P = umat.gradient(*FpJ)
    A = umat.hessian(*FpJ)

    assert P[0].shape == (3, 3, *FpJ[0].shape[-2:])
    assert A[0].shape == (3, 3, 3, 3, *FpJ[0].shape[-2:])


if __name__ == "__main__":
    test_nh()
    test_linear()
    test_linear_planestress()
    test_linear_planestrain()
    test_kinematics()
    test_wrappers()
