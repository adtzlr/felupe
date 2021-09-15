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
import casadi as ca

import numpy as np


def mat_W1(F):
    mu, bulk = 1, 5000

    J = ca.det(F)
    C = ca.transpose(F) @ F
    Cu = J ** (-2 / 3) * C

    return mu / 2 * (ca.trace(Cu) - 3) + bulk / 2 * (J - 1) ** 2


def mat_W2(F, p):
    mu, bulk = 1, 5000

    J = ca.det(F)
    C = ca.transpose(F) @ F
    Cu = J ** (-2 / 3) * C

    return mu / 2 * (ca.trace(Cu) - 3) + p * (J - 1) - 1 / (2 * bulk) * p ** 2


def mat_W3(F, p, J):
    mu, bulk = 1, 5000

    detF = ca.det(F)
    C = ca.transpose(F) @ F
    Cu = detF ** (-2 / 3) * C

    return mu / 2 * (ca.trace(Cu) - 3) + bulk / 2 * (J - 1) ** 2 + p * (detF - J)


def mat_W2T(F, P):
    mu, gamma = 1, 5000

    P = ca.reshape(P, (3, 3))

    C = ca.transpose(F) @ F
    E = (C - ca.DM.eye(3)) / 2
    I1 = ca.trace(E)
    I2 = ca.trace(E @ E)

    return mu * I2 + gamma * I1 ** 2 / 2 - ca.trace(ca.transpose(P) @ F)


def mat_W3T(dxdX, P, F):
    mu, gamma = 1, 5000

    P = ca.reshape(P, (3, 3))
    F = ca.reshape(F, (3, 3))

    C = ca.transpose(F) @ F
    E = (C - ca.DM.eye(3)) / 2
    I1 = ca.trace(E)
    I2 = ca.trace(E @ E)

    return mu * I2 + gamma * I1 ** 2 / 2 + ca.trace(ca.transpose(P) @ (dxdX - F))


def mat_straininvariants(invariants):
    """Calculate first (W_a) and second (W_ab) partial derivatives
    of the strain energy density function w.r.t. the invariants."""

    # header section (don't change)
    # --------------------------------
    I1, I2 = invariants

    W_a = np.zeros((2, *I1.shape))
    W_ab = np.zeros((2, 2, *I1.shape))
    # --------------------------------

    # user code
    # --------------------------------
    mu = 1.0
    W_a[0] = mu

    return W_a, W_ab


def mat_invariants(invariants):
    """Calculate first (W_a) and second (W_ab) partial derivatives
    of the strain energy density function w.r.t. the invariants."""

    # header section (don't change)
    # --------------------------------
    I1, I2, I3 = invariants

    W_a = np.zeros((3, *I1.shape))
    W_ab = np.zeros((3, 3, *I1.shape))
    # --------------------------------

    # user code
    # --------------------------------
    mu = 1.0
    W_a[0] = mu

    return W_a, W_ab


def mat_stretches(stretches):
    """Calculate first (W_a) and second (W_ab) partial
    derivatives of the strain energy density function
    w.r.t. the principal stretches."""

    # header section (don't change)
    # -------------------------------------------
    # get shape
    ndim, ngauss, nelems = stretches.shape
    diag = np.arange(ndim), np.arange(ndim)

    W_a = np.zeros((ndim, ngauss, nelems))
    W_ab = np.zeros((ndim, ndim, ngauss, nelems))
    # -------------------------------------------

    # user code
    # -------------------------------------------
    mu = 1.0
    k = 0.7

    W_a = mu * stretches ** (k - 1)
    W_ab[diag] = mu * (k - 1) * stretches ** (k - 2)

    return W_a, W_ab


def pre():
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    p = fe.Field(r, dim=1)
    J = fe.Field(r, dim=1, values=1)

    F = u.extract(grad=True, add_identity=True)

    return F, u, p, J


def preT():
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    P = fe.Field(r, dim=9)
    F = fe.Field(r, dim=9)

    dxdX = u.extract(grad=True, add_identity=True)

    return dxdX, u, P, F


def test_basic():
    F, u, p, J = pre()

    umat = fe.constitution.NeoHooke(1, 3)
    umat.P(F), umat.A(F)

    umat = fe.constitution.NeoHookeCompressible(1, 3)
    umat.P(F), umat.A(F)

    umat = fe.constitution.LineChange()
    umat.fun(F), umat.grad(F)

    umat = fe.constitution.AreaChange()
    umat.fun(F), umat.grad(F)

    umat = fe.constitution.VolumeChange()
    umat.fun(F), umat.grad(F), umat.hessian(F)

    strain = u.grad(sym=True)
    umat = fe.constitution.models.LinearElastic(E=3, nu=0.3)
    umat.elasticity(strain)
    umat.stress(strain)


def test_invariants():
    F, u, p, J = pre()

    umat = fe.constitution.NeoHooke(1, 3)
    umat.P(F), umat.A(F)

    umat = fe.constitution.NeoHookeCompressible(1, 3)
    umat.P(F), umat.A(F)

    strain = u.grad(sym=True)
    umat = fe.constitution.LinearElastic(E=3, nu=0.3)
    umat.elasticity(strain)
    umat.stress(strain)

    mat_I = fe.constitution.InvariantBased(mat_invariants)
    mat_U = fe.constitution.AsIsochoric(mat_I)
    mat_K = fe.constitution.Hydrostatic(bulk=200.0)
    mat = fe.constitution.Composite(mat_U, mat_K)

    umat = fe.constitution.Material(mat)
    umat.P(F), umat.A(F)


def test_stretch():
    F, u, p, J = pre()

    mat_S = fe.constitution.PrincipalStretchBased(mat_stretches)
    mat_U = fe.constitution.AsIsochoric(mat_S)
    mat_K = fe.constitution.Hydrostatic(bulk=200.0)
    mat = fe.constitution.Composite(mat_U, mat_K)

    umat = fe.constitution.Material(mat)
    umat.P(F), umat.A(F)


def test_threefield():
    F, u, p, J = pre()

    nh = fe.constitution.models.NeoHooke(1, 3)

    fields = fe.FieldMixed((u, p, J))
    F, p, J = fields.extract()

    umat = fe.constitution.GeneralizedThreeField(nh.P, nh.A)
    umat.f(F, p, J), umat.A(F, p, J)


def test_straininvariant():
    F, u, p, J = pre()

    mat_E = fe.constitution.StrainInvariantBased(mat_straininvariants)
    mat_U = fe.constitution.AsIsochoric(mat_E)
    mat_K = fe.constitution.Hydrostatic(bulk=200.0)
    mat = fe.constitution.Composite(mat_U, mat_K)

    umat = fe.constitution.Material(mat)
    umat.P(F), umat.A(F)


def test_ad():
    F, u, p, J = pre()

    fields = fe.FieldMixed((u, p, J))
    F, p, J = fields.extract()

    umat = fe.constitution.StrainEnergyDensity(mat_W1)
    umat.f(F), umat.A(F)

    umat = fe.constitution.StrainEnergyDensityTwoField(mat_W2)
    umat.f(F, p), umat.A(F, p)

    umat = fe.constitution.StrainEnergyDensityThreeField(mat_W3)
    umat.f(F, p, J), umat.A(F, p, J)

    dxdX, u, P, F = preT()

    fields = fe.FieldMixed((u, P, F))
    dxdX, P, F = fields.extract()

    umat = fe.constitution.StrainEnergyDensityTwoFieldTensor(mat_W2T)
    umat.f(dxdX, P), umat.A(dxdX, P)

    umat = fe.constitution.StrainEnergyDensityThreeFieldTensor(mat_W3T)
    umat.f(dxdX, P, F), umat.A(dxdX, P, F)


if __name__ == "__main__":  # pragma: no cover
    test_basic()
    test_invariants()
    test_straininvariant()
    test_stretch()
    test_threefield()
    test_ad()
