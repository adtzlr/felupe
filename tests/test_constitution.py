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

import numpy as np


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

    F = fe.tools.defgrad(u)

    return F, u, p, J


def test_basic():
    F, u, p, J = pre()

    umat = fe.constitution.models.NeoHooke(1, 3)
    umat.P(F), umat.A(F)

    umat = fe.constitution.models.NeoHookeCompressible(1, 3)
    umat.P(F), umat.A(F)

    strain = fe.tools.strain(u)
    umat = fe.constitution.models.LinearElastic(E=3, nu=0.3)
    umat.elasticity(strain)
    umat.stress(strain)


def test_invariants():
    F, u, p, J = pre()

    umat = fe.constitution.models.NeoHooke(1, 3)
    umat.P(F), umat.A(F)

    umat = fe.constitution.models.NeoHookeCompressible(1, 3)
    umat.P(F), umat.A(F)

    strain = fe.tools.strain(u)
    umat = fe.constitution.models.LinearElastic(E=3, nu=0.3)
    umat.elasticity(strain)
    umat.stress(strain)

    mat_I = fe.constitution.InvariantBased(mat_invariants)
    mat_U = fe.constitution.AsIsochoric(mat_I)
    mat_K = fe.constitution.Hydrostatic(bulk=200.0)
    mat = fe.constitution.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat)
    umat.P(F), umat.A(F)

    mat_I = fe.constitution.df0da0.InvariantBased(mat_invariants)
    mat_U = fe.constitution.df0da0.AsIsochoric(mat_I)
    mat_K = fe.constitution.df0da0.Hydrostatic(bulk=200.0)
    mat = fe.constitution.df0da0.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat)
    umat.P(F), umat.A(F)

    mat_I = fe.constitution.df_da_.InvariantBased(mat_invariants)
    mat_U = fe.constitution.df_da_.AsIsochoric(mat_I)
    mat_K = fe.constitution.df_da_.Hydrostatic(bulk=200.0)
    mat = fe.constitution.df_da_.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat)
    umat.P(F), umat.A(F)


def test_stretch():
    F, u, p, J = pre()

    mat_S = fe.constitution.PrincipalStretchBased(mat_stretches)
    mat_U = fe.constitution.AsIsochoric(mat_S)
    mat_K = fe.constitution.Hydrostatic(bulk=200.0)
    mat = fe.constitution.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat)
    umat.P(F), umat.A(F)

    mat_S = fe.constitution.df0da0.PrincipalStretchBased(mat_invariants)
    mat_U = fe.constitution.df0da0.AsIsochoric(mat_S)
    mat_K = fe.constitution.df0da0.Hydrostatic(bulk=200.0)
    mat = fe.constitution.df0da0.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat, parallel=True)
    umat = fe.constitution.MaterialFrom(mat, parallel=False)
    umat.P(F), umat.A(F)

    mat_S = fe.constitution.df_da_.PrincipalStretchBased(mat_invariants)
    mat_U = fe.constitution.df_da_.AsIsochoric(mat_S)
    mat_K = fe.constitution.df_da_.Hydrostatic(bulk=200.0)
    mat = fe.constitution.df_da_.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat, parallel=True)
    umat = fe.constitution.MaterialFrom(mat, parallel=False)
    umat.P(F), umat.A(F)


def test_threefield():
    F, u, p, J = pre()

    umat = fe.constitution.models.NeoHooke(1, 3)

    F, p, J = fe.tools.FpJ((u, p, J))

    vmat = fe.constitution.variation.upJ(umat.P, umat.A)
    vmat.f(F, p, J), vmat.A(F, p, J)


def test_straininvariant():
    F, u, p, J = pre()

    mat_E = fe.constitution.StrainInvariantBased(mat_straininvariants)
    mat_U = fe.constitution.AsIsochoric(mat_E)
    mat_K = fe.constitution.Hydrostatic(bulk=200.0)
    mat = fe.constitution.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat)
    umat.P(F), umat.A(F)

    mat_E = fe.constitution.df0da0.StrainInvariantBased(mat_straininvariants)
    mat_U = fe.constitution.df0da0.AsIsochoric(mat_E)
    mat_K = fe.constitution.df0da0.Hydrostatic(bulk=200.0)
    mat = fe.constitution.df0da0.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat)
    umat.P(F), umat.A(F)

    mat_E = fe.constitution.df_da_.StrainInvariantBased(mat_straininvariants)
    mat_U = fe.constitution.df_da_.AsIsochoric(mat_E)
    mat_K = fe.constitution.df_da_.Hydrostatic(bulk=200.0)
    mat = fe.constitution.df_da_.Composite(mat_U, mat_K)

    umat = fe.constitution.MaterialFrom(mat)
    umat.P(F), umat.A(F)


if __name__ == "__main__":  # pragma: no cover
    test_basic()
    test_invariants()
    test_straininvariant()
    test_stretch()
    test_threefield()
