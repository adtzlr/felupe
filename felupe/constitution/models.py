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

from ..math import (
    dot,
    ddot,
    ddot44,
    transpose,
    majortranspose,
    inv,
    dya,
    cdya,
    cdya_ik,
    cdya_il,
    det,
    identity,
    trace,
    dev,
)


class LinearElastic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu, self.gamma = self.lame(E, nu)

    def stress(self, strain):
        return 2 * self.mu * strain + self.gamma * trace(strain) * identity(strain)

    def elasticity(self, strain):
        I = identity(strain)
        return 2 * self.mu * cdya(I, I) + self.gamma * dya(I, I)

    def lame(self, E, nu):
        mu = E / (2 * (1 + nu))
        gamma = E * nu / ((1 + nu) * (1 - 2 * nu))
        return mu, gamma


class NeoHooke:
    "Nearly-incompressible Neo-Hooke material."

    def __init__(self, mu, bulk):
        self.mu = mu
        self.bulk = bulk

    def P(self, F):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

        δ_u(Π_int) = ∫_V ∂ψ/∂F : δF dV

        """

        mu = self.mu
        bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))

        Pdev = mu * (F - ddot(F, F) / 3 * iFT) * J ** (-2 / 3)
        Pvol = bulk * (J - 1) * J * iFT

        return Pdev + Pvol

    def A(self, F):
        """Linearization w.r.t. displacements of variation of
        total potential energy w.r.t displacements.

        Δ_u(δ_u(Π_int)) = ∫_V δF : ∂²ψ/(∂F∂F) : ΔF dV

        """

        mu = self.mu
        bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))
        eye = identity(F)

        A4_dev = (
            mu
            * (
                cdya_ik(eye, eye)
                - 2 / 3 * dya(F, iFT)
                - 2 / 3 * dya(iFT, F)
                + 2 / 9 * ddot(F, F) * dya(iFT, iFT)
                + 1 / 3 * ddot(F, F) * cdya_il(iFT, iFT)
            )
            * J ** (-2 / 3)
        )

        p = bulk * (J - 1)
        q = p + bulk * J

        A4_vol = J * (q * dya(iFT, iFT) - p * cdya_il(iFT, iFT))

        return A4_dev + A4_vol


class MaterialFromTotalLagrange:
    def __init__(self, material_totallagrange, parallel=True):
        self.totallagrange = material_totallagrange
        self.F = 0
        self.parallel = parallel

    def update(self, F):
        if np.all(F == self.F):
            pass
        else:
            F = F
            J = det(F)
            C = dot(transpose(F), F)
            invC = inv(C, J ** 2)

            S = self.totallagrange.S(F, J, C, invC)

            return F, J, C, invC, S

    def P(self, F):
        kinematics = self.update(F)
        if kinematics is not None:
            self.F, self.J, self.C, self.invC, self.S = kinematics

        return dot(F, self.S)

    def A(self, F):
        kinematics = self.update(F)
        if kinematics is not None:
            self.F, self.J, self.C, self.invC, self.S = kinematics

        C4 = self.totallagrange.C4(self.F, self.J, self.C, self.invC) + cdya_ik(
            self.invC, self.S
        )
        if self.parallel:
            return pushforward13(F, F, C4)
        else:
            return np.einsum("iI...,kK...,IJKL...->iJkL...", F, F, C4, optimize=True)


try:
    from numba import jit, prange

    jitargs = {"nopython": True, "nogil": True, "fastmath": True, "parallel": True}

    @jit(**jitargs)
    def pushforward13(F, G, C4):

        ndim, ngauss, nelems = C4.shape[-3:]

        out = np.zeros((ndim, ndim, ndim, ndim, ngauss, nelems))

        for i in prange(ndim):
            for I in prange(ndim):
                for J in prange(ndim):
                    for k in prange(ndim):
                        for K in prange(ndim):
                            for L in prange(ndim):
                                for p in prange(ngauss):
                                    for e in prange(nelems):
                                        out[i, J, k, L, p, e] += (
                                            F[i, I, p, e]
                                            * G[k, K, p, e]
                                            * C4[I, J, K, L, p, e]
                                        )

        return out


except:

    def pushforward13(F, G, C4):
        return np.einsum("iI...,kK...,IJKL...->iJkL...", F, G, C4, optimize=True)
