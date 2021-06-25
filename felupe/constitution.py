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

from .math import (
    ddot,
    ddot44,
    transpose,
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


class IsochoricProjection:
    def __init__(self, S, C4):
        self.fun_S = S
        self.fun_C4 = C4

    def S(self, C):
        I3 = det(C)
        Sb = I3 ** (-1 / 3) * self.fun_S(C)
        return dot(dev(dot(Sb, C)), inv(C))

    def C4(self, C):
        I3 = det(C)
        eye = identity(C)
        I4 = cdya(eye, eye)
        iC = inv(C)
        P4 = I3 ** (-1 / 3) * (I4 - dya(iC, C))

        Sb = I3 ** (-1 / 3) * self.fun_S(C)

        C4u = self.fun_C4(C)
        if np.allclose(C4b, 0):
            PC4bP = C4u
        else:
            C4b = I3 ** (-2 / 3) * C4u
            PC4bP = ddot44(ddot(P4, C4b), majortranspose(P4))

        SbC = ddot(Sb, C)

        return (
            PC4P
            - 2 / 3 * (dya(Sb, iC) + dya(iC, Sb))
            + 2 / 9 * SbC * dya(iC, iC)
            + 2 / 3 * SbC * cdya(iC, iC)
        )


class TotalLagrangeMaterial:
    def __init__(self, S, C4):
        self.fun_S = S
        self.fun_C4 = C4

    def P(self, F):
        C = dot(transpose(F), F)
        S = self.fun_S(C)
        C4 = self.fun_C4(C)

        return dot(F, S)

    def A(self, F):
        C = dot(transpose(F), F)
        iC = inv(C)
        S = self.fun_S(C)
        C4 = self.fun_C4(C) + cdya_ik(iC, S)
        return np.einsum("iI...,kK...,IJKL...-iJkL...", F, F, C4)


class GeneralizedMixedField:
    def __init__(self, P, A):
        self.fun_P = P
        self.fun_A = A

    def f_u(self, F, p, J):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

        δ_u(Π_int) = ∫_V (∂ψ/∂F + p cof(F)) : δF dV
        """

        return self.Pbb - self.PbbF / 3 * self.iFT + p * self.detF * self.iFT

    def f_p(self, F, p, J):
        """Variation of total potential energy w.r.t pressure.

        δ_p(Π_int) = ∫_V (det(F) - J) δp dV
        """

        return self.detF - J

    def f_J(self, F, p, J):
        """Variation of total potential energy w.r.t volume ratio.

        δ_J(Π_int) = ∫_V (∂U/∂J - p) δJ dV
        """

        return self.PbbF / (3 * J) - p

    def f(self, F, p, J):
        """List of variations of total potential energy w.r.t
        displacements, pressure and volume ratio."""
        self.detF = det(F)
        self.iFT = transpose(inv(F))
        self.Fb = (J / self.detF) ** (1 / 3) * F
        self.Pb = self.fun_P(self.Fb)
        self.Pbb = (J / self.detF) ** (1 / 3) * self.Pb
        self.PbbF = ddot(self.Pbb, F)

        return [self.f_u(F, p, J), self.f_p(F, p, J), self.f_J(F, p, J)]

    def A(self, F, p, J):
        """List of linearized variations of total potential energy w.r.t
        displacements, pressure and volume ratio (these expressions are
        symmetric; A_up = A_pu if derived from a total potential energy
        formulation). List entries have to be arranged as a flattened list
        from the upper triangle blocks:

        [[0 1 2],
         [  3 4],
         [    5]] --> [0 1 2 3 4 5]

        """
        self.detF = det(F)
        self.iFT = transpose(inv(F))
        self.Fb = (J / self.detF) ** (1 / 3) * F
        self.Pbb = (J / self.detF) ** (1 / 3) * self.fun_P(self.Fb)

        self.eye = identity(F)
        self.P4 = cdya_ik(self.eye, self.eye) - 1 / 3 * dya(F, self.iFT)
        self.A4b = self.fun_A(self.Fb)
        self.A4bb = (J / self.detF) ** (2 / 3) * self.A4b

        self.PbbF = ddot(self.Pbb, F)
        self.FA4bb = ddot(F, self.A4bb)
        self.A4bbF = ddot(self.A4bb, F)
        self.FA4bbF = ddot(F, self.A4bbF)

        return [
            self.A_uu(F, p, J),
            self.A_up(F, p, J),
            self.A_uJ(F, p, J),
            self.A_pp(F, p, J),
            self.A_pJ(F, p, J),
            self.A_JJ(F, p, J),
        ]

    def A_uu(self, F, p=None, J=None):
        """Linearization w.r.t. displacements of variation of
        total potential energy w.r.t displacements.

        Δ_u(δ_u(Π_int)) = ∫_V δF : (∂²ψ/(∂F∂F) + p ∂cof(F)/∂F) : ΔF dV

        """

        PbbA4bbF = self.Pbb + self.A4bbF
        PbbFA4bb = self.Pbb + self.FA4bb

        pJ9 = p * self.detF + self.PbbF / 9
        pJ3 = p * self.detF - self.PbbF / 3

        A4 = (
            self.A4bb
            + self.FA4bbF * dya(self.iFT, self.iFT) / 9
            - (dya(PbbA4bbF, self.iFT) + dya(self.iFT, PbbFA4bb)) / 3
            + pJ9 * dya(self.iFT, self.iFT)
            - pJ3 * cdya_il(self.iFT, self.iFT)
        )

        return A4

    def A_pp(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t pressure.

        Δ_p(δ_p(Π_int)) = ∫_V δp 0 Δp dV

        """
        return np.zeros_like(p)

    def A_JJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t volume ratio.

        Δ_J(δ_J(Π_int)) = ∫_V δJ ∂²ψ/(∂J∂J) ΔJ dV

        """

        return (self.FA4bbF - 2 * self.PbbF) / (9 * J ** 2)

    def A_up(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t displacements.

        Δ_p(δ_u(Π_int)) = ∫_V δF : J cof(F) Δp dV

        """

        return self.detF * self.iFT

    def A_uJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t displacements.

        Δ_J(δ_u(Π_int)) = ∫_V δF :  ∂²ψ/(∂F∂J) ΔJ dV

        """

        Ps = self.f_u(F, 0 * p, J)
        return (-self.FA4bbF / 3 * self.iFT + Ps + self.FA4bb) / (3 * J)

    def A_pJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t pressure.

        Δ_J(δ_p(Π_int)) = ∫_V δp (-1) ΔJ dV

        """
        return -np.ones_like(J)
