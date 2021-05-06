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

from .math import ddot, transpose, inv, dya, cdya_ik, cdya_il, det, identity


class NeoHooke:
    def __init__(self, mu, bulk=None):
        self.mu = mu

        # automatic bulk modulus
        if bulk is None:
            self.bulk = 5000 * mu
        else:
            self.bulk = bulk

    def P(self, F, p=None, J=None):
        """1st Piola Kirchhoff stress"""
        mu = self.mu
        iFT = transpose(inv(F))
        detF = det(F)

        # one-field formulation
        if p is None:
            p = self.dUdJ(detF)

        Pdev = mu * (F - ddot(F, F) / 3 * iFT) * detF ** (-2 / 3)
        Pvol = p * detF * iFT

        return Pdev + Pvol

    def f_u(self, F, p, J):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).
        
        δ_u(Π_int) = ∫_V (∂ψ/∂F + p cof(F)) : δF dV
        """
        
        return self.P(F, p, J)

    def f_p(self, F, p, J):
        """Variation of total potential energy w.r.t pressure.
        
        δ_p(Π_int) = ∫_V (det(F) - J) δp dV
        """
        
        return det(F) - J

    def f_J(self, F, p, J):
        """Variation of total potential energy w.r.t volume ratio.
        
        δ_J(Π_int) = ∫_V (∂U/∂J - p) δJ dV
        """
        
        return self.dUdJ(J) - p

    def f(self, F, p, J):
        """List of variations of total potential energy w.r.t
        displacements, pressure and volume ratio."""
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

        mu = self.mu

        detF = det(F)
        iFT = transpose(inv(F))
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
            * detF ** (-2 / 3)
        )

        # one-field formulation
        if p is None:
            p = self.dUdJ(detF)
            q = p + self.d2UdJ2(detF) * detF
        else:
            q = p

        A4_vol = detF * (q * dya(iFT, iFT) - p * cdya_il(iFT, iFT))

        return A4_dev + A4_vol

    def A_pp(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t pressure.
        
        Δ_p(δ_p(Π_int)) = ∫_V δp 0 Δp dV
        
        """
        return np.zeros_like(p)

    def A_JJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t volume ratio.
        
        Δ_J(δ_J(Π_int)) = ∫_V δJ ∂²U/(∂J∂J) ΔJ dV
        
        """
        return self.d2UdJ2(J)

    def A_up(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t displacements.
        
        Δ_p(δ_u(Π_int)) = ∫_V δF : J cof(F) Δp dV
        
        """
        detF = det(F)
        iFT = transpose(inv(F))

        return detF * iFT

    def A_uJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t displacements.
        
        Δ_J(δ_u(Π_int)) = ∫_V δF : 0 ΔJ dV
        
        """
        return np.zeros_like(F)

    def A_pJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t pressure.
        
        Δ_J(δ_p(Π_int)) = ∫_V δp (-1) ΔJ dV
        
        """
        return -np.ones_like(J)

    def dUdJ(self, J):
        """Constitutive material formulation for volumetric behaviour."""
        return self.bulk * (J - 1)

    def d2UdJ2(self, J):
        """Linearization of constitutive material formulation
        for volumetric behaviour."""
        return self.bulk * np.ones_like(J)
