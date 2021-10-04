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
    ddot,
    transpose,
    inv,
    dya,
    cdya_ik,
    cdya_il,
    det,
    identity,
)


class Mixed:
    def __init__(self, gradient, hessian):
        """Three-Field variation for nearly-incompressible materials:
        gradient = 1st Piola-Kirchhoff stress P
        hessian = associated (total) elasticity tensor A4 = dP/dF
        """
        self.fun_P = gradient
        self.fun_A = hessian

    def _gradient_u(self, F, p, J):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

        δ_u(Π_int) = ∫_V (∂ψ/∂F + p cof(F)) : δF dV
        """

        return self.Pbb - self.PbbF / 3 * self.iFT + p * self.detF * self.iFT

    def _gradient_p(self, F, p, J):
        """Variation of total potential energy w.r.t pressure.

        δ_p(Π_int) = ∫_V (det(F) - J) δp dV
        """

        return self.detF - J

    def _gradient_J(self, F, p, J):
        """Variation of total potential energy w.r.t volume ratio.

        δ_J(Π_int) = ∫_V (∂U/∂J - p) δJ dV
        """

        return self.PbbF / (3 * J) - p

    def gradient(self, F, p, J):
        """List of variations of total potential energy w.r.t
        displacements, pressure and volume ratio."""
        self.detF = det(F)
        self.iFT = transpose(inv(F))
        self.Fb = (J / self.detF) ** (1 / 3) * F
        self.Pb = self.fun_P(self.Fb)
        self.Pbb = (J / self.detF) ** (1 / 3) * self.Pb
        self.PbbF = ddot(self.Pbb, F)

        return [
            self._gradient_u(F, p, J),
            self._gradient_p(F, p, J),
            self._gradient_J(F, p, J),
        ]

    def hessian(self, F, p, J):
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
            self._hessian_uu(F, p, J),
            self._hessian_up(F, p, J),
            self._hessian_uJ(F, p, J),
            self._hessian_pp(F, p, J),
            self._hessian_pJ(F, p, J),
            self._hessian_JJ(F, p, J),
        ]

    def _hessian_uu(self, F, p=None, J=None):
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

    def _hessian_pp(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t pressure.

        Δ_p(δ_p(Π_int)) = ∫_V δp 0 Δp dV

        """
        return np.zeros_like(p)

    def _hessian_JJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t volume ratio.

        Δ_J(δ_J(Π_int)) = ∫_V δJ ∂²ψ/(∂J∂J) ΔJ dV

        """

        return (self.FA4bbF - 2 * self.PbbF) / (9 * J ** 2)

    def _hessian_up(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t displacements.

        Δ_p(δ_u(Π_int)) = ∫_V δF : J cof(F) Δp dV

        """

        return self.detF * self.iFT

    def _hessian_uJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t displacements.

        Δ_J(δ_u(Π_int)) = ∫_V δF :  ∂²ψ/(∂F∂J) ΔJ dV

        """

        Ps = self._gradient_u(F, 0 * p, J)
        return (-self.FA4bbF / 3 * self.iFT + Ps + self.FA4bb) / (3 * J)

    def _hessian_pJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t pressure.

        Δ_J(δ_p(Π_int)) = ∫_V δp (-1) ΔJ dV

        """
        return -np.ones_like(J)
