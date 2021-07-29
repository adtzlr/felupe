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

    def elasticity(self, strain, stress=None):
        I = identity(strain)

        elast = 2 * self.mu * cdya(I, I) + self.gamma * dya(I, I)

        if stress is not None:
            elast_stress = cdya_ik(stress, I)
        else:
            elast_stress = 0

        return elast + elast_stress

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


class NeoHookeCompressible:
    "Compressible Neo-Hooke material."

    def __init__(self, mu, bulk):
        self.mu = mu
        self.bulk = bulk

    def P(self, F):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

        δ_u(Π_int) = ∫_V ∂ψ/∂F : δF dV

        """
        J = det(F)
        iFT = transpose(inv(F, J))

        # return self.mu * (F - iFT) + self.bulk * np.log(J) * iFT
        return self.mu * (F - iFT) + self.bulk * (J - 1) * J * iFT

    def A(self, F):
        """Linearization w.r.t. displacements of variation of
        total potential energy w.r.t displacements.

        Δ_u(δ_u(Π_int)) = ∫_V δF : ∂²ψ/(∂F∂F) : ΔF dV

        """

        J = det(F)
        iFT = transpose(inv(F, J))
        eye = identity(F)

        A4_mu = cdya_ik(eye, eye) + cdya_il(eye, eye)
        # A4_bulk = (dya(iFT, iFT) - np.log(J) * cdya_il(iFT, iFT))

        A4_bulk = (2 * J - 1) * J * dya(iFT, iFT) - (J - 1) * J * cdya_il(iFT, iFT)

        return self.mu * A4_mu + self.bulk * A4_bulk
