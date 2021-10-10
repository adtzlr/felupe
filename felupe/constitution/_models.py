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

from ..math import (
    dot,
    ddot,
    transpose,
    inv,
    dya,
    cdya,
    cdya_ik,
    cdya_il,
    det,
    identity,
    trace,
)


class LinearElastic:
    def __init__(self, E=None, nu=None):
        "Linear elastic material with Young's modulus `E` and poisson ratio `nu`."

        self.E = E
        self.nu = nu

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

    def gradient(self, strain, E=None, nu=None):

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        # convert to lame constants
        mu, gamma = self._lame_converter(E, nu)

        return 2 * mu * strain + gamma * trace(strain) * identity(strain)

    def hessian(self, strain, stress=None, E=None, nu=None):

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        # convert to lame constants
        mu, gamma = self._lame_converter(E, nu)

        I = identity(strain)

        elast = 2 * mu * cdya(I, I) + gamma * dya(I, I)

        if stress is not None:
            elast_stress = cdya_ik(stress, I)
        else:
            elast_stress = 0

        return elast + elast_stress

    def _lame_converter(self, E, nu):
        mu = E / (2 * (1 + nu))
        gamma = E * nu / ((1 + nu) * (1 - 2 * nu))
        return mu, gamma


class NeoHooke:
    "Nearly-incompressible isotropic hyperelastic Neo-Hooke material formulation."

    def __init__(self, mu=None, bulk=None):
        "Neo-Hookean material formulation with parameters `mu` and `bulk`."

        self.mu = mu
        self.bulk = bulk

        # aliases for function, gradient and hessian
        self.energy = self.function
        self.stress = self.gradient
        self.elasticity = self.hessian

    def function(self, F, mu=None, bulk=None):
        """Total potential energy

        Π_int = ∫ ψ dV                                              (1)

        --> W = ψ                                                   (2)

        """

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        C = dot(transpose(F), F)

        W = mu / 2 * (J ** (-2 / 3) * trace(C) - 3) + bulk * (J - 1) ** 2 / 2

        return W

    def gradient(self, F, mu=None, bulk=None):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

            δ_u(Π_int) = ∫_V ∂ψ/∂F : δF dV                              (1)

            --> P = ∂ψ/∂F                                               (2)

        """

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))

        Pdev = mu * (F - ddot(F, F) / 3 * iFT) * J ** (-2 / 3)
        Pvol = bulk * (J - 1) * J * iFT

        return Pdev + Pvol

    def hessian(self, F, mu=None, bulk=None):
        """Linearization w.r.t. displacements of variation of
        total potential energy w.r.t displacements.

            Δ_u(δ_u(Π_int)) = ∫_V δF : ∂²ψ/(∂F∂F) : ΔF dV               (1)

            --> A = ∂²ψ/(∂F∂F)                                          (2)

        """

        if mu is None:
            mu = self.mu

        if bulk is None:
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
