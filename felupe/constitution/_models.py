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
    """Isotropic linear-elastic material formulation.

    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """

    def __init__(self, E=None, nu=None):

        self.E = E
        self.nu = nu

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

    def gradient(self, F, E=None, nu=None):
        """Evaluate the stress tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            Deformation gradient (3x3)
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        # convert to lame constants
        mu, gamma = self._lame_converter(E, nu)

        # convert the deformation gradient to strain
        H = F - identity(F)
        strain = (H + transpose(H)) / 2

        return 2 * mu * strain + gamma * trace(strain) * identity(strain)

    def hessian(self, F, E=None, nu=None):
        """Evaluate the elasticity tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            Deformation gradient (3x3)
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        # convert to lame constants
        mu, gamma = self._lame_converter(E, nu)

        # convert the deformation gradient to strain
        I = identity(F)

        elast = 2 * mu * cdya(I, I) + gamma * dya(I, I)

        return elast

    def _lame_converter(self, E, nu):
        """Convert material parameters to first and second Lamé - constants.

        Arguments
        ---------
        E : float
            Young's modulus
        nu : float
            Poisson ratio

        Returns
        -------
        mu : float
            First Lamé - constant (shear modulus)
        gamma : float
            Second Lamé - constant

        """

        mu = E / (2 * (1 + nu))
        gamma = E * nu / ((1 + nu) * (1 - 2 * nu))

        return mu, gamma


class LinearElasticPlaneStrain:
    """Plane-strain isotropic linear-elastic material formulation.

    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """

    def __init__(self, E, nu):

        self.E = E
        self.nu = nu

        self._umat = LinearElasticPlaneStress(*self._convert(self.E, self.nu))

    def _convert(self, E, nu):
        """Convert Lamé - constants to effective plane strain constants.

        Arguments
        ---------
        E : float
            Young's modulus
        nu : float
            Poisson ratio

        Returns
        -------
        float
            Effective Young's modulus for plane strain formulation
        float
            Effective Poisson ratio for plane strain formulation

        """

        if E is None or nu is None:
            E_eff = None
        else:
            E_eff = E / (1 - nu ** 2)

        if nu is None:
            nu_eff = None
        else:
            nu_eff = nu / (1 - nu)

        return E_eff, nu_eff

    def gradient(self, F, E=None, nu=None):
        """Evaluate the 2d-stress tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            In-plane components of stress tensor (2x2)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        return self._umat.gradient(F, *self._convert(E, nu))

    def hessian(self, F, E=None, nu=None):
        """Evaluate the 2d-elasticity tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (2x2x2x2)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        return self._umat.hessian(F, *self._convert(E, nu))

    def strain(self, F, E=None, nu=None):
        """Evaluate the strain tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        e : ndarray
            Strain tensor (3x3)
        """

        e = np.zeros((3, 3, *F.shape[-2:]))

        for a in range(2):
            e[a, a] = F[a, a] - 1

        e[0, 1] = e[1, 0] = F[0, 1] + F[1, 0]

        return e

    def stress(self, F, E=None, nu=None):
        """ "Evaluate the 3d-stress tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        s = np.pad(self.gradient(F, E=E, nu=nu), ((0, 1), (0, 1), (0, 0), (0, 0)))
        s[2, 2] = nu * (s[0, 0] + s[1, 1])

        return s


class LinearElasticPlaneStress:
    """Plane-stress isotropic linear-elastic material formulation.

    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """

    def __init__(self, E, nu):

        self.E = E
        self.nu = nu

    def gradient(self, F, E=None, nu=None):
        """Evaluate the 2d-stress tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            In-plane components of stress tensor (2x2)

        """
        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        stress = np.zeros((2, 2, *F.shape[-2:]))

        stress[0, 0] = E / (1 - nu ** 2) * ((F[0, 0] - 1) + nu * (F[1, 1] - 1))
        stress[1, 1] = E / (1 - nu ** 2) * ((F[1, 1] - 1) + nu * (F[0, 0] - 1))
        stress[0, 1] = E / (1 - nu ** 2) * (1 - nu) / 2 * (F[0, 1] + F[1, 0])
        stress[1, 0] = stress[0, 1]

        return stress

    def hessian(self, F, E=None, nu=None):
        """Evaluate the elasticity tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's  modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (2x2x2x2)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        elast = np.zeros((2, 2, 2, 2, *F.shape[-2:]))

        for a in range(2):
            elast[a, a, a, a] = E / (1 - nu ** 2)

            for b in range(2):
                if b != a:
                    elast[a, a, b, b] = E / (1 - nu ** 2) * nu

        elast[0, 1, 0, 1] = E / (1 - nu ** 2) * (1 - nu) / 2
        elast[1, 0, 1, 0] = elast[1, 0, 0, 1] = elast[0, 1, 1, 0] = elast[0, 1, 0, 1]

        return elast

    def strain(self, F, E=None, nu=None):
        """Evaluate the strain tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        e : ndarray
            Strain tensor (3x3)
        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        e = np.zeros((3, 3, *F.shape[-2:]))

        for a in range(2):
            e[a, a] = F[a, a] - 1

        e[0, 1] = e[1, 0] = F[0, 1] + F[1, 0]
        e[2, 2] = -nu / (1 - nu) * (F[0, 0] + F[1, 1])

        return e

    def stress(self, F, E=None, nu=None):
        """ "Evaluate the 3d-stress tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray
            In-plane components (2x2) of the deformation gradient
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """
        return np.pad(self.gradient(F, E=E, nu=nu), ((0, 1), (0, 1), (0, 0), (0, 0)))


class NeoHooke:
    r"""Nearly-incompressible isotropic hyperelastic Neo-Hooke material formulation.
    
    The total potential energy is defined as the strain energy density 
    per unit undeformed volume integrated over a volumetric region.
    
    .. math::
       
       \Pi_{int} = \int_V \psi \ dV
    
    The variation leads to the first Piola Kirchhoff stress tensor.
    
    .. math::
    
       \delta_u (\Pi_{int}) &= \int_V \frac{\partial \psi}{\partial \boldsymbol{F}} : \delta \boldsymbol{F} \ dV
    
       \boldsymbol{P} &= \frac{\partial \psi}{\partial \boldsymbol{F}}
    
    A further linearization of the above equation gives the corresponding elasticity tensor.
    
    .. math::
    
       \Delta_u (\delta_u (\Pi_{int})) &= \int_V \delta\boldsymbol{F} : \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}} : \Delta\boldsymbol{F}  \ dV
       
       \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}
    
    Arguments
    ---------
    mu : float
        Shear modulus
    bulk : float
        Bulk modulus
    
    """

    def __init__(self, mu=None, bulk=None):

        self.mu = mu
        self.bulk = bulk

        # aliases for function, gradient and hessian
        self.energy = self.function
        self.stress = self.gradient
        self.elasticity = self.hessian

    def function(self, F, mu=None, bulk=None):
        """Total potential energy.

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
