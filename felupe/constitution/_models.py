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
    r"""Isotropic linear-elastic material formulation.

    ..  math::

        \begin{bmatrix}
            \sigma_{11} \\
            \sigma_{22} \\
            \sigma_{33} \\
            \sigma_{12} \\
            \sigma_{23} \\
            \sigma_{31}
        \end{bmatrix} = \frac{E}{(1+\nu)(1-2\nu)}\begin{bmatrix}
            1-\nu & \nu & \nu & 0 & 0 & 0\\
            \nu & 1-\nu & \nu & 0 & 0 & 0\\
            \nu & \nu & 1-\nu & 0 & 0 & 0\\
            0 & 0 & 0 & \frac{1-2\nu}{2} & 0 & 0 \\
            0 & 0 & 0 & 0 & \frac{1-2\nu}{2} & 0 \\
            0 & 0 & 0 & 0 & 0 & \frac{1-2\nu}{2}
        \end{bmatrix} \cdot \begin{bmatrix}
            \varepsilon_{11} \\
            \varepsilon_{22} \\
            \varepsilon_{33} \\
            2 \varepsilon_{12} \\
            2 \varepsilon_{23} \\
            2 \varepsilon_{31}
        \end{bmatrix}

    with the strain tensor

    ..  math::

        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} + \left( \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} \right)^T \right)


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
        """Evaluate the stress tensor (as a function of the deformation
        gradient).

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

        # convert the deformation gradient to strain
        H = F - identity(F)
        strain = (H + transpose(H)) / 2

        # init stress
        stress = np.zeros_like(strain)

        # normal stress components
        for a, b, c in zip([0, 1, 2], [1, 2, 0], [2, 0, 1]):
            stress[a, a] = (1 - nu) * strain[a, a] + nu * (strain[b, b] + strain[c, c])

        # shear stress components
        for a, b in zip([0, 0, 1], [1, 2, 2]):
            stress[a, b] = stress[b, a] = (1 - 2 * nu) / 2 * 2 * strain[a, b]

        return E / (1 + nu) / (1 - 2 * nu) * stress

    def hessian(self, F=None, E=None, nu=None, shape=(1, 1), region=None):
        """Evaluate the elasticity tensor. The Deformation gradient is only
        used for the shape of the trailing axes.

        Arguments
        ---------
        F : ndarray, optional
            Deformation gradient (3x3) (default is None)
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)
        shape : (int, int), optional (default is (1, 1))
            Tuple with shape of the trailing axes (default is None)
        region : Region, optional
            A numeric region for shape of the trailing axes (default is None)

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

        if F is None:
            if region is not None:
                shape = (len(region.quadrature.points), region.mesh.ncells)
        else:
            shape = F.shape[-2:]

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        elast = np.zeros((3, 3, 3, 3, *shape))

        # diagonal normal components
        for i in range(3):
            elast[i, i, i, i] = 1 - nu

            # off-diagonal normal components
            for j in range(3):
                if j != i:
                    elast[i, i, j, j] = nu

        # diagonal shear components (full-symmetric)
        elast[
            [0, 1, 0, 1, 0, 2, 0, 2, 1, 2, 1, 2],
            [1, 0, 1, 0, 2, 0, 2, 0, 2, 1, 2, 1],
            [0, 0, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2],
            [1, 1, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1],
        ] = (1 - 2 * nu) / 2

        return E / (1 + nu) / (1 - 2 * nu) * elast


class LinearElasticTensorNotation:
    r"""Isotropic linear-elastic material formulation.

    ..  math::

        \boldsymbol{\sigma} &= 2 \mu \ \boldsymbol{\varepsilon} + \gamma \ \text{tr}(\boldsymbol{\varepsilon}) \ \boldsymbol{I}

        \frac{\boldsymbol{\partial \sigma}}{\partial \boldsymbol{\varepsilon}} &= 2 \mu \ \boldsymbol{I} \odot \boldsymbol{I} + \gamma \ \boldsymbol{I} \otimes \boldsymbol{I}

    with the strain tensor

    ..  math::

        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} + \left( \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} \right)^T \right)


    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """

    def __init__(self, E=None, nu=None, parallel=False):

        self.parallel = parallel

        self.E = E
        self.nu = nu

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

    def gradient(self, F=None, E=None, nu=None):
        """Evaluate the stress tensor (as a function of the deformation
        gradient).

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

    def hessian(self, F=None, E=None, nu=None, shape=(1, 1), region=None):
        """Evaluate the elasticity tensor. The Deformation gradient is only
        used for the shape of the trailing axes.

        Arguments
        ---------
        F : ndarray, optional
            Deformation gradient (3x3) (default is None)
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)
        shape : (int, int), optional (default is (1, 1))
            Tuple with shape of the trailing axes (default is None)
        region : Region, optional
            A numeric region for shape of the trailing axes (default is None)

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        if F is None:
            if region is not None:
                shape = (len(region.quadrature.points), region.mesh.ncells)
            I = identity(dim=3, shape=shape)
        else:
            I = identity(F)

        # convert to lame constants
        mu, gamma = self._lame_converter(E, nu)

        elast = 2 * mu * cdya(I, I, parallel=self.parallel) + gamma * dya(
            I, I, parallel=self.parallel
        )

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

        self.elasticity = self.hessian

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

        self.elasticity = self.hessian

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

    def hessian(self, F=None, E=None, nu=None, shape=(1, 1), region=None):
        """Evaluate the elasticity tensor from the deformation gradient.

        Arguments
        ---------
        F : ndarray, optional
            In-plane components (2x2) of the deformation gradient (default is None)
        E : float, optional
            Young's  modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)
        shape : (int, int), optional (default is (1, 1))
            Tuple with shape of the trailing axes (default is None)
        region : Region, optional
            A numeric region for shape of the trailing axes (default is None)

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (2x2x2x2)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        if F is None:
            if region is not None:
                shape = (len(region.quadrature.points), region.mesh.ncells)
        else:
            shape = F.shape[-2:]

        elast = np.zeros((2, 2, 2, 2, *shape))

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
    r"""Nearly-incompressible isotropic hyperelastic Neo-Hooke material
    formulation. The strain energy density function of the Neo-Hookean
    material formulation is a linear function of the trace of the
    isochoric part of the right Cauchy-Green deformation tensor.

    In a nearly-incompressible constitutive framework the strain energy
    density is an additive composition of an isochoric and a volumetric
    part. While the isochoric part is defined on the distortional part of
    the deformation gradient, the volumetric part of the strain
    energy function is defined on the determinant of the deformation
    gradient.

    .. math::

        \psi &= \hat{\psi}(\hat{\boldsymbol{C}}) + U(J)

        \hat\psi(\hat{\boldsymbol{C}}) &= \frac{\mu}{2} (\text{tr}(\hat{\boldsymbol{C}}) - 3)

    with

    .. math::

       J &= \text{det}(\boldsymbol{F})

       \hat{\boldsymbol{F}} &= J^{-1/3} \boldsymbol{F}

       \hat{\boldsymbol{C}} &= \hat{\boldsymbol{F}}^T \hat{\boldsymbol{F}}

    The volumetric part of the strain energy density function is a function
    the volume ratio.

    .. math::

       U(J) = \frac{K}{2} (J - 1)^2

    The first Piola-Kirchhoff stress tensor is evaluated as the gradient
    of the strain energy density function. The hessian of the strain
    energy density function enables the corresponding elasticity tensor.

    .. math::

       \boldsymbol{P} &= \frac{\partial \psi}{\partial \boldsymbol{F}}

       \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}

    A chain rule application leads to the following expression for the stress tensor.
    It is formulated as a sum of the **physical**-deviatoric (not the mathematical deviator!) and the physical-hydrostatic stress tensors.

    .. math::

       \boldsymbol{P} &= \boldsymbol{P}' + \boldsymbol{P}_U

       \boldsymbol{P}' &= \frac{\partial \hat{\psi}}{\partial \hat{\boldsymbol{F}}} : \frac{\partial \hat{\boldsymbol{F}}}{\partial \boldsymbol{F}} = \bar{\boldsymbol{P}} - \frac{1}{3} (\bar{\boldsymbol{P}} : \boldsymbol{F}) \boldsymbol{F}^{-T}

       \boldsymbol{P}_U &= \frac{\partial U(J)}{\partial J} \frac{\partial J}{\partial \boldsymbol{F}} = U'(J) J \boldsymbol{F}^{-T}

    with

    .. math::

       \frac{\partial \hat{\boldsymbol{F}}}{\partial \boldsymbol{F}} &= J^{-1/3} \left( \boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I} - \frac{1}{3} \boldsymbol{F} \otimes \boldsymbol{F}^{-T} \right)

       \frac{\partial J}{\partial \boldsymbol{F}} &= J \boldsymbol{F}^{-T}

       \bar{\boldsymbol{P}} &= J^{-1/3} \frac{\partial \hat{\psi}}{\partial \hat{\boldsymbol{F}}}

    With the above partial derivatives the first Piola-Kirchhoff stress
    tensor of the Neo-Hookean material model takes the following form.

    .. math::

       \boldsymbol{P} = \mu J^{-2/3} \left( \boldsymbol{F} - \frac{1}{3} (\boldsymbol{F} : \boldsymbol{F}) \boldsymbol{F}^{-T} \right) + K (J - 1) J \boldsymbol{F}^{-T}

    Again, a chain rule application leads to an expression for the elasticity tensor.

    .. math::

       \mathbb{A} &= \mathbb{A}' + \mathbb{A}_{U}

       \mathbb{A}' &= \bar{\mathbb{A}} - \frac{1}{3} \left( (\bar{\mathbb{A}} : \boldsymbol{F}) \otimes \boldsymbol{F}^{-T} + \boldsymbol{F}^{-T} \otimes (\boldsymbol{F} : \bar{\mathbb{A}}) \right ) + \frac{1}{9} (\boldsymbol{F} : \bar{\mathbb{A}} : \boldsymbol{F}) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T}

       \mathbb{A}_{U} &= (U''(J) J + U'(J)) J \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - U'(J) J \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T}

    with

    .. math::

       \bar{\mathbb{A}} = J^{-1/3} \frac{\partial^2 \hat\psi}{\partial \hat{\boldsymbol{F}}\ \partial \hat{\boldsymbol{F}}} J^{-1/3}

    With the above partial derivatives the (physical-deviatoric and
    -hydrostatic) parts of the elasticity tensor associated
    to the first Piola-Kirchhoff stress tensor of the Neo-Hookean
    material model takes the following form.

    .. math::

       \mathbb{A} &= \mathbb{A}' + \mathbb{A}_{U}

       \mathbb{A}' &= J^{-2/3} \left(\boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I} - \frac{1}{3} \left( \boldsymbol{F} \otimes \boldsymbol{F}^{-T} + \boldsymbol{F}^{-T} \otimes \boldsymbol{F} \right ) + \frac{1}{9} (\boldsymbol{F} : \boldsymbol{F}) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} \right)

       \mathbb{A}_{U} &= K J \left( (2J - 1) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - (J - 1) \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)


    Arguments
    ---------
    mu : float
        Shear modulus
    bulk : float
        Bulk modulus

    """

    def __init__(self, mu=None, bulk=None, parallel=False):

        self.parallel = parallel

        self.mu = mu
        self.bulk = bulk

        # aliases for function, gradient and hessian
        self.energy = self.function
        self.stress = self.gradient
        self.elasticity = self.hessian

    def function(self, F, mu=None, bulk=None):
        """Strain energy density function per unit undeformed volume of the
        Neo-Hookean material formulation.

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
        """

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        C = dot(transpose(F), F, parallel=self.parallel)

        W = mu / 2 * (J ** (-2 / 3) * trace(C) - 3) + bulk * (J - 1) ** 2 / 2

        return W

    def gradient(self, F, mu=None, bulk=None):
        """Gradient of the strain energy density function per unit
        undeformed volume of the Neo-Hookean material formulation.

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
        """

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))

        Pdev = mu * (F - ddot(F, F, parallel=self.parallel) / 3 * iFT) * J ** (-2 / 3)
        Pvol = bulk * (J - 1) * J * iFT

        return Pdev + Pvol

    def hessian(self, F, mu=None, bulk=None):
        """Hessian of the strain energy density function per unit
        undeformed volume of the Neo-Hookean material formulation.

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
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
                cdya_ik(eye, eye, parallel=self.parallel)
                - 2 / 3 * dya(F, iFT, parallel=self.parallel)
                - 2 / 3 * dya(iFT, F, parallel=self.parallel)
                + 2
                / 9
                * ddot(F, F, parallel=self.parallel)
                * dya(iFT, iFT, parallel=self.parallel)
                + 1
                / 3
                * ddot(F, F, parallel=self.parallel)
                * cdya_il(iFT, iFT, parallel=self.parallel)
            )
            * J ** (-2 / 3)
        )

        p = bulk * (J - 1)
        q = p + bulk * J

        A4_vol = J * (
            q * dya(iFT, iFT, parallel=self.parallel)
            - p * cdya_il(iFT, iFT, parallel=self.parallel)
        )

        return A4_dev + A4_vol
