# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from ..math import cdya, dya, identity, trace, transpose
from ._base import ConstitutiveMaterial


def lame_converter(E, nu):
    r"""Convert the pair of given material parameters Young's modulus :math:`E` and
    Poisson ratio :math:`\nu` to first and second Lamé - constants :math:`\lambda` and
    :math:`\mu`.

    Notes
    -----

    ..  math::

        \lambda &= \frac{E \nu}{(1 + \nu) (1 - 2 \nu)}

        \mu &= \frac{E}{2 (1 + \nu)}

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    Returns
    -------
    lmbda : float
        First Lamé - constant.
    mu : float
        Second Lamé - constant (shear modulus).
    """

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    return lmbda, mu


class LinearElastic(ConstitutiveMaterial):
    r"""Isotropic linear-elastic material formulation.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.
    
    Notes
    -----
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

        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \frac{\partial \boldsymbol{u}}
        {\partial \boldsymbol{X}} + \left( \frac{\partial \boldsymbol{u}}
        {\partial \boldsymbol{X}} \right)^T \right)
    
    Examples
    --------
    ..  pyvista-plot::
        :context:
        
        >>> import felupe as fem
        >>> 
        >>> umat = fem.LinearElastic(E=1, nu=0.3)
        >>> ax = umat.plot()
    
    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    """

    def __init__(self, E=None, nu=None):
        self.E = E
        self.nu = nu

        self.kwargs = {"E": self.E, "nu": self.nu}

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def gradient(self, x, E=None, nu=None):
        """Evaluate the stress tensor (as a function of the deformation gradient).

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`boldsymbol{F}` (3x3) as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        F, statevars = x[0], x[-1]

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

        return [E / (1 + nu) / (1 - 2 * nu) * stress, statevars]

    def hessian(self, x=None, E=None, nu=None, shape=(1, 1)):
        """Evaluate the elasticity tensor. The Deformation gradient is only
        used for the shape of the trailing axes.

        Parameters
        ----------
        x : list of ndarray, optional
            List with Deformation gradient :math:`boldsymbol{F}` (3x3) as first item
            (default is None).
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).
        shape : tuple of int, optional
            Tuple with shape of the trailing axes (default is (1, 1)).

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

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

        return [E / (1 + nu) / (1 - 2 * nu) * elast]


class LinearElasticTensorNotation(ConstitutiveMaterial):
    r"""Isotropic linear-elastic material formulation.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    Notes
    -----
    ..  math::

        \boldsymbol{\sigma} &= 2 \mu \ \boldsymbol{\varepsilon} + \gamma \
        \text{tr}(\boldsymbol{\varepsilon}) \ \boldsymbol{I}

        \frac{\boldsymbol{\partial \sigma}}{\partial \boldsymbol{\varepsilon}} &=
        2 \mu \ \boldsymbol{I} \odot \boldsymbol{I} + \gamma \ \boldsymbol{I} \otimes
        \boldsymbol{I}

    with the strain tensor

    ..  math::

        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \frac{\partial \boldsymbol{u}}
        {\partial \boldsymbol{X}} + \left( \frac{\partial \boldsymbol{u}}
        {\partial \boldsymbol{X}} \right)^T \right)

    Examples
    --------
    ..  pyvista-plot::
        :context:
        
        >>> import felupe as fem
        >>> 
        >>> umat = fem.constitution.LinearElasticTensorNotation(E=1, nu=0.3)
        >>> ax = umat.plot()
    
    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    """

    def __init__(self, E=None, nu=None, parallel=False):
        self.parallel = parallel

        self.E = E
        self.nu = nu

        self.kwargs = {"E": self.E, "nu": self.nu}

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def gradient(self, x, E=None, nu=None):
        """Evaluate the stress tensor (as a function of the deformation
        gradient).

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`boldsymbol{F}` (3x3) as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        F, statevars = x[0], x[-1]

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        # convert to lame constants
        gamma, mu = lame_converter(E, nu)

        # convert the deformation gradient to strain
        H = F - identity(F)
        strain = (H + transpose(H)) / 2

        return [2 * mu * strain + gamma * trace(strain) * identity(strain), statevars]

    def hessian(self, x=None, E=None, nu=None, shape=(1, 1)):
        """Evaluate the elasticity tensor. The Deformation gradient is only
        used for the shape of the trailing axes.

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient  :math:`boldsymbol{F}` (3x3) as first item.
            (default is None)
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).
        shape : (int, ...), optional
            Tuple with shape of the trailing axes (default is (1, 1))

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        eye = identity(dim=3, shape=shape)

        # convert to lame constants
        gamma, mu = lame_converter(E, nu)

        elast = 2 * mu * cdya(eye, eye, parallel=self.parallel) + gamma * dya(
            eye, eye, parallel=self.parallel
        )

        return [elast]


class LinearElasticPlaneStrain:
    """Plane-strain isotropic linear-elastic material formulation.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

        self.kwargs = {"E": self.E, "nu": self.nu}

        self._umat = LinearElasticPlaneStress(*self._convert(self.E, self.nu))

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(2), np.zeros(0)]

        self.elasticity = self.hessian

    def _convert(self, E, nu):
        """Convert Lamé - constants to effective plane strain constants.

        Parameters
        ----------
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
            E_eff = E / (1 - nu**2)

        if nu is None:
            nu_eff = None
        else:
            nu_eff = nu / (1 - nu)

        return E_eff, nu_eff

    def gradient(self, x, E=None, nu=None):
        """Evaluate the 2d-stress tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        ndarray
            In-plane components of stress tensor (2x2)

        """
        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        return self._umat.gradient(x, *self._convert(E, nu))

    def hessian(self, x, E=None, nu=None):
        """Evaluate the 2d-elasticity tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (2x2x2x2)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        return self._umat.hessian(x, *self._convert(E, nu))

    def strain(self, x, E=None, nu=None):
        """Evaluate the strain tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        e : ndarray
            Strain tensor (3x3)
        """

        F = x[0]

        e = np.zeros((3, 3, *F.shape[-2:]))

        for a in range(2):
            e[a, a] = F[a, a] - 1

        e[0, 1] = e[1, 0] = F[0, 1] + F[1, 0]

        return [e]

    def stress(self, x, E=None, nu=None):
        """ "Evaluate the 3d-stress tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        F = x[0]

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        s = np.pad(self.gradient(F, E=E, nu=nu)[0], ((0, 1), (0, 1), (0, 0), (0, 0)))
        s[2, 2] = nu * (s[0, 0] + s[1, 1])

        return [s]


class LinearElasticPlaneStress:
    """Plane-stress isotropic linear-elastic material formulation.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

        self.kwargs = {"E": self.E, "nu": self.nu}

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(2), np.zeros(0)]

        self.elasticity = self.hessian

    def gradient(self, x, E=None, nu=None):
        """Evaluate the 2d-stress tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        ndarray
            In-plane components of stress tensor (2x2)

        """

        F, statevars = x[0], x[-1]

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        stress = np.zeros((2, 2, *F.shape[-2:]))

        stress[0, 0] = E / (1 - nu**2) * ((F[0, 0] - 1) + nu * (F[1, 1] - 1))
        stress[1, 1] = E / (1 - nu**2) * ((F[1, 1] - 1) + nu * (F[0, 0] - 1))
        stress[0, 1] = E / (1 - nu**2) * (1 - nu) / 2 * (F[0, 1] + F[1, 0])
        stress[1, 0] = stress[0, 1]

        return [stress, statevars]

    def hessian(self, x=None, E=None, nu=None, shape=(1, 1)):
        """Evaluate the elasticity tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray, optional
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item (default is None)-
        E : float, optional
            Young's  modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).
        shape : tuple of int, optional
            Tuple with shape of the trailing axes (default is (1, 1)).

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (2x2x2x2).

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        elast = np.zeros((2, 2, 2, 2, *shape))

        for a in range(2):
            elast[a, a, a, a] = E / (1 - nu**2)

            for b in range(2):
                if b != a:
                    elast[a, a, b, b] = E / (1 - nu**2) * nu

        elast[0, 1, 0, 1] = E / (1 - nu**2) * (1 - nu) / 2
        elast[1, 0, 1, 0] = elast[1, 0, 0, 1] = elast[0, 1, 1, 0] = elast[0, 1, 0, 1]

        return [elast]

    def strain(self, x, E=None, nu=None):
        """Evaluate the strain tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        e : ndarray
            Strain tensor (3x3)
        """

        F = x[0]

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        e = np.zeros((3, 3, *F.shape[-2:]))

        for a in range(2):
            e[a, a] = F[a, a] - 1

        e[0, 1] = e[1, 0] = F[0, 1] + F[1, 0]
        e[2, 2] = -nu / (1 - nu) * (F[0, 0] + F[1, 1])

        return [e]

    def stress(self, x, E=None, nu=None):
        """ "Evaluate the 3d-stress tensor from the deformation gradient.

        Parameters
        ----------
        x : list of ndarray
            List with In-plane components (2x2) of the Deformation gradient
            :math:`boldsymbol{F}` as first item.
        E : float, optional
            Young's modulus (default is None).
        nu : float, optional
            Poisson ratio (default is None).

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        F = x[0]

        return [
            np.pad(self.gradient(F, E=E, nu=nu)[0], ((0, 1), (0, 1), (0, 0), (0, 0)))
        ]
