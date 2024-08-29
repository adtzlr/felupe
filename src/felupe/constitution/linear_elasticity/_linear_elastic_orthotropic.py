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

from ...math import ddot, identity, transpose
from .._base import ConstitutiveMaterial


class LinearElasticOrthotropic(ConstitutiveMaterial):
    r"""Orthotropic linear-elastic material formulation.

    Parameters
    ----------
    E1 : float
        Young's modulus.
    E2 : float
        Young's modulus.
    E3 : float
        Young's modulus.
    nu12 : float
        Poisson ratio.
    nu23 : float
        Poisson ratio.
    nu13 : float
        Poisson ratio.
    G12 : float
        Shear modulus.
    G23 : float
        Shear modulus.
    G13 : float
        Shear modulus.
    
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
        >>> umat = fem.LinearElasticOrthotropic(
        >>>     E1=1, E2=1, E3=1, nu12=0.3, n23=0.3, nu13=0.3, G12=0.4, G23=0.4, G13=0.4
        >>> )
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

    def __init__(
        self,
        E1=None,
        E2=None,
        E3=None,
        nu12=None,
        nu23=None,
        nu13=None,
        G12=None,
        G23=None,
        G13=None,
    ):
        self.E1 = E1
        self.E2 = E1
        self.E3 = E3

        self.nu12 = nu12
        self.nu23 = nu23
        self.nu13 = nu13

        self.G12 = G12
        self.G23 = G23
        self.G13 = G13

        self.kwargs = {
            "E1": self.E1,
            "E2": self.E2,
            "E3": self.E3,
            "nu12": self.nu12,
            "nu23": self.nu23,
            "nu13": self.nu13,
            "G12": self.G12,
            "G23": self.G23,
            "G13": self.G13,
        }

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def gradient(
        self,
        x,
        E1=None,
        E2=None,
        E3=None,
        nu12=None,
        nu23=None,
        nu13=None,
        G12=None,
        G23=None,
        G13=None,
    ):
        r"""Evaluate the stress tensor (as a function of the deformation gradient).

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`\boldsymbol{F}` (3x3) as first item.
        E1 : float
            Young's modulus.
        E2 : float
            Young's modulus.
        E3 : float
            Young's modulus.
        nu12 : float
            Poisson ratio.
        nu23 : float
            Poisson ratio.
        nu13 : float
            Poisson ratio.
        G12 : float
            Shear modulus.
        G23 : float
            Shear modulus.
        G13 : float
            Shear modulus.

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        F, statevars = x[0], x[-1]

        if E1 is None:
            E1 = self.E1

        if E2 is None:
            E2 = self.E2

        if E3 is None:
            E3 = self.E3

        if nu12 is None:
            nu12 = self.nu12

        if nu23 is None:
            nu23 = self.nu23

        if nu13 is None:
            nu13 = self.nu13

        if G12 is None:
            G12 = self.G12

        if G23 is None:
            G23 = self.G23

        if G13 is None:
            G13 = self.G13

        # convert the deformation gradient to strain
        H = F - identity(F)
        strain = (H + transpose(H)) / 2

        # init stress
        elast = self.hessian(
            x=x,
            E1=E1,
            E2=E2,
            E3=E3,
            nu12=nu12,
            nu23=nu23,
            n13=nu13,
            G12=G12,
            G23=G23,
            G13=G13,
        )[0]

        return [ddot(elast, strain, mode=(4, 2)), statevars]

    def hessian(
        self,
        x=None,
        E1=None,
        E2=None,
        E3=None,
        nu12=None,
        nu23=None,
        nu13=None,
        G12=None,
        G23=None,
        G13=None,
        shape=(1, 1),
        dtype=None,
    ):
        r"""Evaluate the elasticity tensor. The Deformation gradient is only
        used for the shape of the trailing axes.

        Parameters
        ----------
        x : list of ndarray, optional
            List with Deformation gradient :math:`\boldsymbol{F}` (3x3) as first item
            (default is None).
        E1 : float
            Young's modulus.
        E2 : float
            Young's modulus.
        E3 : float
            Young's modulus.
        nu12 : float
            Poisson ratio.
        nu23 : float
            Poisson ratio.
        nu13 : float
            Poisson ratio.
        G12 : float
            Shear modulus.
        G23 : float
            Shear modulus.
        G13 : float
            Shear modulus.
        shape : tuple of int, optional
            Tuple with shape of the trailing axes (default is (1, 1)).

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

        if E1 is None:
            E1 = self.E1

        if E2 is None:
            E2 = self.E2

        if E3 is None:
            E3 = self.E3

        if nu12 is None:
            nu12 = self.nu12

        if nu23 is None:
            nu23 = self.nu23

        if nu13 is None:
            nu13 = self.nu13

        if G12 is None:
            G12 = self.G12

        if G23 is None:
            G23 = self.G23

        if G13 is None:
            G13 = self.G13

        if x is not None:
            dtype = x[0].dtype

        elast = np.zeros((3, 3, 3, 3, *shape), dtype=dtype)

        nu21 = nu12 * E2 / E1
        nu32 = nu23 * E3 / E2
        nu31 = nu13 * E3 / E1

        J = 1 / (1 - nu12 * nu21 - nu23 * nu32 - nu31 * nu13 - 2 * nu21 * nu32 * nu13)

        elast[0, 0, 0, 0] = E1 * (1 - nu23 * nu32) * J
        elast[1, 1, 1, 1] = E2 * (1 - nu13 * nu31) * J
        elast[2, 2, 2, 2] = E3 * (1 - nu12 * nu21) * J

        elast[0, 0, 1, 1] = elast[1, 1, 0, 0] = E1 * (nu21 + nu31 * nu23) * J
        elast[0, 0, 2, 2] = elast[2, 2, 0, 0] = E1 * (nu31 + nu21 * nu32) * J
        elast[1, 1, 2, 2] = elast[2, 2, 1, 1] = E2 * (nu32 + nu12 * nu31) * J

        elast[
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ] = G12

        elast[
            [0, 2, 0, 2],
            [2, 0, 2, 0],
            [0, 0, 2, 2],
            [2, 2, 0, 0],
        ] = G13

        elast[
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 1, 2, 2],
            [2, 2, 1, 1],
        ] = G23

        return [elast]
