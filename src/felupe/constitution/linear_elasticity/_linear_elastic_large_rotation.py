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

from ...math import cdya, dot, dya, identity, trace, transpose
from .._base import ConstitutiveMaterial
from ._lame_converter import lame_converter


class LinearElasticLargeRotation(ConstitutiveMaterial):
    r"""Isotropic linear-elastic material formulation for large rotations.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.
    
    Notes
    -----
    The stress-strain relation of the linear-elastic material formulation is given in
    Eq. :eq:`linear-elastic`
    
    ..  math::
        :label: linear-elastic

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

    with the small-strain tensor from Eq. :eq:`linear-elastic-strain`.

    ..  math::
        :label: linear-elastic-strain

        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \frac{\partial \boldsymbol{u}}
        {\partial \boldsymbol{X}} + \left( \frac{\partial \boldsymbol{u}}
        {\partial \boldsymbol{X}} \right)^T \right)
    
    ..  warning::
        
        This material formulation must not be used in analyses where large rotations,
        large displacements or large strains occur. In this case, consider using a
        :class:`~felupe.Hyperelastic` material formulation instead.
        :class:`~felupe.LinearElasticLargeStrain` is based on a compressible version
        of the Neo-Hookean material formulation and is safe to use for large rotations,
        large displacements and large strains.
    
    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>> 
        >>> umat = fem.LinearElastic(E=1, nu=0.3)
        >>> ax = umat.plot()
    
    See Also
    --------
    felupe.LinearElasticLargeStrain : Linear-elastic material formulation suitable for
        large-strain analyses based on the compressible Neo-Hookean material
        formulation.
    felupe.Hyperelastic : A hyperelastic material definition with a given function for
        the strain energy density function per unit undeformed volume with automatic
        differentiation.
    """

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

        self.kwargs = {"E": self.E, "nu": self.nu}

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def gradient(self, x):
        r"""Evaluate the stress tensor (as a function of the deformation gradient).

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`\boldsymbol{F}` (3x3) as first item.

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        E = self.E
        nu = self.nu

        # convert the deformation gradient to strain
        F, statevars = x[0], x[-1]
        W, Sigma, Vt = np.linalg.svd(F.T, full_matrices=False, hermitian=True)
        R = transpose(dot(Vt.T, W.T))
        RtHR = dot(dot(transpose(R), F - identity(F)), R)
        strain = (RtHR + transpose(RtHR)) / 2

        # init stress
        stress = np.zeros_like(strain)

        # normal stress components
        for a, b, c in zip([0, 1, 2], [1, 2, 0], [2, 0, 1]):
            stress[a, a] = (1 - nu) * strain[a, a] + nu * (strain[b, b] + strain[c, c])

        # shear stress components
        for a, b in zip([0, 0, 1], [1, 2, 2]):
            stress[a, b] = stress[b, a] = (1 - 2 * nu) / 2 * 2 * strain[a, b]

        stress *= E / (1 + nu) / (1 - 2 * nu)
        stress = np.einsum(
            "iI...,jJ...,IJ...->ij...", R, R, stress, optimize=True, out=stress
        )

        return [stress, statevars]

    def hessian(self, x):
        r"""Evaluate the elasticity tensor.

        Parameters
        ----------
        x : list of ndarray, optional
            List with Deformation gradient :math:`\boldsymbol{F}` (3x3) as first item
            (default is None).
        shape : tuple of int, optional
            Tuple with shape of the trailing axes (default is (1, 1)).

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

        E = self.E
        nu = self.nu

        # convert the deformation gradient to strain
        F = x[0]
        W, Sigma, Vt = np.linalg.svd(F.T, full_matrices=False, hermitian=True)
        R = transpose(dot(Vt.T, W.T))

        elast = np.zeros((3, 3, *F.shape), dtype=F.dtype)

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

        elast *= E / (1 + nu) / (1 - 2 * nu)
        elast = np.einsum("iI...,kK...,IJKL...->iJkL...", R, R, elast, optimize=True)

        return [elast]
