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

try:
    from einsumt import einsumt
except ModuleNotFoundError:
    from numpy import einsum as einsumt

from ..math import cdya_ik, cdya_il, det, dot, dya, identity, inv, transpose


class LineChange:
    r"""Line Change.

    .. math::

       d\boldsymbol{x} = \boldsymbol{F} d\boldsymbol{X}

    Gradient:

    .. math::

       \frac{\partial \boldsymbol{F}}{\partial \boldsymbol{F}} =
       \boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I}
    """

    def __init__(self, parallel=False):
        self.parallel = parallel

    def function(self, extract):
        """Line change.

        Arguments
        ---------
        extract : list of ndarray
            List of extracted field values with Deformation gradient as first
            item.

        Returns
        -------
        F : ndarray
            Deformation gradient
        """
        return extract

    def gradient(self, extract, parallel=None):
        """Gradient of line change.

        Arguments
        ---------
        extract : list of ndarray
            List of extracted field values with Deformation gradient as first
            item.

        Returns
        -------
        ndarray
            Gradient of line change
        """

        F = extract[0]

        if parallel is None:
            parallel = self.parallel

        Eye = identity(F)
        return [cdya_ik(Eye, Eye, parallel=parallel)]


class AreaChange:
    r"""Area Change.

    .. math::

       d\boldsymbol{a} = J \boldsymbol{F}^{-T} d\boldsymbol{A}

    Gradient:

    .. math::

       \frac{\partial J \boldsymbol{F}^{-T}}{\partial \boldsymbol{F}} =
       J \left( \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T}
       - \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)

    """

    def __init__(self, parallel=False):
        self.parallel = parallel

    def function(self, extract, N=None, parallel=None):
        """Area change.

        Arguments
        ---------
        extract : list of ndarray
            List of extracted field values with Deformation gradient as first
            item.
        N : ndarray or None, optional
            Area normal vector (default is None)

        Returns
        -------
        ndarray
            Cofactor matrix of the deformation gradient
        """

        F = extract[0]
        J = det(F)

        Fs = J * transpose(inv(F, J))

        if parallel is None:
            parallel = self.parallel

        if N is None:
            return [Fs]
        else:
            return [dot(Fs, N, mode=(2, 1), parallel=parallel)]

    def gradient(self, extract, N=None, parallel=None):
        """Gradient of area change.

        Arguments
        ---------
        extract : list of ndarray
            List of extracted field values with Deformation gradient as first
            item.
        N : ndarray or None, optional
            Area normal vector (default is None)

        Returns
        -------
        ndarray
            Gradient of cofactor matrix of the deformation gradient
        """

        F = extract[0]
        J = det(F)

        if parallel is None:
            parallel = self.parallel

        dJdF = self.function([F])[0]
        dFsdF = (
            dya(dJdF, dJdF, parallel=parallel) - cdya_il(dJdF, dJdF, parallel=parallel)
        ) / J

        if parallel:
            einsum = einsumt
        else:
            einsum = np.einsum

        if N is None:
            return [dFsdF]
        else:
            return [einsum("ijkl...,j...->ikl...", dFsdF, N)]


class VolumeChange:
    r"""Volume Change.

    .. math::

       d\boldsymbol{v} = \text{det}(\boldsymbol{F}) d\boldsymbol{V}

    Gradient and hessian (equivalent to gradient of area change):

    .. math::

       \frac{\partial J}{\partial \boldsymbol{F}} &= J \boldsymbol{F}^{-T}

       \frac{\partial^2 J}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}} &=
       J \left( \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} -
       \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)

    """

    def __init__(self, parallel=False):
        self.parallel = parallel

    def function(self, extract):
        """Gradient of volume change.

        Arguments
        ---------
        extract : list of ndarray
            List of extracted field values with Deformation gradient as first
            item.

        Returns
        -------
        J : ndarray
            Determinant of the deformation gradient
        """
        F = extract[0]
        return [det(F)]

    def gradient(self, extract):
        """Gradient of volume change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient

        Returns
        -------
        ndarray
            Gradient of the determinant of the deformation gradient
        """
        F = extract[0]
        J = self.function([F])[0]
        return [J * transpose(inv(F, J))]

    def hessian(self, extract, parallel=None):
        """Hessian of volume change.

        Arguments
        ---------
        extract : list of ndarray
            List of extracted field values with Deformation gradient as first
            item.

        Returns
        -------
        ndarray
            Hessian of the determinant of the deformation gradient
        """

        F = extract[0]

        if parallel is None:
            parallel = self.parallel

        J = self.function([F])[0]
        dJdF = self.gradient([F])[0]
        return [
            (
                dya(dJdF, dJdF, parallel=parallel)
                - cdya_il(dJdF, dJdF, parallel=parallel)
            )
            / J
        ]
