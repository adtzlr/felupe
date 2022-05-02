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

try:
    from einsumt import einsumt
except:
    print("ImportWarning: Module `einsumt` not found. Fall back to `np.einsum()`.")
    from numpy import einsum as einsumt

from ..math import (
    transpose,
    dot,
    inv,
    dya,
    cdya_ik,
    cdya_il,
    det,
    identity,
)


class LineChange:
    r"""Line Change.

    .. math::

       d\boldsymbol{x} = \boldsymbol{F} d\boldsymbol{X}

    Gradient:

    .. math::

       \frac{\partial \boldsymbol{F}}{\partial \boldsymbol{F}} = \boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I}
    """

    def __init__(self, parallel=False):
        self.parallel = parallel

    def function(self, F):
        """Line change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient

        Returns
        -------
        F : ndarray
            Deformation gradient
        """
        return F

    def gradient(self, F, parallel=None):
        """Gradient of line change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient

        Returns
        -------
        ndarray
            Gradient of line change
        """

        if parallel is None:
            parallel = self.parallel

        Eye = identity(F)
        return cdya_ik(Eye, Eye, parallel=parallel)


class AreaChange:
    r"""Area Change.

    .. math::

       d\boldsymbol{a} = J \boldsymbol{F}^{-T} d\boldsymbol{A}

    Gradient:

    .. math::

       \frac{\partial J \boldsymbol{F}^{-T}}{\partial \boldsymbol{F}} = J \left( \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)

    """

    def __init__(self, parallel=False):
        self.parallel = parallel

    def function(self, F, N=None, parallel=None):
        """Area change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        N : ndarray or None, optional
            Area normal vector (default is None)

        Returns
        -------
        ndarray
            Cofactor matrix of the deformation gradient
        """
        J = det(F)

        Fs = J * transpose(inv(F, J))

        if parallel is None:
            parallel = self.parallel

        if N is None:
            return Fs
        else:
            return dot(Fs, N, parallel=parallel)

    def gradient(self, F, N=None, parallel=None):
        """Gradient of area change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        N : ndarray or None, optional
            Area normal vector (default is None)

        Returns
        -------
        ndarray
            Gradient of cofactor matrix of the deformation gradient
        """

        J = det(F)

        if parallel is None:
            parallel = self.parallel

        dJdF = self.function(F)
        dFsdF = (
            dya(dJdF, dJdF, parallel=parallel) - cdya_il(dJdF, dJdF, parallel=parallel)
        ) / J

        if parallel:
            einsum = einsumt
        else:
            einsum = np.einsum

        if N is None:
            return dFsdF
        else:
            return einsum("ijkl...,j...->ikl...", dFsdF, N)


class VolumeChange:
    r"""Volume Change.

    .. math::

       d\boldsymbol{v} = \text{det}(\boldsymbol{F}) d\boldsymbol{V}

    Gradient and hessian (equivalent to gradient of area change):

    .. math::

       \frac{\partial J}{\partial \boldsymbol{F}} &= J \boldsymbol{F}^{-T}

       \frac{\partial^2 J}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}} &= J \left( \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)

    """

    def __init__(self, parallel=False):
        self.parallel = parallel

    def function(self, F):
        """Gradient of volume change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient

        Returns
        -------
        J : ndarray
            Determinant of the deformation gradient
        """
        return det(F)

    def gradient(self, F):
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

        J = self.function(F)
        return J * transpose(inv(F, J))

    def hessian(self, F, parallel=None):
        """Hessian of volume change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient

        Returns
        -------
        ndarray
            Hessian of the determinant of the deformation gradient
        """

        if parallel is None:
            parallel = self.parallel

        J = self.function(F)
        dJdF = self.gradient(F)
        return (
            dya(dJdF, dJdF, parallel=parallel) - cdya_il(dJdF, dJdF, parallel=parallel)
        ) / J
