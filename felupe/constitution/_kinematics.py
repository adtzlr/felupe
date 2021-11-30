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
    transpose,
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

    def __init__(self):
        pass

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

    def gradient(self, F):
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

        Eye = identity(F)
        return cdya_ik(Eye, Eye)


class AreaChange:
    r"""Area Change.

    .. math::

       d\boldsymbol{a} = J \boldsymbol{F}^{-T} d\boldsymbol{A}

    Gradient:

    .. math::

       \frac{\partial J \boldsymbol{F}^{-T}}{\partial \boldsymbol{F}} = J \left( \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)

    """

    def __init__(self):
        pass

    def function(self, F):
        """Area change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient

        Returns
        -------
        ndarray
            Cofactor matrix of the deformation gradient
        """
        J = det(F)
        return J * transpose(inv(F, J))

    def gradient(self, F):
        """Gradient of area change.

        Arguments
        ---------
        F : ndarray
            Deformation gradient

        Returns
        -------
        ndarray
            Gradient of cofactor matrix of the deformation gradient
        """

        J = det(F)
        dJdF = self.function(F)
        return (dya(dJdF, dJdF) - cdya_il(dJdF, dJdF)) / J


class VolumeChange:
    r"""Volume Change.

    .. math::

       d\boldsymbol{v} = \text{det}(\boldsymbol{F}) d\boldsymbol{V}

    Gradient and hessian (equivalent to gradient of area change):

    .. math::

       \frac{\partial J}{\partial \boldsymbol{F}} &= J \boldsymbol{F}^{-T}

       \frac{\partial^2 J}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}} &= J \left( \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)

    """

    def __init__(self):
        pass

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

    def hessian(self, F):
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

        J = self.function(F)
        dJdF = self.gradient(F)
        return (dya(dJdF, dJdF) - cdya_il(dJdF, dJdF)) / J
