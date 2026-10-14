# SPDX-License-Identifier: GPL-3.0-or-later

from functools import partial

import numpy as np

from ..math import cdya_ik, cdya_il, det, dot, dya, identity, inv, transpose
from ..math._einsum import einsum_chunks


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

        einsum = partial(einsum_chunks, parallel=True) if parallel else np.einsum

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
