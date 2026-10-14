# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from .._base import ConstitutiveMaterial


class LinearElastic1D(ConstitutiveMaterial):
    r"""Isotropic one-dimensional linear-elastic material formulation.

    Parameters
    ----------
    E : float
        Young's modulus.

    Notes
    -----
    The stress-stretch relation of the linear-elastic material formulation is given in
    Eq. :eq:`linear-elastic-1d`

    ..  math::
        :label: linear-elastic-1d

        \sigma = E\ \left(\lambda -  1 \right)

    with the stretch from Eq. :eq:`linear-elastic-strain-1d`.

    ..  math::
        :label: linear-elastic-strain-1d

        \lambda = \frac{l}{L}


    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>>
        >>> umat = fem.LinearElastic1D(E=1)
        >>> ax = umat.plot()

    """

    def __init__(self, E):
        self.E = np.array(E)

        self.kwargs = {"E": self.E}

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.ones(1), np.zeros(0)]

    def gradient(self, x, out=None):
        r"""Evaluate the stress (as a function of the stretch).

        Parameters
        ----------
        x : list of ndarray
            List with the stretch :math:`\lambda` as first item.
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
            Not implemented.

        Returns
        -------
        ndarray
            Stress

        """

        λ, statevars = x[0], x[-1]

        return [self.E * (λ - 1), statevars]

    def hessian(self, x=None, shape=(1,), dtype=None, out=None):
        r"""Evaluate the elasticity. The stretch is only used for the shape of the
        trailing axes.

        Parameters
        ----------
        x : list of ndarray, optional
            List with stretch :math:`\lambda` as first item (default is None).
        shape : tuple of int, optional
            Tuple with shape of the trailing axes (default is (1,)).
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
            Not implemented.

        Returns
        -------
        ndarray
            Elasticity

        """

        return [self.E]
