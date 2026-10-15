# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


class VolumetricStrainEnergy:
    r"""The volumetric part of the strain energy density function as a function of the
    volume ratio, :math:`U(J)`, with its first and second derivatives w.r.t. the volume
    ratio.

    Parameters
    ----------
    bulk : float
        The bulk modulus :math:`K`.
    kind : str, optional
        The kind of the volumetric part of the strain energy density function:
        ``"volume-ratio"`` for :math:`U(J)=K(J-1)^2/2`, ``"stretch"`` for
        :math:`U(J)=9K(J^{1/3}-1)^2/2` or ``"log"`` for :math:`U(J)=K\ln(J)^2/2`, see
        Eq. :eq:`volumetric-strain-energy`. Default is ``"volume-ratio"``.

    Notes
    -----
    Unlike a :class:`~felupe.ConstitutiveMaterial`, which is evaluated by the
    deformation gradient, all methods are evaluated by the volume ratio
    :math:`J=\det(\boldsymbol{F})`, i.e. ``function(J)`` returns :math:`U(J)`,
    ``dUdJ(J)`` returns :math:`U'(J)` and ``d2UdJdJ(J)`` returns :math:`U''(J)`. The
    derivatives w.r.t. the deformation gradient are obtained by the chain rule with the
    derivatives of the volume ratio, see :class:`~felupe.VolumeChange`.

    The kinds are quadratic in the Seth-Hill strain :math:`E_m = (\lambda^m - 1) / m`
    of order :math:`m` of the volumetric stretch :math:`\lambda = J^{1/3}`, see Eq.
    :eq:`volumetric-strain-energy`, with :math:`m=3` for ``"volume-ratio"``,
    :math:`m=1` for ``"stretch"`` and the limit :math:`m \rightarrow 0` for ``"log"``.

    ..  math::
        :label: volumetric-strain-energy

        U(J) = \frac{9K}{2} E_m^2 = \frac{9K}{2 m^2} \left( J^{m/3} - 1 \right)^2

    For all kinds, the second derivative at the undeformed configuration is the bulk
    modulus, :math:`U''(1)=K`. The volumetric part and its first and second derivatives
    are given in Eq. :eq:`volumetric-strain-energy-volume-ratio` for
    ``kind="volume-ratio"`` (default),

    ..  math::
        :label: volumetric-strain-energy-volume-ratio

        U(J) &= \frac{K}{2} \left( J - 1 \right)^2

        U'(J) &= K \left( J - 1 \right)

        U''(J) &= K

    in Eq. :eq:`volumetric-strain-energy-stretch` for ``kind="stretch"``

    ..  math::
        :label: volumetric-strain-energy-stretch

        U(J) &= \frac{9K}{2} \left( J^{1/3} - 1 \right)^2

        U'(J) &= 3K\ \frac{J^{1/3} - 1}{J^{2/3}}

        U''(J) &= K\ \frac{2 - J^{1/3}}{J^{5/3}}

    and in Eq. :eq:`volumetric-strain-energy-log` for ``kind="log"``.

    ..  math::
        :label: volumetric-strain-energy-log

        U(J) &= \frac{K}{2} \ln \left( J \right)^2

        U'(J) &= K\ \frac{\ln \left( J \right)}{J}

        U''(J) &= K\ \frac{1 - \ln \left( J \right)}{J^2}

    The kinds are evaluated by these expressions of the volume ratio, i.e. not by the
    general form, which is singular for :math:`m=0`. The cubic root of the kind
    ``"stretch"`` is evaluated by :func:`numpy.cbrt`, which is real-valued also for
    negative (linearized) volume ratios. The smaller the order :math:`m`, the stiffer is
    the volumetric response in compression and the softer in tension. The first
    derivative :math:`U'(J)`, i.e. the hydrostatic part of the Cauchy stress, is
    unbounded for :math:`J \rightarrow 0` for the kinds ``"stretch"`` and ``"log"``. In
    tension, it is bounded by :math:`3K/4` at :math:`J=8` and by :math:`K/e` at
    :math:`J=e` for these kinds, where their second derivatives change their sign.

    A custom volumetric part of the strain energy density function is a subclass which
    implements the methods ``function(J)``, ``dUdJ(J)`` and ``d2UdJdJ(J)``. The
    undeformed configuration is free of stress only if :math:`U'(1)=0`, and a positive
    second derivative :math:`U''(J)>0` is required for a stable volumetric response.

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> U = fem.VolumetricStrainEnergy(bulk=5000, kind="stretch")
    >>> J = np.array([0.9, 1.0, 1.1])
    >>>
    >>> U.function(J)
    array([26.79710794,  0.        , 23.44513171])

    >>> U.dUdJ(J)
    array([-555.32721366,    0.        ,  454.39256332])

    >>> U.d2UdJdJ(J)
    array([6165.48776188, 5000.        , 4127.92559565])

    A custom volumetric part of the strain energy density function is defined by a
    subclass.

    >>> class MyVolumetricStrainEnergy(fem.VolumetricStrainEnergy):
    ...     def __init__(self, bulk):
    ...         self.bulk = bulk
    ...
    ...     def function(self, J):
    ...         return self.bulk / 4 * (J**2 - 1 - 2 * np.log(J))
    ...
    ...     def dUdJ(self, J):
    ...         return self.bulk / 2 * (J - 1 / J)
    ...
    ...     def d2UdJdJ(self, J):
    ...         return self.bulk / 2 * (1 + 1 / J**2)
    >>>
    >>> U = MyVolumetricStrainEnergy(bulk=5000)
    >>> U.d2UdJdJ(1.0)
    5000.0

    See Also
    --------
    felupe.VolumeChange : Volume Change.
    felupe.SolidBodyNearlyIncompressible : A (nearly) incompressible solid body with
        methods for the assembly of sparse vectors/matrices.
    """

    kinds = ("volume-ratio", "stretch", "log")

    def __init__(self, bulk, kind="volume-ratio"):
        if kind not in self.kinds:
            raise ValueError(
                f'Unknown kind="{kind}". Available kinds are '
                + ", ".join(f'"{available}"' for available in self.kinds)
                + "."
            )

        if bulk is None:
            raise ValueError(
                f'kind="{kind}" requires the bulk modulus, e.g. bulk=5000.'
            )

        self.bulk = bulk
        self.kind = kind

    def function(self, J):
        r"""Return the volumetric part of the strain energy density function
        :math:`U(J)`.

        Parameters
        ----------
        J : float or ndarray
            The volume ratio.

        Returns
        -------
        float or ndarray
            The volumetric part of the strain energy density function.
        """

        K = self.bulk

        if self.kind == "stretch":
            return 9 * K * (np.cbrt(J) - 1) ** 2 / 2

        if self.kind == "log":
            return K * np.log(J) ** 2 / 2

        return K * (J - 1) ** 2 / 2

    def dUdJ(self, J):
        r"""Return the first derivative of the volumetric part of the strain energy
        density function w.r.t. the volume ratio :math:`U'(J)`.

        Parameters
        ----------
        J : float or ndarray
            The volume ratio.

        Returns
        -------
        float or ndarray
            The first derivative w.r.t. the volume ratio.
        """

        K = self.bulk

        if self.kind == "stretch":
            return 3 * K * (np.cbrt(J) - 1) / np.cbrt(J) ** 2

        if self.kind == "log":
            return K * np.log(J) / J

        return K * (J - 1)

    def d2UdJdJ(self, J):
        r"""Return the second derivative of the volumetric part of the strain energy
        density function w.r.t. the volume ratio :math:`U''(J)`.

        Parameters
        ----------
        J : float or ndarray
            The volume ratio.

        Returns
        -------
        float or ndarray
            The second derivative w.r.t. the volume ratio. For
            ``kind="volume-ratio"``, it is the bulk modulus (a scalar).
        """

        K = self.bulk

        if self.kind == "stretch":
            return K * (2 - np.cbrt(J)) / np.cbrt(J) ** 5

        if self.kind == "log":
            return K * (1 - np.log(J)) / (J**2)

        return K
