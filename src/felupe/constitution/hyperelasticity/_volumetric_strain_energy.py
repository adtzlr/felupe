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
    deformation gradient, it is evaluated by the volume ratio
    :math:`J=\det(\boldsymbol{F})`. An instance ``U`` is callable, i.e. ``U(J)``
    returns :math:`U(J)`, ``U.dUdJ(J)`` returns :math:`U'(J)` and ``U.d2UdJdJ(J)``
    returns :math:`U''(J)`. The
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
    implements the methods ``__call__(J)``, ``dUdJ(J)`` and ``d2UdJdJ(J)``. The
    undeformed configuration is free of stress only if :math:`U'(1)=0`, and a positive
    second derivative :math:`U''(J)>0` is required for a stable volumetric response.

    The volumetric part of the strain energy density function is passed by the argument
    ``volumetric`` to :class:`~felupe.NeoHooke`, :class:`~felupe.Volumetric`,
    :class:`~felupe.NearlyIncompressible` and
    :class:`~felupe.SolidBodyNearlyIncompressible`, either as a kind together with the
    bulk modulus or as an instance (then, without the bulk modulus).

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> U = fem.VolumetricStrainEnergy(bulk=5000, kind="stretch")
    >>> J = np.array([0.9, 1.0, 1.1])
    >>>
    >>> U(J)
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
    ...     def __call__(self, J):
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

    The kinds or an instance are passed to the materials and the solid bodies.

    >>> umat = fem.NeoHooke(mu=1, bulk=5000, volumetric="stretch")
    >>> umat = fem.NeoHooke(mu=1, volumetric=U)
    >>>
    >>> mesh = fem.Cube(n=3)
    >>> field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    >>> umat = fem.NeoHooke(mu=1)
    >>> solid = fem.SolidBodyNearlyIncompressible(umat, field, volumetric=U)

    See Also
    --------
    felupe.VolumeChange : Volume Change.
    felupe.NeoHooke : Nearly-incompressible isotropic hyperelastic Neo-Hookean material
        formulation.
    felupe.Volumetric : Neo-Hookean material formulation with deactivated shear modulus.
    felupe.NearlyIncompressible : A nearly-incompressible material formulation to
        augment the distortional part of the strain energy function by a volumetric part
        and a constraint equation.
    felupe.SolidBodyNearlyIncompressible : A (nearly) incompressible solid body with
        methods for the assembly of sparse vectors/matrices.
    """

    kinds = ("volume-ratio", "stretch", "log")

    def __init__(self, bulk, kind="volume-ratio"):
        self._check_kind(kind)

        if bulk is None:
            raise ValueError(
                f'kind="{kind}" requires the bulk modulus, e.g. bulk=5000.'
            )

        self.bulk = bulk
        self.kind = kind

    @classmethod
    def _check_kind(cls, kind):
        "Raise an error for an unknown kind."

        if kind not in cls.kinds:
            raise ValueError(
                f'Unknown kind="{kind}". Available kinds are '
                + ", ".join(f'"{available}"' for available in cls.kinds)
                + "."
            )

    def __call__(self, J):
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


def _volumetric_strain_energy(volumetric, bulk):
    """Return the volumetric part of the strain energy density function, given by a
    kind with the bulk modulus or by an instance of
    :class:`~felupe.VolumetricStrainEnergy`. None is returned for a kind without the
    bulk modulus.

    Parameters
    ----------
    volumetric : str or VolumetricStrainEnergy
        The kind or the volumetric part of the strain energy density function.
    bulk : float or None
        The bulk modulus. It must be None if ``volumetric`` is an instance, because the
        instance holds its own parameters.

    Returns
    -------
    VolumetricStrainEnergy or None
        The volumetric part of the strain energy density function.
    """

    if isinstance(volumetric, str):
        VolumetricStrainEnergy._check_kind(volumetric)

        if bulk is None:
            return None

        return VolumetricStrainEnergy(bulk=bulk, kind=volumetric)

    if bulk is not None:
        raise ValueError(
            "The bulk modulus must be None if the volumetric part of the strain energy "
            "density function is given by an instance, which holds its own parameters, "
            'e.g. volumetric=VolumetricStrainEnergy(bulk=5000, kind="stretch").'
        )

    return volumetric
