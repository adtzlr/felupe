# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace

import numpy as np

from ..math import (
    cdya_il,
    ddot,
    det,
    dya,
    identity,
    inv,
    ravel,
    reshape,
    transpose,
)
from ._base import ConstitutiveMaterial
from .hyperelasticity._volumetric_strain_energy import _volumetric_strain_energy


class NearlyIncompressible(ConstitutiveMaterial):
    r"""A nearly-incompressible material formulation to augment the distortional part of
    the strain energy function by a volumetric part and a constraint equation.

    Notes
    -----
    The total potential energy of internal forces is given in Eq.
    :eq:`nearlyincompressible`.

    ..  math::
        :label: nearlyincompressible

        \Pi_{int}(\boldsymbol{F}, p, \bar{J}) =
            \int_V \hat{\psi}(\boldsymbol{F})\ dV +
            \int_V U(\bar{J})\ dV +
            \int_V p (J - \bar{J})\ dV

    The volumetric part of the strain energy density function :math:`U(\bar{J})` is
    given by ``volumetric``, see :class:`~felupe.VolumetricStrainEnergy`. Its first and
    second derivatives w.r.t. the volume ratio are abbreviated by
    :math:`\bar{U}'=U'(\bar{J})` and :math:`\bar{U}''=U''(\bar{J})`, e.g. for
    ``volumetric="volume-ratio"`` (default)

    ..  math::

        U(\bar{J}) &= \frac{K}{2} \left( \bar{J} - 1 \right)^2

        \bar{U}' &= K \left( \bar{J} - 1 \right)

        \bar{U}'' &= K

    Parameters
    ----------
    material : ConstitutiveMaterial
        A hyperelastic material definition for the distortional part of the strain
        energy density function :math:`\hat{\psi}(\boldsymbol{F})` with methods for the
        ``gradient`` :math:`\partial_\boldsymbol{F}(\hat{\psi})` and the ``hessian``
        :math:`\partial_\boldsymbol{F}[\partial_\boldsymbol{F}(\hat{\psi})]` w.r.t the
        deformation gradient tensor :math:`\boldsymbol{F}`.
    bulk : float or None, optional
        The bulk modulus :math:`K` of the volumetric part of the strain energy density
        function, which is required if ``volumetric`` is a kind. It must be None if
        ``volumetric`` is an instance, which holds its own parameters (default is None).
    parallel : bool or None, optional
        A flag to evaluate the gradient and the hessian in chunks of cells by a pool of
        threads. If None, the flag is taken from the material (default is None).
    volumetric : str or VolumetricStrainEnergy, optional
        The volumetric part of the strain energy density function :math:`U(\bar{J})`,
        given by a kind with the bulk modulus, ``"volume-ratio"`` for
        :math:`U(\bar{J})=K(\bar{J}-1)^2/2`, ``"stretch"`` for
        :math:`U(\bar{J})=9K(\bar{J}^{1/3}-1)^2/2` or ``"log"`` for
        :math:`U(\bar{J})=K\ln(\bar{J})^2/2`, or by an instance of
        :class:`~felupe.VolumetricStrainEnergy`, e.g. of a custom subclass. Default is
        ``"volume-ratio"``.

    Examples
    --------

    ..  plot::

        >>> import felupe as fem
        >>>
        >>> field = fem.FieldsMixed(fem.RegionHexahedron(fem.Cube(n=6)), n=3)
        >>> boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        >>> umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
        >>> solid = fem.SolidBody(umat, field)

    The volumetric part of the strain energy density function is given by a kind or by
    an instance of :class:`~felupe.VolumetricStrainEnergy`.

    >>> umat = fem.NearlyIncompressible(
    ...     fem.NeoHooke(mu=1), bulk=5000, volumetric="stretch"
    ... )

    See Also
    --------
    ThreeFieldVariation : Hu-Washizu hydrostatic-volumetric selective three-field
        variation for nearly-incompressible material formulations.

    """

    def __init__(self, material, bulk=None, parallel=None, volumetric="volume-ratio"):
        self.material = self.fun = material

        if parallel is None:
            parallel = getattr(material, "parallel", False)

        self.parallel = parallel
        self.bulk = bulk

        # volumetric part of the strain energy density function U(J), given by a kind
        # (with the bulk modulus) or by an instance
        self._volumetric = volumetric

        if _volumetric_strain_energy(volumetric, bulk) is None:
            raise ValueError(
                f'volumetric="{volumetric}" requires the bulk modulus, e.g. bulk=5000.'
            )

        self.x = [material.x[0], np.ones(1), np.ones(1), material.x[-1]]

    @property
    def volumetric(self):
        """The volumetric part of the strain energy density function, see
        :class:`~felupe.VolumetricStrainEnergy`. For a kind, it is created with the
        current bulk modulus.
        """
        return _volumetric_strain_energy(self._volumetric, self.bulk)

    def _gradient(self, x):
        r"""Return a list with the gradient of the strain energy density function
        w.r.t. the fields displacements, pressure and volume ratio.

        Parameters
        ----------
        x : list of ndarray
            List of extracted field values with the deformation gradient tensor
            :math:`\boldsymbol{F}` as first, the pressure :math:`p` as
            second and the volume ratio :math:`\bar{J}` as third list item. Initial
            state variables are stored in the last (fourth) list item.

        Returns
        -------
        list of ndarrays
            List of gradients w.r.t. the input variables :math:`\boldsymbol{F}`,
            :math:`p` and :math:`\bar{J}`. The last item of the list contains the
            updated state variables.

        Notes
        -----
        ..  math::

            \delta_\boldsymbol{u}(\Pi_{int}) &=
                \int_V \left( \frac{\partial \hat{\psi}}{\partial \boldsymbol{F}} +
                p\ J \boldsymbol{F}^{-T} \right) : \delta\boldsymbol{F}\ dV

            \delta_p(\Pi_{int}) &=
                \int_V \left( J - \bar{J} \right)\ \delta p\ dV

            \delta_\bar{J}(\Pi_{int}) &=
                \int_V \left( \bar{U}' - p \right)\ \delta \bar{J}\ dV

        """
        [F, p, J], statevars = x[:3], x[-1]
        dWdF, statevars_new = self.material.gradient([F, statevars])
        dWdF += p * transpose(inv(F, determinant=1.0))
        dWdp = det(F) - J
        dWdJ = self.volumetric.dUdJ(J) - p
        return [dWdF, dWdp, dWdJ, statevars_new]

    def _hessian(self, x):
        r"""Return a list with the hessian of the strain energy density function
        w.r.t. the fields displacements, pressure and volume ratio.

        Parameters
        ----------
        x : list of ndarray
            List of extracted field values with the deformation gradient tensor
            :math:`\boldsymbol{F}` as first, the pressure :math:`p` as
            second and the volume ratio :math:`\bar{J}` as third list item. Initial
            state variables are stored in the last (fourth) list item.

        Returns
        -------
        list of ndarrays
            List of the hessian w.r.t. the input variables :math:`\boldsymbol{F}`,
            :math:`p` and :math:`\bar{J}`. The upper-triangle items of the hessian are
            returned as the items of the list.

        Notes
        -----
        ..  math::

            \Delta_\boldsymbol{u}\delta_\boldsymbol{u}(\Pi_{int}) &= \int_V
                \delta\boldsymbol{F} : \left[
                \frac{\partial^2 \hat{\psi}}
                {\partial\boldsymbol{F}\ \partial\boldsymbol{F}} +
                p\ J \left( \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} -
                \boldsymbol{F}^{-T} \overset{il}{\odot} \boldsymbol{F}^{-T} \right)
                \right] : \Delta\boldsymbol{F}\ dV

            \Delta_p\delta_\boldsymbol{u}(\Pi_{int}) &= \int_V
                \delta\boldsymbol{F} : J \boldsymbol{F}^{-T}\ \Delta p\ dV

            \Delta_\bar{J}\delta_\boldsymbol{u}(\Pi_{int}) &= \int_V
                \delta\boldsymbol{F} : \boldsymbol{0}\ \Delta \bar{J}\ dV

            \Delta_p\delta_p(\Pi_{int}) &= \int_V \delta p\ (0)\ \Delta p\ dV

            \Delta_p\delta_\bar{J}(\Pi_{int}) &= \int_V
                \delta \bar{J}\ (-1)\ \Delta p\ dV

            \Delta_\bar{J}\delta_\bar{J}(\Pi_{int}) &= \int_V
                \delta \bar{J}\ \bar{U}''\ \Delta \bar{J}\ dV

        """
        [F, p, J], statevars = x[:3], x[-1]
        detF = det(F)
        iFT = transpose(inv(F, determinant=detF))
        d2WdFdF = self.material.hessian([F, statevars])[0]
        d2WdFdF += p * detF * (dya(iFT, iFT) - cdya_il(iFT, iFT))
        d2WdFdp = detF * iFT
        d2WdFdJ = None  # np.zeros_like(F)
        d2Wdpdp = None  # np.zeros_like(p)
        d2WdpdJ = -np.ones_like(J)
        d2WdJdJ = self.volumetric.d2UdJdJ(J) * np.ones_like(J)
        return [d2WdFdF, d2WdFdp, d2WdFdJ, d2Wdpdp, d2WdpdJ, d2WdJdJ]


class ThreeFieldVariation(ConstitutiveMaterial):
    r"""Hu-Washizu hydrostatic-volumetric selective
    :math:`(\boldsymbol{u},p,J)` - three-field variation for nearly-
    incompressible material formulations. The total potential energy
    for nearly-incompressible hyperelasticity is formulated with a
    determinant-modified deformation gradient. Pressure and volume ratio fields
    should be kept one order lower than the interpolation order of the
    displacement field, e.g. linear displacement fields should be paired with
    element-constant (mean) values of pressure and volume ratio.

    The total potential energy of internal forces is defined with a strain
    energy density function in terms of a determinant-modified deformation
    gradient and an additional control equation.

    ..  math::

        \Pi &= \Pi_{int} + \Pi_{ext}

        \Pi_{int} &= \int_V \psi(\boldsymbol{F}) \ dV \qquad \rightarrow
        \qquad \Pi_{int}(\boldsymbol{u},p,J) = \int_V \psi(\overline{\boldsymbol{F}})
        \ dV + \int_V p (J-\overline{J}) \ dV

        \overline{\boldsymbol{F}} &=
        \left(\frac{\overline{J}}{J}\right)^{1/3} \boldsymbol{F}

    The variations of the total potential energy w.r.t.
    :math:`(\boldsymbol{u},p,J)` lead to the following expressions. We denote
    first partial derivatives as :math:`\boldsymbol{f}_{(\bullet)}` and second
    partial derivatives as :math:`\boldsymbol{A}_{(\bullet,\bullet)}`.

    ..  math::

        \delta_{\boldsymbol{u}} \Pi_{int} &= \int_V \boldsymbol{P} :
        \delta \boldsymbol{F} \ dV = \int_V \left( \frac{\partial \psi}
        {\partial \overline{\boldsymbol{F}}} : \frac{\partial \overline{\boldsymbol{F}}}
        {\partial \boldsymbol{F}} + p J \boldsymbol{F}^{-T} \right) :
        \delta \boldsymbol{F} \ dV

        \delta_{p} \Pi_{int} &= \int_V f_{p} \ \delta p \ dV =
        \int_V (J - \overline{J}) \ \delta p \ dV

        \delta_{\overline{J}} \Pi_{int} &= \int_V f_{\overline{J}} \ \delta \overline{J}
        \ dV = \int_V \left( \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} :
        \frac{\partial \overline{\boldsymbol{F}}}{\partial \overline{J}} - p \right) :
        \delta \overline{J} \ dV

    The projection tensors from the variations lead the following results.

    ..  math::

        \frac{\partial \overline{\boldsymbol{F}}}{\partial \boldsymbol{F}} &=
        \left(\frac{\overline{J}}{J}\right)^{1/3} \left( \boldsymbol{I}
        \overset{ik}{\odot} \boldsymbol{I} - \frac{1}{3} \boldsymbol{F} \otimes
        \boldsymbol{F}^{-T} \right)

        \frac{\partial \overline{\boldsymbol{F}}}{\partial \overline{J}} &=
        \frac{1}{3 \overline{J}} \overline{\boldsymbol{F}}

    The double-dot products from the variations are now evaluated.

    ..  math::

        \overline{\boldsymbol{P}} &= \frac{\partial \psi}{\partial
        \overline{\boldsymbol{F}}} = \overline{\overline{\boldsymbol{P}}} - \frac{1}{3}
        \left(  \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right)
        \boldsymbol{F}^{-T} \qquad \text{with} \qquad
        \overline{\overline{\boldsymbol{P}}} = \left(\frac{\overline{J}}{J}\right)^{1/3}
        \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}}

        \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} :
        \frac{1}{3 \overline{J}} \overline{\boldsymbol{F}} &= \frac{1}{3 \overline{J}}
        \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F}

    We now have three formulas; one for the first Piola Kirchhoff stress and
    two additional control equations.

    ..  math::

        \boldsymbol{P} &=
        \overline{\overline{\boldsymbol{P}}} - \frac{1}{3} \left(
        \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right)
        \boldsymbol{F}^{-T}

        f_p &= J - \overline{J}

        f_{\overline{J}} &=  \frac{1}{3 \overline{J}} \left(
        \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) - p

    A linearization of the above formulas gives six equations (only results are
    given here).

    ..  math::

        \mathbb{A}_{\boldsymbol{u},\boldsymbol{u}} &=  \overline{\overline{\mathbb{A}}}
        + \frac{1}{9} \left(  \boldsymbol{F} : \overline{\overline{\mathbb{A}}} :
        \boldsymbol{F} \right) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} -
        \frac{1}{3} \left( \boldsymbol{F}^{-T} \otimes \left(
        \overline{\overline{\boldsymbol{P}}} + \boldsymbol{F} :
        \overline{\overline{\mathbb{A}}} \right) + \left(
        \overline{\overline{\boldsymbol{P}}} + \overline{\overline{\mathbb{A}}} :
        \boldsymbol{F} \right) \otimes \boldsymbol{F}^{-T} \right)

        &+\left( p J + \frac{1}{9} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F}
        \right) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \left( p J -
        \frac{1}{3} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right)
        \boldsymbol{F}^{-T} \overset{il}{\odot} \boldsymbol{F}^{-T}

        A_{p,p} &= 0

        A_{\overline{J},\overline{J}} &= \frac{1}{9 \overline{J}^2} \left(
        \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) -
        2 \left( \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right)

        \boldsymbol{A}_{\boldsymbol{u},p} &= \boldsymbol{A}_{p, \boldsymbol{u}} =
        J \boldsymbol{F}^{-T}

        \boldsymbol{A}_{\boldsymbol{u},\overline{J}} &= \boldsymbol{A}_{\overline{J},
        \boldsymbol{u}} = \frac{1}{3 \overline{J}} \left( \boldsymbol{P}' +
        \boldsymbol{F} : \overline{\overline{\mathbb{A}}} - \frac{1}{3} \left(
        \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right)
        \boldsymbol{F}^{-T} \right)

        A_{p,\overline{J}} &= A_{\overline{J}, p} = -1


    with

    ..  math::

        \overline{\overline{\mathbb{A}}} = \left(\frac{\overline{J}}{J}\right)^{1/3}
        \frac{\partial^2 \psi}{\partial \overline{\boldsymbol{F}} \partial
        \overline{\boldsymbol{F}}} \left(\frac{\overline{J}}{J}\right)^{1/3}

    as well as

    ..  math::

        \boldsymbol{P}' = \boldsymbol{P} - p J \boldsymbol{F}^{-T}

    Parameters
    ----------
    material : ConstitutiveMaterial
        A hyperelastic material definition for the strain energy density function with
        methods for the ``gradient`` and the ``hessian`` w.r.t the deformation gradient
        tensor.
    parallel : bool or None, optional
        A flag to evaluate the gradient and the hessian in chunks of cells by a pool of
        threads. If None, the flag is taken from the material (default is None).

    """

    def __init__(self, material, parallel=None):
        self.material = self.fun = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], np.ones(1), np.ones(1), material.x[-1]]

        if parallel is None:
            parallel = getattr(material, "parallel", False)

        self.parallel = parallel

    def _gradient_u(self, k, p, J):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

        ..  code-block::

            δ_u(Π_int) = ∫_V (∂ψ/∂F + p cof(F)) : δF dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            Gradient w.r.t. the deformation gradient

        """

        return k.Pbb - k.PbbF / 3 * k.iFT + p * k.detF * k.iFT

    def _gradient_p(self, k, p, J):
        """Variation of total potential energy w.r.t pressure.

        ..  code-block::

            δ_p(Π_int) = ∫_V (det(F) - J) δp dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            Gradient w.r.t. the pressure

        """

        return k.detF - J

    def _gradient_J(self, k, p, J):
        """Variation of total potential energy w.r.t volume ratio.

        ..  code-block::

            δ_J(Π_int) = ∫_V (∂U/∂J - p) δJ dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            Gradient w.r.t. the volume ratio

        """

        return k.PbbF / (3 * J) - p

    def _gradient(self, x):
        r"""Return a list of variations of the total potential energy w.r.t. the fields
        displacements, pressure and volume ratio.

        Parameters
        ----------
        x : list of ndarray
            List of extracted field values with the Deformation gradient tensor
            :math:`\boldsymbol{F}` as first, the hydrostatic pressure :math:`p` as
            second and the volume ratio :math:`\bar{J}` as third list item.

        Returns
        -------
        list of ndarrays
            List of gradients w.r.t. the input variables :math:`\boldsymbol{F}`,
            :math:`p` and :math:`\bar{J}`.

        """
        [F, p, J], statevars = x[:3], x[-1]
        k = SimpleNamespace()

        k.detF = det(F)
        k.iFT = transpose(inv(F))
        k.Fb = (J / k.detF) ** (1 / 3) * F
        k.Pb, statevars_new = self.material.gradient([k.Fb, statevars])
        k.Pbb = (J / k.detF) ** (1 / 3) * k.Pb
        k.PbbF = ddot(k.Pbb, F, mode=(2, 2))

        return [
            self._gradient_u(k, p, J),
            self._gradient_p(k, p, J),
            self._gradient_J(k, p, J),
            statevars_new,
        ]

    def _hessian(self, x):
        r"""List of linearized variations of total potential energy w.r.t
        displacements, pressure and volume ratio (these expressions are
        symmetric; ``A_up = A_pu`` if derived from a total potential energy
        formulation). List entries have to be arranged as a flattened list
        from the upper triangle blocks:

        ..  code-block::

            Δ_u(δ_u(Π_int)) = ∫_V δF : (∂²ψ/(∂F∂F) + p ∂cof(F)/∂F) : ΔF dV
            Δ_p(δ_u(Π_int)) = ∫_V δF : J cof(F) Δp dV
            Δ_J(δ_u(Π_int)) = ∫_V δF :  ∂²ψ/(∂F∂J) ΔJ dV
            Δ_p(δ_p(Π_int)) = ∫_V δp 0 Δp dV
            Δ_J(δ_p(Π_int)) = ∫_V δp (-1) ΔJ dV
            Δ_J(δ_J(Π_int)) = ∫_V δJ ∂²ψ/(∂J∂J) ΔJ dV

            [[0 1 2],
             [  3 4],
             [    5]] --> [0 1 2 3 4 5]


        Arguments
        ---------
        extract : list of ndarray
            List of extracted field values with Deformation gradient ``F``
            as first, the hydrostatic pressure ``p`` as second and the
            volume ratio ``J`` as third item.

        Returns
        -------
        list of ndarrays
            List of hessians in upper triangle order

        """

        [F, p, J], statevars = x[:3], x[-1]
        k = SimpleNamespace()

        k.detF = det(F)
        k.iFT = transpose(inv(F))
        k.Fb = (J / k.detF) ** (1 / 3) * F
        k.Pbb = (J / k.detF) ** (1 / 3) * self.material.gradient([k.Fb, statevars])[0]

        k.A4b = self.material.hessian([k.Fb, statevars])[0]
        k.A4bb = (J / k.detF) ** (2 / 3) * k.A4b

        k.PbbF = ddot(k.Pbb, F, mode=(2, 2))
        k.FA4bb = ddot(F, k.A4bb, mode=(2, 4))
        k.A4bbF = ddot(k.A4bb, F, mode=(4, 2))
        k.FA4bbF = ddot(F, k.A4bbF, mode=(2, 2))

        return [
            self._hessian_uu(k, p, J),
            self._hessian_up(k, p, J),
            self._hessian_uJ(k, p, J),
            self._hessian_pp(k, p, J),
            self._hessian_pJ(k, p, J),
            self._hessian_JJ(k, p, J),
        ]

    def _hessian_uu(self, k, p=None, J=None):
        """Linearization w.r.t. displacements of variation of
        total potential energy w.r.t displacements.

        ..  code-block::

            Δ_u(δ_u(Π_int)) = ∫_V δF : (∂²ψ/(∂F∂F) + p ∂cof(F)/∂F) : ΔF dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            u,u - part of hessian
        """

        PbbA4bbF = k.Pbb + k.A4bbF
        PbbFA4bb = k.Pbb + k.FA4bb

        pJ9 = p * k.detF + k.PbbF / 9
        pJ3 = p * k.detF - k.PbbF / 3

        A4 = (
            k.A4bb
            + k.FA4bbF * dya(k.iFT, k.iFT) / 9
            - (dya(PbbA4bbF, k.iFT) + dya(k.iFT, PbbFA4bb)) / 3
            + pJ9 * dya(k.iFT, k.iFT)
            - pJ3 * cdya_il(k.iFT, k.iFT)
        )

        return A4

    def _hessian_pp(self, k, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t pressure.

        ..  code-block::

            Δ_p(δ_p(Π_int)) = ∫_V δp 0 Δp dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            p,p - part of hessian
        """
        return None  # np.zeros_like(p)

    def _hessian_JJ(self, k, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t volume ratio.

        ..  code-block::

            Δ_J(δ_J(Π_int)) = ∫_V δJ ∂²ψ/(∂J∂J) ΔJ dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            J,J - part of hessian
        """

        return (k.FA4bbF - 2 * k.PbbF) / (9 * J**2)

    def _hessian_up(self, k, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t displacements.

        ..  code-block::

            Δ_p(δ_u(Π_int)) = ∫_V δF : J cof(F) Δp dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            u,p - part of hessian
        """

        return k.detF * k.iFT

    def _hessian_uJ(self, k, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t displacements.

        ..  code-block::

            Δ_J(δ_u(Π_int)) = ∫_V δF :  ∂²ψ/(∂F∂J) ΔJ dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            u,J - part of hessian
        """

        Ps = self._gradient_u(k, 0 * p, J)
        return (-k.FA4bbF / 3 * k.iFT + Ps + k.FA4bb) / (3 * J)

    def _hessian_pJ(self, k, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t pressure.

        ..  code-block::

            Δ_J(δ_p(Π_int)) = ∫_V δp (-1) ΔJ dV

        Arguments
        ---------
        k : SimpleNamespace
            Intermediate results of a chunk of cells, e.g. ``k.detF``.
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            p,J - part of hessian
        """
        return -np.ones_like(J)


class ThirdMediumContactMixed(ConstitutiveMaterial):
    r"""A mixed formulation for a stabilized third medium contact [1]_.

    The second-order regularization of the third medium is expressed by an additional
    tensor-valued field :math:`\boldsymbol{\theta}` which approximates the displacement
    gradient :math:`\boldsymbol{H} = \boldsymbol{F} - \boldsymbol{1}`. Hence, only
    first-order derivatives of the fields are required.

    Parameters
    ----------
    material : ConstitutiveMaterial
        A hyperelastic material definition for the strain energy density function
        :math:`\psi(\boldsymbol{F})` of the third medium with methods for the
        ``gradient`` and the ``hessian`` w.r.t. the deformation gradient tensor.
    gamma : float
        The scaling factor :math:`\gamma` of the strain energy density function of the
        third medium.
    alpha_r : float
        The regularization parameter :math:`\alpha_r` for the gradient of the
        tensor-valued field.
    p_r : float
        The penalty parameter :math:`p_r` which enforces the constraint
        :math:`\boldsymbol{\theta} = \boldsymbol{H}`.
    parallel : bool or None, optional
        A flag to evaluate the gradient and the hessian in chunks of cells by a pool of
        threads. If None, the flag is taken from the material (default is None).

    Notes
    -----
    The strain energy density function is given in Eq.
    :eq:`third-medium-contact-mixed`.

    ..  math::
        :label: third-medium-contact-mixed

        \Psi(\boldsymbol{F}, \boldsymbol{\theta}, \nabla \boldsymbol{\theta}) =
            \gamma\ \psi(\boldsymbol{F})
            + \frac{p_r}{2} \left( \boldsymbol{\theta} - \boldsymbol{H} \right) :
              \left( \boldsymbol{\theta} - \boldsymbol{H} \right)
            + \frac{\alpha_r}{2}\ \nabla \boldsymbol{\theta} ~\vdots~
              \nabla \boldsymbol{\theta}

    The field :math:`\boldsymbol{\theta}` is a (non-axisymmetric) :class:`~felupe.Field`
    with nine components. It is taken twice in the field container, where the
    gradient is evaluated only for the second occurrence. This results in the list of
    kinematic quantities :math:`(\boldsymbol{F}, \boldsymbol{\theta},
    \nabla \boldsymbol{\theta})`. For two-dimensional regions, the gradient
    :math:`\nabla \boldsymbol{\theta}` is evaluated w.r.t. the in-plane coordinates.

    Examples
    --------

    ..  plot::

        >>> import felupe as fem
        >>>
        >>> region = fem.RegionQuad(fem.Rectangle(n=6))
        >>> field = fem.FieldContainer(
        ...     [fem.FieldPlaneStrain(region, dim=2), fem.Field(region, dim=9)],
        ...     take=[0, 1, 1],
        ... )
        >>> boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        >>> umat = fem.ThirdMediumContactMixed(
        ...     fem.NeoHooke(mu=1, bulk=20), gamma=1e-5, alpha_r=1e-4, p_r=1e-2
        ... )
        >>> solid = fem.SolidBody(umat, field, grad=[True, False, True])

    References
    ----------
    ..  [1] M. Vorwerk, J. Schröder, and P. Wriggers, "A mixed finite element
        formulation for stabilized third medium contact", Computer Methods in Applied
        Mechanics and Engineering, vol. 463, p. 119416, Jan. 2027. doi:
        `10.1016/j.cma.2026.119416 <https://doi.org/10.1016/j.cma.2026.119416>`_.

    """

    def __init__(
        self,
        material,
        gamma,
        alpha_r,
        p_r,
        parallel=None,
    ):
        self.material = self.fun = material

        if parallel is None:
            parallel = getattr(material, "parallel", False)

        self.parallel = parallel
        self.gamma = gamma
        self.alpha_r = alpha_r
        self.p_r = p_r

        # for plane-strain or axisymmetric first-fields, x[2].shape is (9, 2)
        # for cartesian 3d fields, x[2].shape is (9, 3)
        self.x = [material.x[0], np.ones(9), np.ones((9, 3)), material.x[-1]]

    def _gradient(self, x):
        [F, θ, grad_θ], statevars = x[:3], x[-1]
        dWdF, statevars_new = self.material.gradient([F, statevars])
        dWdF *= self.gamma

        # ψ_p = p_r * (θ - dudX : θ - dudX) / 2
        # ψ_g = α_r * (∇θ : ∇θ) / 2

        dudX = ravel(F - identity(F))
        dWdθ = self.p_r * (θ - dudX)
        dWdgradθ = self.alpha_r * grad_θ

        dWdF -= reshape(dWdθ, (3, 3))

        return [dWdF, dWdθ, dWdgradθ, statevars_new]

    def _hessian(self, x):
        # the hessian doesn't depend on the values of θ
        F, grad_θ, statevars = x[0], x[2], x[-1]

        # for plane-strain or axisymmetric first-fields, grad_θ.shape is (9, 2)
        # for cartesian 3d fields, grad_θ.shape is (9, 3)
        grad_θ_size = np.prod(grad_θ.shape[:2])
        grad_θ_inplane_dim = grad_θ.shape[1]

        identity_9_9 = identity(dim=9, shape=(1, 1))
        identity_grad_θ = identity(dim=grad_θ_size, shape=(1, 1))

        # ψ_p = p_r * (θ - dudX : θ - dudX) / 2
        # ψ_g = α_r * (∇θ : ∇θ) / 2

        d2WdFdF = self.material.hessian([F, statevars])[0]
        d2WdFdF *= self.gamma
        d2WdFdF += self.p_r * reshape(identity_9_9, (3, 3, 3, 3))

        d2WdFdθ = -self.p_r * reshape(identity_9_9, (3, 3, 9))
        d2WdFdgradθ = None
        d2Wdθdθ = self.p_r * identity_9_9
        d2Wdθdgradθ = None
        d2Wdgradθdgradθ = self.alpha_r * reshape(
            identity_grad_θ,
            shape=(9, grad_θ_inplane_dim, 9, grad_θ_inplane_dim),
        )
        return [d2WdFdF, d2WdFdθ, d2WdFdgradθ, d2Wdθdθ, d2Wdθdgradθ, d2Wdgradθdgradθ]
