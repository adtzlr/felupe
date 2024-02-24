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

import inspect

import numpy as np

from ..math import cdya_ik, cdya_il, ddot, det, dya, identity, inv, transpose
from ._base import ConstitutiveMaterial


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

    The volumetric part of the strain energy density function is denoted in Eq.
    :eq:`nearlyincompressible-volumetric` along with its first and second derivatives.

    ..  math::
        :label: nearlyincompressible-volumetric

        \bar{U} &= \frac{K}{2} \left( \bar{J} - 1 \right)^2

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
    bulk : float
        The bulk modulus :math:`K` for the volumetric part of the strain energy
        function.
    parallel : bool, optional
        A flag to invoke parallel (threaded) math operations (default is False).
    dUdJ : callable, optional
        A function which evaluates the derivative of the volumetric part of the strain
        energy function :math:`\bar{U}'` w.r.t. the volume ratio :math:`\bar{J}`.
        Function signature must be ``lambda J, bulk: dUdJ``. Default is
        :math:`\bar{U}' = K (\bar{J} - 1)` or ``lambda J, bulk: bulk * (J - 1)``.
    d2UdJdJ : callable, optional
        A function which evaluates the second derivative of the volumetric part of the
        strain energy function :math:`\bar{U}''` w.r.t. the volume ratio
        :math:`\bar{J}`. Function signature must be ``lambda J, bulk: d2UdJdJ``.
        Default is :math:`\bar{U}'' = K` or ``lambda J, bulk: bulk``.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> field = fem.FieldsMixed(fem.RegionHexahedron(fem.Cube(n=6)), n=3)
    >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
    >>> umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
    >>> solid = fem.SolidBody(umat, field)
    >>> job = fem.Job(steps=[fem.Step(items=[solid], boundaries=boundaries)]).evaluate()

    See Also
    --------
    ThreeFieldVariation : Hu-Washizu hydrostatic-volumetric selective three-field
        variation for nearly-incompressible material formulations.

    """

    def __init__(
        self,
        material,
        bulk,
        parallel=False,
        dUdJ=lambda J, bulk: bulk * (J - 1),
        d2UdJdJ=lambda J, bulk: bulk,
    ):
        self.material = self.fun = material
        self.parallel = parallel
        self.bulk = bulk
        self.dUdJ = dUdJ
        self.d2UdJdJ = d2UdJdJ
        self.x = [material.x[0], np.ones(1), np.ones(1), material.x[-1]]

    def gradient(self, x, out=None):
        r"""Return a list with the gradient of the strain energy density function
        w.r.t. the fields displacements, pressure and volume ratio.

        Parameters
        ----------
        x : list of ndarray
            List of extracted field values with the deformation gradient tensor
            :math:`\boldsymbol{F}` as first, the pressure :math:`p` as
            second and the volume ratio :math:`\bar{J}` as third list item. Initial
            state variables are stored in the last (fourth) list item.
        out : ndarray or None, optional
            A location into which the result is stored (default is None).

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
        kwargs = {}
        if "out" in inspect.signature(self.material.gradient).parameters:
            kwargs["out"] = out

        [F, p, J], statevars = x[:3], x[-1]
        dWdF, statevars_new = self.material.gradient([F, statevars], **kwargs)
        dWdF += p * transpose(inv(F, determinant=1.0))
        dWdp = det(F) - J
        dWdJ = self.dUdJ(J, self.bulk) - p
        return [dWdF, dWdp, dWdJ, statevars_new]

    def hessian(self, x, out=None):
        r"""Return a list with the hessian of the strain energy density function
        w.r.t. the fields displacements, pressure and volume ratio.

        Parameters
        ----------
        x : list of ndarray
            List of extracted field values with the deformation gradient tensor
            :math:`\boldsymbol{F}` as first, the pressure :math:`p` as
            second and the volume ratio :math:`\bar{J}` as third list item. Initial
            state variables are stored in the last (fourth) list item.
        out : ndarray or None, optional
            A location into which the result is stored (default is None).

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
        kwargs = {}
        if "out" in inspect.signature(self.material.hessian).parameters:
            kwargs["out"] = out

        [F, p, J], statevars = x[:3], x[-1]
        detF = det(F)
        iFT = transpose(inv(F, determinant=detF))
        d2WdFdF = self.material.hessian([F, statevars], **kwargs)[0]
        d2WdFdF += p * detF * (dya(iFT, iFT) - cdya_il(iFT, iFT))
        d2WdFdp = detF * iFT
        d2WdFdJ = np.zeros_like(F)
        d2Wdpdp = np.zeros_like(p)
        d2WdpdJ = -np.ones_like(J)
        d2WdJdJ = self.d2UdJdJ(J, self.bulk) * np.ones_like(J)
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
    parallel : bool, optional
        A flag to invoke parallel (threaded) math operations (default is False).

    """

    def __init__(self, material, parallel=False):
        self.material = self.fun = material

        self._fun_P = self.material.gradient
        self._fun_A = self.material.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], np.ones(1), np.ones(1), material.x[-1]]

        self.parallel = parallel

    def _gradient_u(self, F, p, J):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

        ..  code-block::

            δ_u(Π_int) = ∫_V (∂ψ/∂F + p cof(F)) : δF dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            Gradient w.r.t. the deformation gradient

        """

        return self._Pbb - self._PbbF / 3 * self._iFT + p * self._detF * self._iFT

    def _gradient_p(self, F, p, J):
        """Variation of total potential energy w.r.t pressure.

        ..  code-block::

            δ_p(Π_int) = ∫_V (det(F) - J) δp dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            Gradient w.r.t. the pressure

        """

        return self._detF - J

    def _gradient_J(self, F, p, J):
        """Variation of total potential energy w.r.t volume ratio.

        ..  code-block::

            δ_J(Π_int) = ∫_V (∂U/∂J - p) δJ dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            Gradient w.r.t. the volume ratio

        """

        return self._PbbF / (3 * J) - p

    def gradient(self, x):
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
        kwargs = {}

        [F, p, J], statevars = x[:3], x[-1]

        self._detF = det(F)
        self._iFT = transpose(inv(F))
        self._Fb = (J / self._detF) ** (1 / 3) * F
        self._Pb, statevars_new = self._fun_P([self._Fb, statevars])
        self._Pbb = (J / self._detF) ** (1 / 3) * self._Pb
        self._PbbF = ddot(self._Pbb, F, mode=(2, 2), parallel=self.parallel)

        return [
            self._gradient_u(F, p, J),
            self._gradient_p(F, p, J),
            self._gradient_J(F, p, J),
            statevars_new,
        ]

    def hessian(self, x):
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

        self._detF = det(F)
        self._iFT = transpose(inv(F))
        self._Fb = (J / self._detF) ** (1 / 3) * F
        self._Pbb = (J / self._detF) ** (1 / 3) * self._fun_P([self._Fb, statevars])[0]

        self._eye = identity(F)
        self._P4 = cdya_ik(self._eye, self._eye, parallel=self.parallel) - 1 / 3 * dya(
            F, self._iFT, parallel=self.parallel
        )
        self._A4b = self._fun_A([self._Fb, statevars])[0]
        self._A4bb = (J / self._detF) ** (2 / 3) * self._A4b

        self._PbbF = ddot(self._Pbb, F, mode=(2, 2), parallel=self.parallel)
        self._FA4bb = ddot(F, self._A4bb, mode=(2, 4), parallel=self.parallel)
        self._A4bbF = ddot(self._A4bb, F, mode=(4, 2), parallel=self.parallel)
        self._FA4bbF = ddot(F, self._A4bbF, mode=(2, 2), parallel=self.parallel)

        return [
            self._hessian_uu(F, p, J),
            self._hessian_up(F, p, J),
            self._hessian_uJ(F, p, J),
            self._hessian_pp(F, p, J),
            self._hessian_pJ(F, p, J),
            self._hessian_JJ(F, p, J),
        ]

    def _hessian_uu(self, F, p=None, J=None):
        """Linearization w.r.t. displacements of variation of
        total potential energy w.r.t displacements.

        ..  code-block::

            Δ_u(δ_u(Π_int)) = ∫_V δF : (∂²ψ/(∂F∂F) + p ∂cof(F)/∂F) : ΔF dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            u,u - part of hessian
        """

        PbbA4bbF = self._Pbb + self._A4bbF
        PbbFA4bb = self._Pbb + self._FA4bb

        pJ9 = p * self._detF + self._PbbF / 9
        pJ3 = p * self._detF - self._PbbF / 3

        A4 = (
            self._A4bb
            + self._FA4bbF * dya(self._iFT, self._iFT, parallel=self.parallel) / 9
            - (
                dya(PbbA4bbF, self._iFT, parallel=self.parallel)
                + dya(self._iFT, PbbFA4bb, parallel=self.parallel)
            )
            / 3
            + pJ9 * dya(self._iFT, self._iFT, parallel=self.parallel)
            - pJ3 * cdya_il(self._iFT, self._iFT, parallel=self.parallel)
        )

        return A4

    def _hessian_pp(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t pressure.

        ..  code-block::

            Δ_p(δ_p(Π_int)) = ∫_V δp 0 Δp dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            p,p - part of hessian
        """
        return np.zeros_like(p)

    def _hessian_JJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t volume ratio.

        ..  code-block::

            Δ_J(δ_J(Π_int)) = ∫_V δJ ∂²ψ/(∂J∂J) ΔJ dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            J,J - part of hessian
        """

        return (self._FA4bbF - 2 * self._PbbF) / (9 * J**2)

    def _hessian_up(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t displacements.

        ..  code-block::

            Δ_p(δ_u(Π_int)) = ∫_V δF : J cof(F) Δp dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            u,p - part of hessian
        """

        return self._detF * self._iFT

    def _hessian_uJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t displacements.

        ..  code-block::

            Δ_J(δ_u(Π_int)) = ∫_V δF :  ∂²ψ/(∂F∂J) ΔJ dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
        p : ndarray
            Hydrostatic pressure
        J : ndarray
            Volume ratio

        Returns
        -------
        ndarray
            u,J - part of hessian
        """

        Ps = self._gradient_u(F, 0 * p, J)
        return (-self._FA4bbF / 3 * self._iFT + Ps + self._FA4bb) / (3 * J)

    def _hessian_pJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t pressure.

        ..  code-block::

            Δ_J(δ_p(Π_int)) = ∫_V δp (-1) ΔJ dV

        Arguments
        ---------
        F : ndarray
            Deformation gradient
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
