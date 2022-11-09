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

from ..math import (
    ddot,
    transpose,
    inv,
    dya,
    cdya_ik,
    cdya_il,
    det,
    identity,
)


class ThreeFieldVariation:
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

        \Pi_{int} &= \int_V \psi(\boldsymbol{F}) \ dV \qquad \rightarrow \qquad \Pi_{int}(\boldsymbol{u},p,J) = \int_V \psi(\overline{\boldsymbol{F}}) \ dV + \int_V p (J-\overline{J}) \ dV

        \overline{\boldsymbol{F}} &= \left(\frac{\overline{J}}{J}\right)^{1/3} \boldsymbol{F}

    The variations of the total potential energy w.r.t.
    :math:`(\boldsymbol{u},p,J)` lead to the following expressions. We denote
    first partial derivatives as :math:`\boldsymbol{f}_{(\bullet)}` and second
    partial derivatives as :math:`\boldsymbol{A}_{(\bullet,\bullet)}`.

    ..  math::

        \delta_{\boldsymbol{u}} \Pi_{int} &= \int_V \boldsymbol{f}_{\boldsymbol{u}} : \delta \boldsymbol{F} \ dV = \int_V \left( \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} : \frac{\partial \overline{\boldsymbol{F}}}{\partial \boldsymbol{F}} + p J \boldsymbol{F}^{-T} \right) : \delta \boldsymbol{F} \ dV

        \delta_{p} \Pi_{int} &= \int_V f_{p} \ \delta p \ dV = \int_V (J - \overline{J}) \ \delta p \ dV

        \delta_{\overline{J}} \Pi_{int} &= \int_V f_{\overline{J}} \ \delta \overline{J} \ dV = \int_V \left( \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} : \frac{\partial \overline{\boldsymbol{F}}}{\partial \overline{J}} - p \right) : \delta \overline{J} \ dV

    The projection tensors from the variations lead the following results.

    ..  math::

        \frac{\partial \overline{\boldsymbol{F}}}{\partial \boldsymbol{F}} &= \left(\frac{\overline{J}}{J}\right)^{1/3} \left( \boldsymbol{I} \overset{ik}{\odot} \boldsymbol{I} - \frac{1}{3} \boldsymbol{F} \otimes \boldsymbol{F}^{-T} \right)

        \frac{\partial \overline{\boldsymbol{F}}}{\partial \overline{J}} &= \frac{1}{3 \overline{J}} \overline{\boldsymbol{F}}

    The double-dot products from the variations are now evaluated.

    ..  math::

        \overline{\boldsymbol{P}} &= \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} = \overline{\overline{\boldsymbol{P}}} - \frac{1}{3} \left(  \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \qquad \text{with} \qquad \overline{\overline{\boldsymbol{P}}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}}

        \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} : \frac{1}{3 \overline{J}} \overline{\boldsymbol{F}} &= \frac{1}{3 \overline{J}} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F}

    We now have three formulas; one for the first Piola Kirchhoff stress and
    two additional control equations.

    ..  math::

        \boldsymbol{f}_{\boldsymbol{u}} (= \boldsymbol{P}) &= \overline{\overline{\boldsymbol{P}}} - \frac{1}{3} \left(  \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T}

        f_p &= J - \overline{J}

        f_{\overline{J}} &=  \frac{1}{3 \overline{J}} \left( \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) - p

    A linearization of the above formulas gives six equations (only results are
    given here).

    ..  math::

        \mathbb{A}_{\boldsymbol{u},\boldsymbol{u}} &=  \overline{\overline{\mathbb{A}}} + \frac{1}{9} \left(  \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \frac{1}{3} \left( \boldsymbol{F}^{-T} \otimes \left( \overline{\overline{\boldsymbol{P}}} + \boldsymbol{F} : \overline{\overline{\mathbb{A}}} \right) + \left( \overline{\overline{\boldsymbol{P}}} + \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) \otimes \boldsymbol{F}^{-T} \right)

        &+\left( p J + \frac{1}{9} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \left( p J - \frac{1}{3} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \overset{il}{\odot} \boldsymbol{F}^{-T}

        A_{p,p} &= 0

        A_{\overline{J},\overline{J}} &= \frac{1}{9 \overline{J}^2} \left( \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) - 2 \left( \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right)

        \boldsymbol{A}_{\boldsymbol{u},p} &= \boldsymbol{A}_{p, \boldsymbol{u}} = J \boldsymbol{F}^{-T}

        \boldsymbol{A}_{\boldsymbol{u},\overline{J}} &= \boldsymbol{A}_{\overline{J}, \boldsymbol{u}} = \frac{1}{3 \overline{J}} \left( \boldsymbol{P}' + \boldsymbol{F} : \overline{\overline{\mathbb{A}}} - \frac{1}{3} \left( \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \right)

        A_{p,\overline{J}} &= A_{\overline{J}, p} = -1


    with

    ..  math::

        \overline{\overline{\mathbb{A}}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \frac{\partial^2 \psi}{\partial \overline{\boldsymbol{F}} \partial \overline{\boldsymbol{F}}} \left(\frac{\overline{J}}{J}\right)^{1/3}

    as well as

    ..  math::

        \boldsymbol{P}' = \boldsymbol{P} - p J \boldsymbol{F}^{-T}

    Arguments
    ---------
    material : Material
        A material definition with ``gradient`` and ``hessian`` methods.

    Attributes
    ----------
    fun_P : function
        Method for gradient evaluation
    fun_A : function
        Method for hessian evaluation
    detF : ndarray
        Determinant of deformation gradient
    iFT : ndarray
        Transpose of inverse of the deformation gradient
    Fb : ndarray
        Determinant-modified deformation gradient
    Pb : ndarray
        First Piola-Kirchhoff stress tensor (in determinant-modified framework)
    Pbb : ndarray
        Determinant-modification multiplied by ``Pb``
    PbbF : ndarray
        Double-dot product of ``Pb`` and the deformation gradient

    """

    def __init__(self, material, parallel=False):

        self.fun_P = material.gradient
        self.fun_A = material.hessian

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

        return self.Pbb - self.PbbF / 3 * self.iFT + p * self.detF * self.iFT

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

        return self.detF - J

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

        return self.PbbF / (3 * J) - p

    def gradient(self, x):
        r"""List of variations of total potential energy w.r.t
        displacements, pressure and volume ratio.

        ..  code-block::

            δ_u(Π_int) = ∫_V (∂ψ/∂F + p cof(F)) : δF dV
            δ_p(Π_int) = ∫_V (det(F) - J) δp dV
            δ_J(Π_int) = ∫_V (∂U/∂J - p) δJ dV

        Arguments
        ---------
        x : list of ndarray
            List of extracted field values with Deformation gradient ``F``
            as first, the hydrostatic pressure ``p`` as second and the
            volume ratio ``J`` as third item.

        Returns
        -------
        list of ndarrays
            List of gradients w.r.t. the input variables F, p and J

        """

        [F, p, J], statevars = x[:3], x[-1]

        self.detF = det(F)
        self.iFT = transpose(inv(F))
        self.Fb = (J / self.detF) ** (1 / 3) * F
        self.Pb, statevars_new = self.fun_P([self.Fb, statevars])
        self.Pbb = (J / self.detF) ** (1 / 3) * self.Pb
        self.PbbF = ddot(self.Pbb, F, parallel=self.parallel)

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

        self.detF = det(F)
        self.iFT = transpose(inv(F))
        self.Fb = (J / self.detF) ** (1 / 3) * F
        self.Pbb = (J / self.detF) ** (1 / 3) * self.fun_P([self.Fb, statevars])[0]

        self.eye = identity(F)
        self.P4 = cdya_ik(self.eye, self.eye, parallel=self.parallel) - 1 / 3 * dya(
            F, self.iFT, parallel=self.parallel
        )
        self.A4b = self.fun_A([self.Fb, statevars])[0]
        self.A4bb = (J / self.detF) ** (2 / 3) * self.A4b

        self.PbbF = ddot(self.Pbb, F, parallel=self.parallel)
        self.FA4bb = ddot(F, self.A4bb, parallel=self.parallel)
        self.A4bbF = ddot(self.A4bb, F, parallel=self.parallel)
        self.FA4bbF = ddot(F, self.A4bbF, parallel=self.parallel)

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

        PbbA4bbF = self.Pbb + self.A4bbF
        PbbFA4bb = self.Pbb + self.FA4bb

        pJ9 = p * self.detF + self.PbbF / 9
        pJ3 = p * self.detF - self.PbbF / 3

        A4 = (
            self.A4bb
            + self.FA4bbF * dya(self.iFT, self.iFT, parallel=self.parallel) / 9
            - (
                dya(PbbA4bbF, self.iFT, parallel=self.parallel)
                + dya(self.iFT, PbbFA4bb, parallel=self.parallel)
            )
            / 3
            + pJ9 * dya(self.iFT, self.iFT, parallel=self.parallel)
            - pJ3 * cdya_il(self.iFT, self.iFT, parallel=self.parallel)
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

        return (self.FA4bbF - 2 * self.PbbF) / (9 * J**2)

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

        return self.detF * self.iFT

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
        return (-self.FA4bbF / 3 * self.iFT + Ps + self.FA4bb) / (3 * J)

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
