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

from ..field._axi import FieldAxisymmetric
from ..field._base import Field
from ._cartesian import IntegralFormCartesian


class IntegralFormAxisymmetric(IntegralFormCartesian):
    r"""An Integral Form for axisymmetric fields.

    Notes
    -----
    Axisymmetric scenarios are modeled with a 2D-mesh and consequently, a 2D element
    formulation. The rotation axis is chosen along the global X-axis
    :math:`(X,Y,Z) \widehat{=} (Z,R,\varphi)`. The 3x3 deformation gradient consists of
    an in-plane 2x2 sub-matrix and one additional entry for the out-of-plane stretch
    which is equal to the ratio of deformed and undeformed radius.

    ..  math::

        \boldsymbol{F} = \begin{bmatrix}
                \boldsymbol{F}_{(2D)} & \boldsymbol{0} \\
                \boldsymbol{0}^T & \frac{r}{R}
            \end{bmatrix}

    The variation of the deformation gradient consists of both in- and out-of-plane
    contributions.

    ..  math::

        \delta \boldsymbol{F}_{(2D)} = \delta \frac{
                \partial \boldsymbol{u}}{\partial \boldsymbol{X}
            } \qquad \text{and} \qquad \delta \left(\frac{r}{R}\right)
            = \frac{\delta u_r}{R}

    Again, the internal virtual work leads to two seperate terms.

    ..  math::

        -\delta W_{int} = \int_V \boldsymbol{P} : \delta \boldsymbol{F} \ dV
            = \int_V \boldsymbol{P}_{(2D)} : \delta \boldsymbol{F}_{(2D)} \ dV
            + \int_V \frac{P_{33}}{R} : \delta u_r \ dV

    The differential volume is further expressed as a product of the differential
    in-plane area and the differential arc length. The arc length integral is finally
    pre-evaluated.

    ..  math::

        \int_V dV = \int_{\varphi=0}^{2\pi} \int_A R\ dA\ d\varphi = 2\pi \int_A R\ dA

    Inserting the differential volume integral into the expression of internal virtual
    work, this leads to:

    ..  math::

        -\delta W_{int} = 2\pi \int_A \boldsymbol{P}_{(2D)}
            : \delta \boldsymbol{F}_{(2D)} \ R \ dA
            + 2\pi \int_A P_{33} : \delta u_r \ dA

    A Linearization of the internal virtual work expression gives four terms.

    ..  math::

        -\Delta \delta W_{int} &= \Delta_{(2D)} \delta_{(2D)} W_{int}
            + \Delta_{33} \delta_{(2D)} W_{int} + \Delta_{(2D)} \delta_{33} W_{int}
            + \Delta_{33} \delta_{33} W_{int}

        -\Delta_{(2D)} \delta_{(2D)} W_{int} &= 2\pi \int_A
            \delta \boldsymbol{F}_{(2D)} : \mathbb{A}_{(2D),(2D)} :
                \Delta \boldsymbol{F}_{(2D)} \ R \ dA

        -\Delta_{33} \delta_{(2D)} W_{int} &= 2\pi \int_A
            \delta \boldsymbol{F}_{(2D)} : \mathbb{A}_{(2D),33} : \Delta u_r \ dA

        -\Delta_{(2D)} \delta_{33} W_{int} &= 2\pi \int_A
            \delta u_r : \mathbb{A}_{33,(2D)} : \Delta \boldsymbol{F}_{(2D)} \ dA

        -\Delta_{33} \delta_{33} W_{int} &= 2\pi \int_A
            \delta u_r : \frac{\mathbb{A}_{33,33}}{R} : \Delta u_r \ dA

    with

    ..  math::

        \mathbb{A}_{(2D),(2D)} &= \frac{
            \partial \psi}{\partial \boldsymbol{F}_{(2D)} \partial \boldsymbol{F}_{(2D)}
        }

        \mathbb{A}_{(2D),33} &= \frac{
            \partial \psi}{\partial \boldsymbol{F}_{(2D)} \partial F^3_{\hphantom{3}3}}
            \left ( = \mathbb{A}_{33,(2D)} \right )

        \mathbb{A}_{33,33} &= \frac{\partial \psi}{F^3_{\hphantom{3}3}
            \partial F^3_{\hphantom{3}3}}

    See Also
    --------
    felupe.IntegralForm : Mixed-field integral form container with methods for integration and assembly.
    felupe.IntegralFormCartesian : Single-field integral form.

    """

    def __init__(self, fun, v, dV, u=None, grad_v=True, grad_u=True):
        R = v.radius
        self.dV = 2 * np.pi * R * dV

        if u is None:
            if isinstance(v, FieldAxisymmetric):
                self.mode = 1

                if grad_v:
                    fun_2d = fun[:-1, :-1]
                    fun_zz = fun[(-1,), (-1,)] / R
                else:
                    fun_2d = fun[:-1]
                    fun_zz = fun[-1].reshape(1, *fun[-1].shape) / R

                form_a = IntegralFormCartesian(fun_2d, v, self.dV, grad_v=grad_v)
                form_b = IntegralFormCartesian(fun_zz, v.scalar, self.dV)

                self.forms = [form_a, form_b]

            else:
                self.mode = 10

                form_a = IntegralFormCartesian(fun, v, self.dV, grad_v=False)
                self.forms = [
                    form_a,
                ]

        else:
            if isinstance(v, FieldAxisymmetric) and isinstance(u, FieldAxisymmetric):
                self.mode = 2

                if grad_v and grad_u:
                    form_aa = IntegralFormCartesian(
                        fun[:-1, :-1, :-1, :-1], v, self.dV, u, True, True
                    )
                    form_bb = IntegralFormCartesian(
                        fun[-1, -1, -1, -1] / R**2,
                        v.scalar,
                        self.dV,
                        u.scalar,
                        False,
                        False,
                    )
                    form_ba = IntegralFormCartesian(
                        fun[-1, -1, :-1, :-1] / R, v.scalar, self.dV, u, False, True
                    )
                    form_ab = IntegralFormCartesian(
                        fun[:-1, :-1, -1, -1] / R, v, self.dV, u.scalar, True, False
                    )

                if not grad_v and grad_u:
                    form_aa = IntegralFormCartesian(
                        fun[:-1, :-1, :-1], v, self.dV, u, False, True
                    )
                    form_bb = IntegralFormCartesian(
                        fun[-1, -1, -1] / R**2,
                        v.scalar,
                        self.dV,
                        u.scalar,
                        False,
                        False,
                    )
                    form_ba = IntegralFormCartesian(
                        fun[-1, :-1, :-1] / R, v.scalar, self.dV, u, False, True
                    )
                    form_ab = IntegralFormCartesian(
                        fun[:-1, -1, -1] / R, v, self.dV, u.scalar, False, False
                    )

                self.forms = [form_aa, form_bb, form_ba, form_ab]

            elif isinstance(v, FieldAxisymmetric) and isinstance(u, Field):
                self.mode = 30

                form_a = IntegralFormCartesian(
                    fun[:-1, :-1], v, self.dV, u, True, False
                )
                form_b = IntegralFormCartesian(
                    fun[-1, -1] / R, v.scalar, self.dV, u, False, False
                )

                self.forms = [form_a, form_b]

            elif isinstance(v, Field) and isinstance(u, Field):
                self.mode = 40

                form_a = IntegralFormCartesian(fun, v, self.dV, u, False, False)

                self.forms = [
                    form_a,
                ]

    def integrate(self, parallel=False, out=None):
        values = [form.integrate(parallel=parallel) for form in self.forms]

        if self.mode == 1:
            values[0] += np.pad(values[1], ((0, 0), (1, 0), (0, 0)))
            val = values[0]

        if self.mode == 30:
            if len(values[0].shape) > 4:
                values[0] = values[0][:, :, 0, 0]
            if len(values[1].shape) > 4:
                values[1] = values[1][:, :, 0, 0]
            a, b, e = values[1].shape
            values[1] = values[1].reshape(a, 1, b, e)

            values[0] += np.pad(values[1], ((0, 0), (1, 0), (0, 0), (0, 0)))
            val = values[0]

        elif self.mode == 2:
            a, b, e = values[1].shape
            values[1] = values[1].reshape(a, 1, b, 1, e)
            values[1] = np.pad(values[1], ((0, 0), (1, 0), (0, 0), (1, 0), (0, 0)))

            a, b, i, e = values[2].shape
            values[2] = values[2].reshape(a, 1, b, i, e)
            values[2] = np.pad(values[2], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))

            a, i, b, e = values[3].shape
            values[3] = values[3].reshape(a, i, b, 1, e)
            values[3] = np.pad(values[3], ((0, 0), (0, 0), (0, 0), (1, 0), (0, 0)))

            for i in range(1, len(values)):
                values[0] += values[i]

            val = values[0]

        elif self.mode == 10 or self.mode == 40:
            val = values[0]

        return val

    def assemble(self, values=None, parallel=False, out=None):
        if values is None:
            values = self.integrate(parallel=parallel, out=out)
        return self.forms[0].assemble(values)
