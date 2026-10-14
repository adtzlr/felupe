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

from typing import NamedTuple

import numpy as np

from ..field import Field, FieldAxisymmetric
from ._cartesian import IntegralFormCartesian


class _Part(NamedTuple):
    "A part of a field in an axisymmetric integral form."

    field: Field
    grad: bool
    axes: tuple
    components: slice
    out_of_plane: int


def _parts(field, grad):
    """Return the parts of a field in an axisymmetric integral form. The axes of the
    function belong to the three-dimensional values or gradients of an axisymmetric
    field (the out-of-plane component is the last one, see the notes of
    :class:`IntegralFormAxisymmetric`). The parts are evaluated by cartesian integral
    forms. Their results have an axis of the components of each field, which is
    added to the given components of the field.
    """

    if isinstance(field, FieldAxisymmetric):
        n = 1 + int(grad)
        return [
            # the in-plane values or gradients
            _Part(field, grad, (slice(-1),) * n, slice(None), 0),
            # the out-of-plane value or gradient v_r / R, i.e. the radial component
            _Part(
                field.scalar,
                False,
                (slice(-1, None),) + (-1,) * int(grad),
                slice(1, 2),
                1,
            ),
        ]

    # all values or gradients of a field without an out-of-plane component, with an
    # additional axis for the components of a scalar field
    axes = (slice(None),) if field.dim > 1 else (None,)
    return [_Part(field, grad, axes + (slice(None),) * int(grad), slice(None), 0)]


class IntegralFormAxisymmetric(IntegralFormCartesian):
    r"""An Integral Form for axisymmetric fields.

    Parameters
    ----------
    fun : array
        The pre-evaluated function array. If the array is not contiguous, a C-contiguous
        copy is made.
    v : FieldAxisymmetric
        The axisymmetric test field.
    dV : array
        The differential volumes.
    u : FieldAxisymmetric, optional
        If a field is passed, a bilinear form is created (default is None).
    grad_v : bool, optional
        Flag to activate the gradient on the test field ``v`` (default is False).
    grad_u : bool, optional
        Flag to activate the gradient on the trial field ``u`` (default is False).

    Notes
    -----
    Axisymmetric integral forms are created with a two-dimensional mesh, a two-
    dimensional element formulation and an axisymmetric field.

    ..  note::

        The rotation axis is chosen along the global X-axis
        :math:`(X,Y,Z) \widehat{=} (Z,R,\varphi)`.

    The three-dimensional deformation gradient consists of four in-plane components and
    one additional non-zero entry for the out-of-plane stretch, which is equal to the
    ratio of the deformed and the undeformed radius.

    ..  math::

        \boldsymbol{F} = \begin{bmatrix}
                \boldsymbol{F}_{(2D)} & \boldsymbol{0} \\
                \boldsymbol{0}^T & \frac{r}{R}
            \end{bmatrix}

    The variation of the deformation gradient consists of both in- and out-of-plane
    contributions.

    ..  math::

        \delta \boldsymbol{F}_{(2D)} = \delta \left( \frac{
                \partial \boldsymbol{u}}{\partial \boldsymbol{X}
            } \right) \qquad \text{and} \qquad \delta \left(\frac{r}{R}\right)
            = \frac{\delta u_r}{R}

    Again, the internal virtual work leads to two separate terms.

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

        \mathbb{A}_{(2D)~(2D)} &= \frac{
            \partial \psi}{\partial \boldsymbol{F}_{(2D)}~\partial \boldsymbol{F}_{(2D)}
        }

        \mathbb{A}_{(2D)~33} &= \frac{
            \partial \psi}{\partial \boldsymbol{F}_{(2D)}\ \partial F_{33}}
            \left ( = \mathbb{A}_{33~(2D)} \right )

        \mathbb{A}_{33~33} &= \frac{\partial \psi}{\partial F_{33}~\partial F_{33}}

    See Also
    --------
    felupe.IntegralForm : Mixed-field integral form container with methods for
        integration and assembly.
    felupe.assembly.IntegralFormCartesian : Single-field integral form.

    """

    def __init__(self, fun, v, dV, u=None, grad_v=True, grad_u=True):
        R = v.radius
        self.dV = 2 * np.pi * R * dV

        if v.dim == 1:
            self.is_scalar = True
            self._init_scalar(fun, v, self.dV, u=u, grad_v=grad_v, grad_u=grad_u)
        else:
            self.is_scalar = False
            self._init_vector(fun, v, self.dV, u=u, grad_v=grad_v, grad_u=grad_u)

    def _init_scalar(self, fun, v, dV, u, grad_v, grad_u):
        self.forms = [
            IntegralFormCartesian(fun, v, self.dV, u=u, grad_v=grad_v, grad_u=grad_u)
        ]

    def _init_vector(self, fun, v, dV, u, grad_v, grad_u):
        self.components = None
        self.squeeze = False

        if u is not None and fun is None:
            # an empty (None) block of a bilinear form, e.g. a zero-valued coupling
            # between two fields, is handled by the cartesian integral form
            self.forms = [IntegralFormCartesian(None, v, dV, u, False, False)]
            return

        if not isinstance(v, FieldAxisymmetric):
            # a test field without an out-of-plane component
            self.forms = [IntegralFormCartesian(fun, v, dV, u, grad_v, grad_u)]
            return

        # the integral form is the sum of the cartesian integral forms of all pairs of
        # the in-plane and the out-of-plane parts of the test and the trial field
        R = v.radius
        trial_parts = [None] if u is None else _parts(u, grad_u)

        # the axis of the components of a scalar trial field is removed from the result
        self.squeeze = u is not None and u.dim == 1

        self.forms = []
        self.components = []

        for test in _parts(v, grad_v):
            for trial in trial_parts:
                if trial is None:  # linear form
                    f = fun[test.axes]
                    out_of_plane = test.out_of_plane
                    components = (slice(None), test.components)
                else:
                    f = fun[test.axes + trial.axes]
                    out_of_plane = test.out_of_plane + trial.out_of_plane
                    components = (
                        slice(None),
                        test.components,
                        slice(None),
                        trial.components,
                    )

                # the out-of-plane parts are divided by the radius, i.e. v_r / R
                if out_of_plane > 0:
                    f = f / R**out_of_plane

                if trial is None:
                    form = IntegralFormCartesian(f, test.field, dV, grad_v=test.grad)
                else:
                    form = IntegralFormCartesian(
                        f, test.field, dV, trial.field, test.grad, trial.grad
                    )

                self.forms.append(form)
                self.components.append(components)

    def integrate(self, parallel=False, out=None):
        if self.is_scalar:
            return self._integrate_scalar(parallel=parallel, out=out)
        else:
            return self._integrate_vector(parallel=parallel, out=out)

    def _integrate_scalar(self, parallel=False, out=None):
        return self.forms[0].integrate(parallel=parallel, out=out)

    def _integrate_vector(self, parallel=False, out=None):
        # the first form is the in-plane part (or the whole form) with the shape of the
        # result, the out-of-plane parts are added to the radial components
        values = self.forms[0].integrate(parallel=parallel)

        if self.components is not None:
            for form, components in zip(self.forms[1:], self.components[1:]):
                values[components] += form.integrate(parallel=parallel)

        if self.squeeze:
            a, i, b, _, c = values.shape
            values = values.reshape(a, i, b, c)

        return values

    def assemble(self, values=None, parallel=False, out=None):
        if values is None:
            values = self.integrate(parallel=parallel, out=out)
        return self.forms[0].assemble(values)
