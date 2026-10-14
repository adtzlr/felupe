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

import warnings

import numpy as np
from scipy.sparse import csr_matrix

from ..assembly import IntegralForm
from ..constitution import AreaChange
from ..field import FieldAxisymmetric
from ..math import det, dot, transpose


class Assemble:
    "A class with methods for assembling vectors and matrices of an Item."

    def __init__(self, vector=None, matrix=None, mass=None, multiplier=None):

        self.vector = self._empty_matrix
        self.matrix = self._empty_matrix
        self.mass = mass
        self.multiplier = multiplier

        if vector is not None:
            self.vector = vector

        if matrix is not None:
            self.matrix = matrix

    def _empty_matrix(self, field=None, **kwargs):
        return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))


class Evaluate:
    "A class with evaluate methods of an Item."

    def __init__(
        self, gradient, hessian, stress=None, cauchy_stress=None, kirchhoff_stress=None
    ):
        self.gradient = gradient
        self.hessian = hessian

        if cauchy_stress is not None:
            self.cauchy_stress = cauchy_stress
            self.kirchhoff_stress = kirchhoff_stress

        if stress is None:
            stress = lambda field=None, **kwargs: gradient(field, **kwargs)[0]

        self.stress = stress


class Results:
    "A class with intermediate results of an Item."

    def __init__(self, stress=False, elasticity=False):
        self.force = None
        self._force = None
        self.stiffness = None
        self.kinematics = None
        self.statevars = None
        self._statevars = None

        self.gradient = None
        self.hessian = None

        self.force_values = None
        self.stiffness_values = None

        self._force_values = None
        self._stiffness_values = None

        if stress:
            self.stress = None

        if elasticity:
            self.elasticity = None

    def update_statevars(self):
        if self._statevars is not None:
            self.statevars = self._statevars


def kirchhoff_stress(P, F):
    r"""Return the Kirchhoff stress tensor for given first Piola-Kirchhoff stress and
    deformation gradient tensors.

    ..  math::

        \boldsymbol{\tau} = \boldsymbol{P} \boldsymbol{F}^T

    The tensors are given by a solid body, which defines the stress tensor of its
    formulation, e.g. including the contribution of an internal pressure field.
    """

    return dot(P, transpose(F))


def cauchy_stress(P, F):
    r"""Return the Cauchy stress tensor for given first Piola-Kirchhoff stress and
    deformation gradient tensors.

    ..  math::

        \boldsymbol{\sigma} = \frac{1}{J} \boldsymbol{P} \boldsymbol{F}^T

    For two-dimensional tensors, the volume ratio :math:`J` is not available and the
    Kirchhoff stress tensor is returned (with a warning).
    """

    if P.shape[:2] == (2, 2):
        warnings.warn(
            "\n".join(
                [
                    "Cauchy stress tensor can't be evaluated on a 2d-Field.",
                    "Falling-back to the Kirchhoff stress tensor.",
                ]
            )
        )
        J = 1
    else:
        J = det(F)

    return dot(P, transpose(F)) / J


def mass_matrix(field, density, form=IntegralForm):
    r"""Return the assembled mass matrix of the first field of a field container.

    ..  math::

        \int_V \rho\ \delta \boldsymbol{u} \cdot \Delta \boldsymbol{u}\ dV

    The integral form of a vector-valued axisymmetric field takes a function with an
    additional out-of-plane component, which is multiplied by :math:`u_r / R` (like the
    circumferential stretch). The displacements of an axisymmetric field have no
    circumferential component, i.e. the out-of-plane component of the function is zero
    and the mass matrix is the in-plane mass matrix of the volumes of the rings
    :math:`dV = 2 \pi R\ dA`. A scalar axisymmetric field, e.g. the temperature of a
    :class:`~felupe.thermal.SolidBodyThermal`, has no out-of-plane component.
    """

    field = field[0].as_container()
    identity = np.eye(field[0].dim)

    if isinstance(field[0], FieldAxisymmetric) and field[0].dim > 1:
        identity = np.pad(identity, (0, 1))  # zero out-of-plane component

    integral = form(
        fun=[density * identity.reshape(*identity.shape, 1, 1)],
        v=field,
        u=field,
        dV=field.region.dV,
        grad_v=[False],
        grad_u=[False],
    )

    return integral.assemble()


class StateNearlyIncompressible:
    r"""A State with internal cell-wise constant dual fields for (nearly) incompressible
    solid bodies.

    Notes
    -----
    The internal fields :math:`p` and :math:`\bar{J}` are treated as internal fields, directly derived
    from the displacement field. Hence, these fields are not exported to the global degrees of freedom.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field.

    See Also
    --------
    felupe.SolidBodyNearlyIncompressible : A (nearly) incompressible solid body with
        methods for the assembly of sparse vectors/matrices.
    """

    def __init__(self, field):
        self.field = field
        self.dJdF = AreaChange().function
        self.v = None

        # initial values (on mesh-points) of the displacement field
        self.u = field[0].values.copy()

        # deformation gradient
        self.F = field.extract()
        self.detF = None

        # cell-values of the internal pressure and volume-ratio fields
        self.p = np.zeros(field.region.mesh.ncells)
        self.J = np.ones(field.region.mesh.ncells)

    def integrate_shape_function_gradient(self, parallel=False, out=None):
        r"""Integrated sub-block matrix containing the shape-functions gradient w.r.t.
        the deformed coordinates :math:`\boldsymbol{K}_{\boldsymbol{u}p}`.

        ..  math::

            \int_V \delta \boldsymbol{F} : \frac{\partial J}{\partial \boldsymbol{F}}
                ~ dV\ \Delta p
            \longrightarrow \boldsymbol{K}_{\boldsymbol{u}p}

        """

        return IntegralForm(
            fun=self.dJdF(self.F), v=self.field, dV=self.field.region.dV
        ).integrate(parallel=parallel, out=out)[0]

    def volume(self):
        r"""Return the cell volumes of the deformed configuration.

        ..  math::

            v = \int_V J ~ dV

        """
        dV = self.field.region.dV
        if isinstance(self.field[0], FieldAxisymmetric):
            R = self.field[0].radius
            dA = self.field.region.dV
            dV = 2 * np.pi * R * dA
        dv = det(self.F[0]) * dV
        return dv.sum(0, out=self.v)
