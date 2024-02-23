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

from ..assembly import IntegralForm
from ..constitution import AreaChange
from ..field import FieldAxisymmetric
from ..math import det


class Assemble:
    "A class with assembly methods of a SolidBody."

    def __init__(self, vector, matrix):
        self.vector = vector
        self.matrix = matrix


class Evaluate:
    "A class with evaluate methods of a SolidBody."

    def __init__(self, gradient, hessian, cauchy_stress=None, kirchhoff_stress=None):
        self.gradient = gradient
        self.hessian = hessian

        if cauchy_stress is not None:
            self.cauchy_stress = cauchy_stress
            self.kirchhoff_stress = kirchhoff_stress


class Results:
    "A class with intermediate results of a SolidBody."

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


class StateNearlyIncompressible:
    r"""A State with internal cell-wise constant dual fields for (nearly) incompressible
    solid bodies.

    Notes
    -----
    The internal fields :math:`p` and :math:`\bar{J}` are treated as state variables,
    directly derived from the displacement field. Hence, these dual fields are not
    exported to the global degrees of freedom.

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
        self.u = field[0].values

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
