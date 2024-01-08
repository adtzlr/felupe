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

        self.force_values = None
        self.stiffness_values = None

        if stress:
            self.stress = None

        if elasticity:
            self.elasticity = None

    def update_statevars(self):
        if self._statevars is not None:
            self.statevars = self._statevars


class StateNearlyIncompressible:
    "A State with internal fields for (nearly) incompressible solid bodies."

    def __init__(self, field):
        self.field = field
        self.dJdF = AreaChange().function

        # initial values (on mesh-points) of the displacement field
        self.u = field[0].values

        # deformation gradient
        self.F = field.extract()

        # cell-values of the internal pressure and volume-ratio fields
        self.p = np.zeros(field.region.mesh.ncells)
        self.J = np.ones(field.region.mesh.ncells)

    def h(self, parallel=False):
        r"""Integrated shape-function gradient w.r.t. the deformed coordinates.

        ..  math::

            \int_V \frac{\partial J}{\partial \boldsymbol{F}} :
                \delta \boldsymbol{F} ~ dV

        """

        return IntegralForm(
            fun=self.dJdF(self.F), v=self.field, dV=self.field.region.dV
        ).integrate(parallel=parallel)[0]

    def v(self):
        r"""Cell volumes of the deformed configuration.

        ..  math::

            v = \int_V J ~ dV

        """
        dV = self.field.region.dV
        if isinstance(self.field[0], FieldAxisymmetric):
            R = self.field[0].radius
            dA = self.field.region.dV
            dV = 2 * np.pi * R * dA
        return (det(self.F[0]) * dV).sum(0)
