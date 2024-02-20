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

from ..assembly import IntegralForm, IntegralFormCartesian
from ..constitution import AreaChange
from ..field import FieldAxisymmetric, FieldDual
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


class StateNearlyIncompressibleX:
    "A State with internal fields for (nearly) incompressible solid bodies."

    def __init__(self, field, pressure=None, volume_ratio=None, **kwargs):
        self.field = field
        self.u = self.field[0].values

        self.pressure = pressure
        self.volume_ratio = volume_ratio

        if self.pressure is None:
            self.pressure = FieldDual(field.region, **kwargs)
        if self.volume_ratio is None:
            self.volume_ratio = FieldDual(field.region, values=1, **kwargs)

        self.dJdF = AreaChange().function

        # deformation gradient
        self.F = field.extract()

        # inverse of volume matrix
        self.inv_V = np.linalg.inv(self.volume().T).T
        self.h = self.integrate_shape_function_gradient()

    def integrate_shape_function_gradient(self, parallel=False):
        r"""Return the Integrated shape function gradient matrix w.r.t. the deformed
        coordinates.

        Notes
        -----
        ..  math::

            h = \int_V \delta \boldsymbol{F} : \frac{\partial J}{\partial\boldsymbol{F}}
                \ \Delta p ~ dV

        """
        self.h = IntegralForm(
            fun=self.dJdF(self.F),
            v=self.field,
            u=self.pressure & None,
            dV=self.field.region.dV,
            grad_v=[True],
            grad_u=[False],
        ).integrate(parallel=parallel)[0]
        return self.h

    def volume(self, parallel=False):
        r"""Return integrated differential (undeformed) volumes matrix with dual-trial
        and dual-test fields.

        Notes
        -----
        ..  math::

            V = \int_V \delta p \Delta p ~ dV

        """
        return IntegralForm(
            fun=[np.ones((1, 1))],
            v=self.pressure & None,
            dV=self.field.region.dV,
            u=self.pressure & None,
            grad_v=[False],
            grad_u=[False],
        ).integrate(parallel=parallel)[0]

    def fp(self, parallel=False):
        dV = self.field.region.dV
        J = self.volume_ratio.interpolate()
        v = self.pressure & None
        return IntegralForm(
            fun=[det(self.F[0]) - J], v=v, dV=dV, grad_v=[False]
        ).integrate(parallel=parallel)[0]

    def fJ(self, bulk, parallel=False):
        dV = self.field.region.dV
        p = self.pressure.interpolate()
        J = self.volume_ratio.interpolate()
        v = self.pressure & None
        return IntegralForm(
            fun=[bulk * (J - 1) - p], v=v, dV=dV, grad_v=[False]
        ).integrate(parallel=parallel)[0]

    def constraint(self, bulk, parallel=False):
        dV = self.field.region.dV
        p = self.pressure.interpolate()
        detF = det(self.F[0])
        v = self.pressure & None
        return IntegralForm(
            fun=[bulk * (detF - 1) - p], v=v, dV=dV, grad_v=[False]
        ).integrate(parallel=parallel)[0]
