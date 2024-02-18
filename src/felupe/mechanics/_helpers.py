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
        self.displacement = self.field[0].copy()
        self.pressure = FieldDual(field.region)
        self.volume_ratio = FieldDual(field.region, values=1)
        self.dJdF = AreaChange().function

        # deformation gradient
        self.F = field.extract()

        # displacements u, (dual) pressure p and (dual) volume-ratio J
        # field-values at mesh-point p per cell c: u_pic, p_pc, J_pc
        # mesh = self.field.region.mesh
        # self.u = field[0].values[mesh.cells].transpose([1, 2, 0]).copy()
        # self.p = np.zeros_like(self.field_dual.values[mesh.cells]).transpose([1, 2, 0])
        # self.J = np.ones_like(self.field_dual.values[mesh.cells]).transpose([1, 2, 0])
        

    def int_tr_dhudx_hp_dv(self, parallel=False):
        r"""Symmetric sub-block integrated (but not assembled) matrix values.

        Notes
        -----
        ..  math::

            \int_V \delta \boldsymbol{F} : \frac{\partial J}{\partial \boldsymbol{F}}
                \ \Delta p ~ dV

        """
        return IntegralForm(
            fun=self.dJdF(self.F),
            v=self.field,
            u=self.pressure & None,
            dV=self.field.region.dV,
            grad_v=[True],
            grad_u=[False],
        ).integrate(parallel=parallel)[0]

    def int_hJ_hJ_dv(self, parallel=False):
        r"""Symmetric sub-block integrated (but not assembled) matrix values.

        Notes
        -----
        ..  math::

            \int_V \delta \bar{J} \Delta \bar{J} ~ dV

        """
        return IntegralFormCartesian(
            fun=np.ones((1, 1)),
            v=self.volume_ratio,
            dV=self.field.region.dV,
            u=self.volume_ratio,
            grad_v=False,
            grad_u=False,
        ).integrate(parallel=parallel)

    def fp(self, parallel=False):
        dV = self.field.region.dV
        J = self.volume_ratio.interpolate()
        return IntegralFormCartesian(
            fun=det(self.F[0]) - J, v=self.pressure, dV=dV, grad_v=False
        ).integrate(parallel=parallel)[0]

    def fJ(self, bulk, parallel=False):
        dV = self.field.region.dV
        p = self.pressure.interpolate()
        return IntegralFormCartesian(
            fun=-p + bulk, v=self.volume_ratio, dV=dV, grad_v=False
        ).integrate(parallel=parallel)[0]