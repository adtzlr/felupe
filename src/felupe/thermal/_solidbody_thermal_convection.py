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
from ..mechanics import Assemble, Results


class SolidBodyThermalConvection:
    "A thermal solid body for thermal convection on a boundary."

    def __init__(self, field, coefficient, temperature):
        self.field = field
        self._normals = self.field.region.normals

        self.results = Results()
        self.results.temperature = temperature

        if coefficient is not None:
            self.results.coefficient = coefficient

        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )

    def update(self, coefficient):
        self.__init__(self.field, coefficient)

    def _vector(self, field=None, parallel=False, resize=None):
        if field is not None:
            self.field = field

        temperature = self.field.extract(grad=False)[0]
        fun = [-self.results.coefficient * (temperature - self.results.temperature)]

        self.results.force = IntegralForm(
            fun=fun, v=self.field, dV=self.field.region.dV, grad_v=[False]
        ).assemble(parallel=parallel)

        if resize is not None:
            self.results.force.resize(*resize.shape)

        return self.results.force

    def _matrix(self, field=None, parallel=False, resize=None):
        if field is not None:
            self.field = field

        fun = [-self.results.coefficient * np.ones((1, 1))]

        self.results.stiffness = IntegralForm(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
            grad_u=[False],
        ).assemble(parallel=parallel)

        if resize is not None:
            self.results.stiffness.resize(*resize.shape)

        return self.results.stiffness
