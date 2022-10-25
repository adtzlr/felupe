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

from .._assembly import IntegralFormMixed
from ..constitution import AreaChange
from ._helpers import Assemble, Results


class SolidBodyPressure:
    "A hydrostatic pressure boundary on a SolidBody."

    def __init__(self, field, pressure=None):

        self.field = field
        self._normals = self.field.region.normals

        self.results = Results()
        self.results.kinematics = self._extract(self.field)

        if pressure is not None:
            self.results.pressure = pressure
        else:
            self.results.pressure = 0

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self._area_change = AreaChange()

    def update(self, pressure):

        self.__init__(self.field, pressure)

    def _extract(self, field):

        self.field = field
        self.results.kinematics = self.field.extract()

        return self.results.kinematics

    def _vector(
        self, field=None, pressure=None, parallel=False, jit=False, resize=None
    ):

        if field is not None:

            self._update(field)
            self.results.kinematics = self._extract(self.field)

        fun = self._area_change.function(
            self.results.kinematics,
            self._normals,
            parallel=parallel,
        )

        if pressure is not None:
            self.results.pressure = pressure

        fun[0] *= self.results.pressure

        self.results.force = IntegralFormMixed(
            fun=fun, v=self.field, dV=self.field.region.dV, grad_v=[False]
        ).assemble(parallel=parallel, jit=jit)

        if resize is not None:
            self.results.force.resize(*resize.shape)

        return self.results.force

    def _matrix(
        self, field=None, pressure=None, parallel=False, jit=False, resize=None
    ):

        if field is not None:

            self._update(field)
            self.results.kinematics = self._extract(self.field)

        fun = self._area_change.gradient(
            self.results.kinematics,
            self._normals,
            parallel=parallel,
        )

        if pressure is not None:
            self.results.pressure = pressure

        fun[0] *= self.results.pressure

        self.results.stiffness = IntegralFormMixed(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
            grad_u=[True],
        ).assemble(parallel=parallel, jit=jit)

        if resize is not None:
            self.results.stiffness.resize(*resize.shape)

        return self.results.stiffness

    def _update(self, other_field, field=None):

        if field is not None:
            self.field = field

        self.field[0].values = other_field[0].values
        self.results.kinematics = self._extract(self.field)

        return self.field
