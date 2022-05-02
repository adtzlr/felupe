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

from .._field import Field, FieldMixed, FieldsMixed, FieldAxisymmetric
from .._assembly import IntegralForm, IntegralFormMixed, IntegralFormAxisymmetric
from ..constitution import AreaChange
from ._helpers import Assemble, Results


class SolidBodyPressure:
    "A hydrostatic pressure boundary on a SolidBody."

    def __init__(self, field):

        self.field = field

        self._dV = self.field.region.dV
        self._normals = self.field.region.normals

        self.results = Results()
        self.results.kinematics = self._extract(self.field)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self._form = {
            Field: IntegralForm,
            FieldMixed: IntegralFormMixed,
            FieldsMixed: IntegralFormMixed,
            FieldAxisymmetric: IntegralFormAxisymmetric,
        }[type(self.field)]

        self._kwargs = {
            Field: dict(dV=self._dV, grad_v=True, grad_u=True),
            FieldMixed: dict(dV=self._dV),
            FieldsMixed: dict(dV=self._dV),
            FieldAxisymmetric: dict(dV=self._dV, grad_v=True, grad_u=True),
        }[type(self.field)]

        self._IntForm = {
            Field: IntegralForm,
            FieldAxisymmetric: IntegralFormAxisymmetric,
        }[type(self.field)]

        self._area_change = AreaChange()

    def _extract(self, field):

        self.field = field
        self.results.kinematics = (self.field.extract(),)

        return self.results.kinematics

    def _vector(self, field=None, pressure=1, parallel=False, jit=False, resize=None):

        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(field)

        self.results.pressure = pressure

        fun = pressure * self._area_change.function(
            *self.results.kinematics,
            self._normals,
            parallel=parallel,
        )

        self.results.force = self._IntForm(
            fun=fun, v=self.field, dV=self._dV, grad_v=False
        ).assemble(parallel=parallel, jit=jit)

        if resize is not None:
            self.results.force.resize(*resize.shape)

        return self.results.force

    def _matrix(self, field=None, pressure=1, parallel=False, jit=False, resize=None):

        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(field)

        self.results.pressure = pressure

        fun = pressure * self._area_change.gradient(
            *self.results.kinematics,
            self._normals,
            parallel=parallel,
        )
        self.results.stiffness = self._IntForm(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self._dV,
            grad_v=False,
            grad_u=True,
        ).assemble(parallel=parallel, jit=jit)

        if resize is not None:
            self.results.stiffness.resize(*resize.shape)

        return self.results.stiffness

    def update(self, other_field, field=None):

        if field is not None:
            self.field = field

        if isinstance(other_field, FieldMixed) or isinstance(other_field, FieldsMixed):
            self.field.values = other_field[0].values
        else:
            self.field.values = other_field.values

        self.results.kinematics = self._extract(self.field)

        return self.field
