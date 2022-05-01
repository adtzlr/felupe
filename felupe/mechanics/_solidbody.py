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


class SolidBodyPressure:
    def __init__(self, umat, field):

        self.umat = umat
        self.field = field

        self.dV = self.field.region.dV
        self.normals = self.field.region.normals

        self.kinematics = self.extract(self.field)
        self.force = None
        self.stiffness = None

        self.form = {
            Field: IntegralForm,
            FieldMixed: IntegralFormMixed,
            FieldsMixed: IntegralFormMixed,
            FieldAxisymmetric: IntegralFormAxisymmetric,
        }[type(self.field)]

        self.kwargs = {
            Field: dict(dV=self.dV, grad_v=True, grad_u=True),
            FieldMixed: dict(dV=self.dV),
            FieldsMixed: dict(dV=self.dV),
            FieldAxisymmetric: dict(dV=self.dV, grad_v=True, grad_u=True),
        }[type(self.field)]

        self.area_change = AreaChange()

    def extract(self, field):

        self.field = field
        self.kinematics = (self.field.extract(),)

        return self.kinematics

    def vector(self, field=None, pressure=1, parallel=False, jit=False, resize=None):

        if field is not None:
            self.field = field
            self.kinematics = self.extract(field)

        self.pressure = pressure

        fun = pressure * self.area_change.function(
            *self.kinematics,
            self.normals,
        )

        self.force = IntegralForm(
            fun=fun, v=self.field, dV=self.dV, grad_v=False
        ).assemble(parallel=parallel, jit=jit)

        if resize is not None:
            self.force.resize(*resize.shape)

        return self.force

    def matrix(self, field=None, pressure=1, parallel=False, jit=False, resize=None):

        if field is not None:
            self.field = field
            self.kinematics = self.extract(field)

        self.pressure = pressure

        fun = pressure * self.area_change.gradient(
            *self.kinematics,
            self.normals,
        )
        self.stiffness = IntegralForm(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.dV,
            grad_v=False,
            grad_u=True,
        ).assemble(parallel=parallel, jit=jit)

        if resize is not None:
            self.stiffness.resize(*resize.shape)

        return self.stiffness

    def update(self, other_field, field=None):

        if field is not None:
            self.field = field

        if isinstance(other_field, FieldMixed) or isinstance(other_field, FieldsMixed):
            self.field.values = other_field[0].values
        else:
            self.field.values = other_field.values

        self.kinematics = self.extract(self.field)

        return self.field


class SolidBody:
    def __init__(self, umat, field):

        self.umat = umat
        self.field = field

        if isinstance(field, FieldMixed):
            self.dV = self.field[0].region.dV
        else:
            self.dV = self.field.region.dV

        self.kinematics = self.extract(self.field)
        self.force = None
        self.stiffness = None
        self.stress = None
        self.elasticity = None

        self.form = {
            Field: IntegralForm,
            FieldMixed: IntegralFormMixed,
            FieldsMixed: IntegralFormMixed,
            FieldAxisymmetric: IntegralFormAxisymmetric,
        }[type(self.field)]

        self.kwargs = {
            Field: dict(dV=self.dV, grad_v=True, grad_u=True),
            FieldMixed: dict(dV=self.dV),
            FieldsMixed: dict(dV=self.dV),
            FieldAxisymmetric: dict(dV=self.dV, grad_v=True, grad_u=True),
        }[type(self.field)]

    def vector(self, field=None, parallel=False, jit=False, items=None):

        if field is not None:
            self.field = field

        self.stress = self.gradient(field)

        self.force = self.form(
            fun=self.stress[slice(items)],
            v=self.field,
            **self.kwargs,
        ).assemble(parallel=parallel, jit=jit)

        return self.force

    def matrix(self, field=None, parallel=False, jit=False, items=None):

        if field is not None:
            self.field = field

        self.elasticity = self.hessian(field)

        self.stiffness = self.form(
            fun=self.elasticity[slice(items)],
            v=self.field,
            u=self.field,
            **self.kwargs,
        ).assemble(parallel=parallel, jit=jit)

        return self.stiffness

    def extract(self, field):

        self.field = field

        self.kinematics = self.field.extract()
        if isinstance(self.field, Field):
            self.kinematics = (self.kinematics,)

        return self.kinematics

    def gradient(self, field=None, *args, **kwargs):

        if field is not None:
            self.field = field
            self.kinematics = self.extract(self.field)

        self.stress = self.umat.gradient(*self.kinematics, *args, **kwargs)

        return self.stress

    def hessian(self, field=None, *args, **kwargs):

        if field is not None:
            self.field = field
            self.kinematics = self.extract(self.field)

        self.elasticity = self.umat.hessian(*self.kinematics, *args, **kwargs)

        return self.elasticity
