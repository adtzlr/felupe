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


class SolidBodyPressure:
    def __init__(self):
        pass


class Pressure:
    def __init__(self):
        pass

    def vector(self, field):
        pass

    def matrix(self, field):
        pass


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
