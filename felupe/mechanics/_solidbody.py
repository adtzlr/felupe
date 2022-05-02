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
from ..math import inv, dot
from ._helpers import Assemble, Evaluate, Results


class SolidBody:
    "A SolidBody with methods for the assembly of sparse vectors/matrices."

    def __init__(self, umat, field):

        self.umat = umat
        self.field = field

        if isinstance(field, FieldMixed):
            self._dV = self.field[0].region.dV
        else:
            self._dV = self.field.region.dV

        self.results = Results(stress=True, elasticity=True)
        self.results.kinematics = self._extract(self.field)

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
        )

        self._area_change = AreaChange()

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

    def _vector(
        self, field=None, parallel=False, jit=False, items=None, args=(), kwargs={}
    ):

        if field is not None:
            self.field = field

        self.results.stress = self._gradient(field, args=args, kwargs=kwargs)

        self.results.force = self._form(
            fun=self.results.stress[slice(items)],
            v=self.field,
            **self._kwargs,
        ).assemble(parallel=parallel, jit=jit)

        return self.results.force

    def _matrix(
        self, field=None, parallel=False, jit=False, items=None, args=(), kwargs={}
    ):

        if field is not None:
            self.field = field

        self.results.elasticity = self._hessian(field, args=args, kwargs=kwargs)

        self.results.stiffness = self._form(
            fun=self.results.elasticity[slice(items)],
            v=self.field,
            u=self.field,
            **self._kwargs,
        ).assemble(parallel=parallel, jit=jit)

        return self.results.stiffness

    def _extract(self, field):

        self.field = field

        self.results.kinematics = self.field.extract()
        if isinstance(self.field, Field):
            self.results.kinematics = (self.results.kinematics,)

        return self.results.kinematics

    def _gradient(self, field=None, args=(), kwargs={}):

        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(self.field)

        self.results.stress = self.umat.gradient(
            *self.results.kinematics, *args, **kwargs
        )

        return self.results.stress

    def _hessian(self, field=None, args=(), kwargs={}):

        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(self.field)

        self.results.elasticity = self.umat.hessian(
            *self.results.kinematics, *args, **kwargs
        )

        return self.results.elasticity

    def _cauchy_stress(self, field=None):

        self._gradient(field)

        if len(self.results.kinematics) > 1:
            P = self.results.stress[0]
        else:
            P = self.results.stress

        JiFT = self._area_change.function(self.results.kinematics[0])

        return dot(P, inv(JiFT))
