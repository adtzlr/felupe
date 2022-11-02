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

import numpy as np

from .._field import Field
from .._assembly import IntegralFormMixed
from ..constitution import AreaChange
from ..math import dot, transpose, det, dya, ddot, identity
from ._helpers import Assemble, Evaluate, Results


class SolidBodyNearlyIncompressible:
    """A (nearly) incompressible SolidBody with methods for the assembly of
    sparse vectors/matrices."""

    def __init__(self, umat, field, bulk):

        self.umat = umat
        self.field = field
        self.bulk = bulk

        self._area_change = AreaChange()
        self._form = IntegralFormMixed

        # internal state variables
        self.p = np.zeros(self.field.region.mesh.ncells)

        # volume of undeformed configuration
        self.V = self.field.region.dV.sum(0)

        self.results = Results(stress=True, elasticity=True)
        self.results.kinematics = self._extract(self.field)

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
            kirchhoff_stress=self._kirchhoff_stress,
        )

    def _vector(
        self, field=None, parallel=False, jit=False, items=None, args=(), kwargs={}
    ):

        if field is not None:
            self.field = field

        self.results.stress = self._gradient(
            field, parallel=False, jit=False, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        )

        dUdJ = self.bulk * (self.v / self.V - 1)
        r = [form.integrate(parallel=parallel, jit=jit)[0] + dUdJ * self.H]

        self.results.force = form.assemble(values=r, parallel=parallel, jit=jit)

        return self.results.force

    def _matrix(
        self, field=None, parallel=False, jit=False, items=None, args=(), kwargs={}
    ):

        if field is not None:
            self.field = field

        self.results.elasticity = self._hessian(
            field, parallel=False, jit=False, args=args, kwargs=kwargs
        )

        p = self.p - self.dp

        fun = [
            self.results.elasticity[0]
            + p * self._area_change.gradient(self.results.kinematics)[0]
        ]

        form = self._form(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        K = [
            form.integrate(parallel=parallel, jit=jit)[0]
            + self.bulk / self.V * dya(self.H, self.H)
        ]

        self.results.stiffness = form.assemble(values=K, parallel=parallel, jit=jit)

        return self.results.stiffness

    def _extract(self, field, parallel=False, jit=False):

        # change of displacement field
        self.du = (field[0].values - self.field[0].values)[
            field.region.mesh.cells
        ].transpose([1, 2, 0])

        self.field = field
        self.results.kinematics = self.field.extract()

        self.H = self._form(
            fun=self._area_change.function(self.results.kinematics),
            v=self.field,
            dV=self.field.region.dV,
        ).integrate(parallel=parallel, jit=jit)[0]

        # volume of deformed configuration
        mesh = self.field.region.mesh.copy()
        mesh.points += self.field[0].values
        self.v = type(self.field.region)(mesh).dV.sum(0)

        # change of state variables due to change of displacement field
        dUdJ = self.bulk * (self.v / self.V - 1)
        self.dp = -self.p + dUdJ + self.bulk / self.V * ddot(self.H, self.du, n=1)

        # update state variables
        self.p += self.dp

        return self.results.kinematics

    def _gradient(self, field=None, parallel=False, jit=False, args=(), kwargs={}):

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel, jit=jit)

        self.results.stress = self.umat.gradient(
            self.results.kinematics, *args, **kwargs
        )

        return self.results.stress

    def _hessian(self, field=None, parallel=False, jit=False, args=(), kwargs={}):

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel, jit=jit)

        self.results.elasticity = self.umat.hessian(
            self.results.kinematics, *args, **kwargs
        )

        return self.results.elasticity

    def _kirchhoff_stress(self, field=None):

        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]
        J = det(F)

        return dot(P, transpose(F)) + identity(dim=3) * self.p * J

    def _cauchy_stress(self, field=None):

        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]
        J = det(F)

        return dot(P, transpose(F)) / J + identity(dim=3) * self.p
