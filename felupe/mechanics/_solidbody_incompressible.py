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
        self.J = np.ones(self.field.region.mesh.ncells)
        self.h = np.zeros(
            (
                self.field.region.mesh.cells.shape[1],  # points per cell
                self.field[0].dim,  # dimension
                self.field.region.mesh.ncells,  # cells
            )
        )
        self.dp = self.dJ = self.u = 0

        # volume of undeformed configuration
        self.V = self.field.region.dV.sum(0)
        self.v = self.V

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

        values = [
            form.integrate(parallel=parallel, jit=jit)[0]
            + self.h * (self.bulk * (self.v / self.V - 1) - self.p)
        ]

        self.results.force = form.assemble(values=values, parallel=parallel, jit=jit)

        return self.results.force

    def _matrix(
        self, field=None, parallel=False, jit=False, items=None, args=(), kwargs={}
    ):

        if field is not None:
            self.field = field

        self.results.elasticity = self._hessian(
            field, parallel=False, jit=False, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.elasticity,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        values = [
            form.integrate(parallel=parallel, jit=jit)[0]
            + self.bulk / self.V * dya(self.h, self.h)
        ]

        self.results.stiffness = form.assemble(
            values=values, parallel=parallel, jit=jit
        )

        return self.results.stiffness

    def _extract(self, field, parallel=False, jit=False):

        u = field[0].values
        self.du = (u - self.u)[self.field.region.mesh.cells].transpose([1, 2, 0])

        # change of state variables due to change of displacement field
        self.dJ = ddot(self.h, self.du, n=1) / self.V + (self.v / self.V - self.J)
        self.dp = self.bulk * (self.dJ + self.v / self.V - 1) - self.p

        self.field = field
        self.results.kinematics = self.field.extract()

        # update state variables
        self.p += self.dp
        self.J += self.dJ
        self.u = u

        dJdF = self._area_change.function(self.results.kinematics)
        self.h = self._form(fun=dJdF, v=self.field, dV=self.field.region.dV,).integrate(
            parallel=parallel, jit=jit
        )[0]

        # volume of deformed configuration
        self.v = (det(self.results.kinematics[0]) * self.field.region.dV).sum(0)

        return self.results.kinematics

    def _gradient(self, field=None, parallel=False, jit=False, args=(), kwargs={}):

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel, jit=jit)

        dJdF = self._area_change.function(self.results.kinematics)
        self.results.stress = [
            self.umat.gradient(self.results.kinematics, *args, **kwargs)[0]
            + self.p * dJdF[0]
        ]

        return self.results.stress

    def _hessian(self, field=None, parallel=False, jit=False, args=(), kwargs={}):

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel, jit=jit)

        d2JdF2 = self._area_change.gradient(self.results.kinematics)
        self.results.elasticity = [
            self.umat.hessian(self.results.kinematics, *args, **kwargs)[0]
            + self.p * d2JdF2[0]
        ]

        return self.results.elasticity

    def _kirchhoff_stress(self, field=None):

        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]

        return dot(P, transpose(F))

    def _cauchy_stress(self, field=None):

        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]
        J = det(F)

        return dot(P, transpose(F)) / J
