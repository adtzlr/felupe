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
from ..constitution import AreaChange
from ..field import FieldContainer
from ..math import det, dot, transpose
from ._helpers import Assemble, Evaluate, Results, StateNearlyIncompressibleX
from ._solidbody import Solid


class SolidBodyNearlyIncompressibleX(Solid):
    def __init__(self, umat, field, bulk, state=None, statevars=None, **kwargs):
        self.umat = umat
        self.field = field
        self.bulk = bulk
        self._area_change = AreaChange()
        self.results = Results(stress=True, elasticity=True)

        if statevars is not None:
            self.results.statevars = statevars
        else:
            statevars_shape = (0,)
            if hasattr(umat, "x"):
                statevars_shape = umat.x[-1].shape
            self.results.statevars = np.zeros(
                (
                    *statevars_shape,
                    field.region.quadrature.npoints,
                    field.region.mesh.ncells,
                )
            )

        self.results.state = state
        if self.results.state is None:
            self.results.state = StateNearlyIncompressibleX(field, **kwargs)

        self.results.kinematics = self._extract(self.field)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
            kirchhoff_stress=self._kirchhoff_stress,
        )

    def _vector(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        self.results.stress = self._gradient(
            field, parallel=parallel, args=args, kwargs=kwargs
        )
        form = IntegralForm(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.integrate_shape_function_gradient()
        inv_V = self.results.state.inv_V
        constraint = self.results.state.constraint(bulk=self.bulk)

        values = form.integrate(parallel=parallel)
        np.add(
            values[0],
            np.einsum("aib...,bc...,cj...->ai...", h, inv_V, constraint),
            out=values[0],
        )

        self.results.force = form.assemble(values=values)

        return self.results.force

    def _matrix(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        self.results.elasticity = self._hessian(
            field, parallel=parallel, args=args, kwargs=kwargs
        )

        form = IntegralForm(
            fun=self.results.elasticity,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.integrate_shape_function_gradient()
        inv_V = self.results.state.inv_V

        values = form.integrate(parallel=parallel, out=self.results.stiffness_values)
        np.add(
            values[0],
            np.einsum("aic...,bjd...,cd...->aibj...", h, h, inv_V, optimize=True)
            * self.bulk,
            out=values[0],
        )

        self.results.stiffness = form.assemble(values=values)

        return self.results.stiffness

    def _extract(self, field, parallel=False):
        mesh = self.field.region.mesh

        u = field[0].values
        u0 = self.results.state.u

        mesh_dual = self.results.state.volume_ratio.region.mesh
        p = self.results.state.pressure.values[mesh_dual.cells].transpose([1, 2, 0])
        J = self.results.state.volume_ratio.values[mesh_dual.cells].transpose([1, 2, 0])

        h = self.results.state.integrate_shape_function_gradient()
        inv_V = self.results.state.inv_V

        # change of internal field values due to change of displacement field
        du = (u - u0)[mesh.cells].transpose([1, 2, 0])
        dJ = np.einsum(
            "ai...,aib...,bc...,j->cj...", du, h, inv_V, np.ones(1), optimize=True
        )
        dJ = np.add(dJ, dot(inv_V, self.results.state.fp()), out=dJ)
        dp = self.bulk * (dJ + J - 1) - p

        self.field = field
        self.results.kinematics = self.results.state.F = self.field.extract(
            out=self.results.kinematics
        )

        form = IntegralForm(
            fun=[np.ones(1)],
            v=FieldContainer([self.results.state.pressure]),
            dV=self.field.region.dV,
            grad_v=[False],
        )
        assemble = lambda values: form.assemble([np.asarray(values)]).toarray()

        # update state variables
        self.results.state.u = u
        self.results.state.pressure.values[:] += assemble(dp)
        self.results.state.volume_ratio.values[:] += assemble(dJ)

        return self.results.kinematics

    def _gradient(self, field=None, parallel=False, args=(), kwargs={}):
        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

        F = self.results.kinematics[0]
        p = self.results.state.pressure.interpolate()
        statevars = self.results.statevars

        gradient = self.umat.gradient([F, statevars], *args, **kwargs)

        dJdF = self._area_change.function
        self.results.stress = [gradient[0] + p * dJdF([F])[0]]
        self.results._statevars = gradient[-1]

        return self.results.stress

    def _hessian(self, field=None, parallel=False, args=(), kwargs={}):
        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

        F = self.results.kinematics[0]
        p = self.results.state.pressure.interpolate()
        statevars = self.results.statevars

        d2JdF2 = self._area_change.gradient
        self.results.elasticity = [
            self.umat.hessian([F, statevars], *args, **kwargs)[0] + p * d2JdF2([F])[0]
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
