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


import inspect

import numpy as np

from ..assembly import IntegralForm
from ..math import dot, dya, identity, sqrt
from ._helpers import Assemble, Evaluate, Results
from ._solidbody import Solid


class Truss(Solid):
    def __init__(self, umat, field, area):
        self.field = field
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.evaluate = Evaluate(gradient=self._gradient, hessian=self._hessian)
        self.results = Results(stress=True, elasticity=True)

        self.umat = umat
        self.area = np.array(area)

        self.form_vector = IntegralForm(
            [None],
            v=self.field,
            dV=None,
            grad_v=[False],
        )
        self.form_matrix = IntegralForm(
            [None],
            v=self.field,
            dV=None,
            u=self.field,
            grad_v=[False],
            grad_u=[False],
        )

        cells = self.field.region.mesh.cells
        self.X = self.field.region.mesh.points[cells].T
        dX = self.X[:, 1] - self.X[:, 0]
        self.length_undeformed = sqrt(dot(dX, dX, mode=(1, 1)))

    def _kinematics(self, field):

        cells = self.field.region.mesh.cells
        u = field[0].values[cells].T
        x = self.X + u

        dx = x[:, 1] - x[:, 0]

        length_deformed = sqrt(dot(dx, dx, mode=(1, 1)))

        stretch = length_deformed / self.length_undeformed
        normal_deformed = dx / length_deformed

        return stretch, length_deformed, normal_deformed

    def _gradient(self, field=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.field = field
            self.results.kinematics = self._kinematics(field)

        if "out" in inspect.signature(self.umat.gradient).parameters:
            kwargs["out"] = self.results.gradient

        gradient = self.umat.gradient(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )
        self.results.gradient = self.results.stress = gradient[0]
        self.results._statevars = gradient[-1]

        return self.results.gradient

    def _hessian(self, field=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.field = field
            self.results.kinematics = self._kinematics(field)

        if "out" in inspect.signature(self.umat.hessian).parameters:
            kwargs["out"] = self.results.hessian

        hessian = self.umat.hessian(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )
        self.results.hessian = self.results.elasticity = hessian[0]

        return self.results.hessian

    def _vector(self, field=None, parallel=None, args=(), kwargs=None):

        self.results.stress = self._gradient(field, args=args, kwargs=kwargs)
        normal_deformed = self.results.kinematics[2]

        force = self.results.stress * self.area
        r = force * np.array([-normal_deformed, normal_deformed])  # (a, i, cell)

        self.results.force = self.form_vector.assemble(values=[r], parallel=parallel)

        return self.results.force

    def _matrix(self, field=None, parallel=None, args=(), kwargs=None):

        self.results.hessian = self._hessian(field, args=args, kwargs=kwargs)
        L = self.length_undeformed
        stretch, l, n = self.results.kinematics

        S = self.results.stress = self._gradient(field, args=args, kwargs=kwargs)
        dSdE = self.results.elasticity = self._hessian(field, args=args, kwargs=kwargs)

        m = dya(n, n, mode=1)
        eye = identity(dim=len(n), shape=(1,))
        K_EE = dSdE / L * self.area * m + S / l * self.area * (eye - m)

        K = np.stack([[K_EE, -K_EE], [-K_EE, K_EE]])  # (a, b, i, j, cell)

        self.results.stiffness = self.form_matrix.assemble(
            values=[K.transpose([0, 2, 1, 3, 4])], parallel=parallel
        )

        return self.results.stiffness
