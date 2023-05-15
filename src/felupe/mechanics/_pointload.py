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
from scipy.sparse import csr_matrix

from ._helpers import Assemble, Results


class PointLoad:
    "A point load with methods for the assembly of sparse vectors/matrices."

    def __init__(self, field, points, values=None, apply_on=0, axisymmetric=False):
        self.field = field
        self.points = points

        if values is None:
            self.values = 0
        else:
            self.values = values

        self.apply_on = apply_on
        self.axisymmetric = axisymmetric

        self.results = Results()
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def update(self, values):
        self.__init__(self.field, self.points, values, self.apply_on, self.axisymmetric)

    def _vector(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        force = [np.zeros_like(f.values) for f in self.field.fields]
        force[self.apply_on][self.points] += self.values

        if self.axisymmetric:
            mesh_points = self.field[0].region.mesh.points
            radius = mesh_points[self.points, 1].reshape(-1, 1)
            force[self.apply_on][self.points] *= 2 * np.pi * radius

        self.results.force = csr_matrix(
            np.concatenate([f.ravel() for f in force]).reshape(-1, 1)
        )

        return -self.results.force

    def _matrix(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        n = np.sum(self.field.fieldsizes)
        self.results.stiffness = csr_matrix(([0], ([0], [0])), shape=(n, n))

        return self.results.stiffness
