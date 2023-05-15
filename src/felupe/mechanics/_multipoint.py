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
from scipy.sparse import eye, lil_matrix

from ._helpers import Assemble, Results


class MultiPointConstraint:
    def __init__(
        self, field, points, centerpoint, skip=(False, False, False), multiplier=1e3
    ):
        "RBE2 Multi-point-constraint."
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.asarray(points)
        self.centerpoint = centerpoint
        self.mask = ~np.array(skip, dtype=bool)[: self.mesh.dim]
        self.axes = np.arange(self.mesh.dim)[self.mask]
        self.multiplier = multiplier

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def _vector(self, field=None, parallel=False):
        "Calculate vector of residuals with RBE2 contributions."

        if field is not None:
            self.field = field

        u = self.field.fields[0].values
        N = self.multiplier * (-u[self.points] + u[self.centerpoint])
        N[:, ~self.mask] = 0

        r = lil_matrix(u.shape)
        r[self.points] = -N
        r[self.centerpoint] = N.sum(axis=0)

        self.results.force = r.reshape(-1, 1).tocsr()
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate stiffness with RBE2 contributions."

        if field is not None:
            self.field = field

        indices = np.arange(self.mesh.ndof).reshape(self.mesh.points.shape)
        td = [indices[self.points.reshape(-1, 1), ax].ravel() for ax in self.axes]
        cd = [indices[self.centerpoint, ax].ravel() for ax in self.axes]

        L = lil_matrix((self.mesh.ndof, self.mesh.ndof))

        for t, c in zip(td, cd):
            L[t.reshape(-1, 1), t] = eye(len(t)) * self.multiplier
            L[t.reshape(-1, 1), c] = -self.multiplier
            L[c.reshape(-1, 1), t] = -self.multiplier
            L[c.reshape(-1, 1), c] = eye(len(c)) * self.multiplier * len(self.points)

        self.results.stiffness = L.tocsr()
        return self.results.stiffness


class MultiPointContact:
    def __init__(
        self, field, points, centerpoint, skip=(False, False, False), multiplier=1e6
    ):
        "RBE2 Multi-point-bolt-constraint."
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.asarray(points)
        self.centerpoint = centerpoint
        self.mask = ~np.array(skip, dtype=bool)[: self.mesh.dim]
        self.axes = np.arange(self.mesh.dim)[self.mask]
        self.multiplier = multiplier

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def _vector(self, field=None, parallel=False):
        "Calculate vector of residuals with RBE2 contributions."

        if field is not None:
            self.field = field

        u = self.field.fields[0].values

        Xc = self.mesh.points[self.centerpoint]
        Xt = self.mesh.points[self.points]

        xc = u[self.centerpoint] + Xc
        xt = u[self.points] + Xt

        mask = np.sign(-Xt + Xc) == np.sign(-xt + xc)
        mask[:, ~self.mask] = True
        n = -xt + xc
        n[mask] = 0

        r = lil_matrix(u.shape)
        r[self.points] = -self.multiplier * n
        r[self.centerpoint] = self.multiplier * n.sum(axis=0)

        self.results.force = r.reshape(-1, 1).tocsr()
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate stiffness with RBE2 contributions."

        if field is not None:
            self.field = field

        u = self.field.fields[0].values

        Xc = self.mesh.points[self.centerpoint]
        Xt = self.mesh.points[self.points]

        xc = u[self.centerpoint] + Xc
        xt = u[self.points] + Xt

        mask = np.sign(-Xt + Xc) != np.sign(-xt + xc)
        masks = [mask[:, ax] for ax in self.axes]

        indices = np.arange(self.mesh.ndof).reshape(self.mesh.points.shape)
        td = [indices[self.points.reshape(-1, 1), ax].ravel() for ax in self.axes]
        cd = [indices[self.centerpoint, ax].ravel() for ax in self.axes]

        L = lil_matrix((self.mesh.ndof, self.mesh.ndof))

        for t, c, m in zip(td, cd, masks):
            L[t[m].reshape(-1, 1), t[m]] = eye(len(t[m])) * self.multiplier
            L[t[m].reshape(-1, 1), c] = -self.multiplier
            L[c.reshape(-1, 1), t[m]] = -self.multiplier
            L[c.reshape(-1, 1), c] = eye(len(c)) * self.multiplier * len(self.points[m])

        self.results.stiffness = L.tocsr()
        return self.results.stiffness
