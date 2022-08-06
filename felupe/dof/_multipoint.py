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

import sparse


class MultiPointConstraint:
    def __init__(
        self, mesh, points, centerpoint, skip=(False, False, False), multiplier=1e3
    ):
        "RBE2 Multi-point-constraint."
        self.mesh = mesh
        self.points = points
        self.centerpoint = centerpoint
        self.mask = ~np.array(skip, dtype=bool)[: mesh.dim]
        self.axes = np.arange(mesh.dim)[self.mask]
        self.multiplier = multiplier

    def stiffness(self, field=None):
        "Calculate stiffness with RBE2 contributions."
        L = sparse.DOK(
            shape=(self.mesh.npoints, self.mesh.dim, self.mesh.npoints, self.mesh.dim)
        )
        c = self.centerpoint
        for t in self.points:
            for d in self.axes:
                L[t, d, t, d] = self.multiplier
                L[t, d, c, d] = -self.multiplier
                L[c, d, t, d] = -self.multiplier
                L[c, d, c, d] += self.multiplier
        return (
            sparse.COO(L)
            .reshape(
                (self.mesh.npoints * self.mesh.dim, self.mesh.npoints * self.mesh.dim)
            )
            .tocsr()
        )

    def residuals(self, field):
        "Calculate vector of residuals with RBE2 contributions."
        r = sparse.DOK(shape=(self.mesh.npoints, self.mesh.dim))
        c = self.centerpoint
        for t in self.points:
            for d in self.axes:
                N = self.multiplier * (
                    -field.fields[0].values[t, d] + field.fields[0].values[c, d]
                )
                r[t, d] = -N
                r[c, d] += N
        return sparse.COO(r).reshape((-1, 1)).tocsr()


class MultiPointContact:
    def __init__(
        self, mesh, points, centerpoint, skip=(False, False, False), multiplier=1e6
    ):
        "RBE2 Multi-point-bolt-constraint."
        self.mesh = mesh
        self.points = points
        self.centerpoint = centerpoint
        self.mask = ~np.array(skip, dtype=bool)[: mesh.dim]
        self.axes = np.arange(mesh.dim)[self.mask]
        self.multiplier = multiplier

    def stiffness(self, field):
        "Calculate stiffness with RBE2 contributions."
        L = sparse.DOK(
            shape=(self.mesh.npoints, self.mesh.dim, self.mesh.npoints, self.mesh.dim)
        )
        c = self.centerpoint
        for t in self.points:
            for d in self.axes:
                Xc = self.mesh.points[c, d]
                Xt = self.mesh.points[t, d]
                xc = field.fields[0].values[c, d] + Xc
                xt = field.fields[0].values[t, d] + Xt
                # n = 0
                if np.sign(-Xt + Xc) != np.sign(-xt + xc):
                    # n = -xt + xc
                    L[t, d, t, d] = self.multiplier
                    L[t, d, c, d] = -self.multiplier
                    L[c, d, t, d] = -self.multiplier
                    L[c, d, c, d] += self.multiplier
        return (
            sparse.COO(L)
            .reshape(
                (self.mesh.npoints * self.mesh.dim, self.mesh.npoints * self.mesh.dim)
            )
            .tocsr()
        )

    def residuals(self, field):
        "Calculate vector of residuals with RBE2 contributions."
        r = sparse.DOK(shape=(self.mesh.npoints, self.mesh.dim))
        c = self.centerpoint
        for t in self.points:
            for d in self.axes:
                Xc = self.mesh.points[c, d]
                Xt = self.mesh.points[t, d]
                xc = field.fields[0].values[c, d] + Xc
                xt = field.fields[0].values[t, d] + Xt
                if np.sign(-Xt + Xc) != np.sign(-xt + xc):
                    n = -xt + xc
                    r[t, d] = -self.multiplier * n
                    r[c, d] += self.multiplier * n
        return sparse.COO(r).reshape((-1, 1)).tocsr()
