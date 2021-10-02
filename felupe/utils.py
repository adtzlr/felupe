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

from collections import namedtuple
from copy import deepcopy as copy

import numpy as np
from scipy.sparse import csr_matrix as sparsematrix
from scipy.interpolate import interp1d

import meshio

from .form import IntegralFormMixed
from .solve import partition, solve
from .math import identity, grad, dot, transpose, eigvals, det, interpolate, norms
from .field import Field

from .mesh import revolve
from .region import Region
from .quadrature import GaussLegendre
from .element import Quad, Hexahedron

from .dof import apply, partition as dofpartition, extend as dofextend


def topoints(values, region, sym=True, mode="tensor"):

    rows = region.mesh.cells.T.ravel()
    cols = np.zeros_like(rows)

    if mode == "tensor":
        ndim = values.shape[0]
        if sym:
            if ndim == 3:
                ij = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
            elif ndim == 2:
                ij = [(0, 0), (1, 1), (0, 1)]
        else:
            if ndim == 3:
                ij = [
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (1, 1),
                    (1, 2),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                ]
            elif ndim == 2:
                ij = [(0, 0), (0, 1), (1, 0), (1, 1)]

        out = Field(region, dim=len(ij)).values

        for a, (i, j) in enumerate(ij):
            out[:, a] = (
                sparsematrix(
                    (values.reshape(ndim, ndim, -1)[i, j], (rows, cols)),
                    shape=(region.mesh.npoints, 1),
                ).toarray()[:, 0]
                / region.mesh.cells_per_point
            )

    elif mode == "scalar":
        out = sparsematrix(
            (values.ravel(), (rows, cols)), shape=(region.mesh.npoints, 1)
        ).toarray()[:, 0]
        out = out / region.mesh.cells_per_point

    return out


def save(
    region,
    fields,
    r=None,
    K=None,
    F=None,
    f=None,
    A=None,
    unstack=None,
    converged=True,
    filename="result.vtk",
    cell_data=None,
    point_data=None,
):

    if unstack is not None:
        if "mixed" in str(type(fields)):
            fields = fields.fields
        u = fields[0]
    else:
        u = fields

    mesh = region.mesh

    if point_data is None:
        point_data = {}

    point_data["Displacements"] = u.values

    if r is not None:
        if unstack is not None:
            reactionforces = np.split(r, unstack)[0]
        else:
            reactionforces = r

        point_data["ReactionForce"] = reactionforces.reshape(*u.values.shape)

    if f is not None:
        # 1st Piola Kirchhoff stress
        if unstack is not None:
            P = f[0]
        else:
            P = f

        # cauchy stress at integration points
        s = dot(P, transpose(F)) / det(F)
        sp = np.sort(eigvals(s), axis=0)

        # shift stresses to points and average nodal values
        cauchy = topoints(s, region=region, sym=True)
        cauchyprinc = [topoints(sp_i, region=region, mode="scalar") for sp_i in sp]

        point_data["CauchyStress"] = cauchy

        point_data["MaxPrincipalCauchyStress"] = cauchyprinc[2]
        point_data["IntPrincipalCauchyStress"] = cauchyprinc[1]
        point_data["MinPrincipalCauchyStress"] = cauchyprinc[0]

        point_data["MaxPrincipalShearCauchyStress"] = cauchyprinc[2] - cauchyprinc[0]

    mesh = meshio.Mesh(
        points=mesh.points,
        cells=[(mesh.cell_type, mesh.cells),],  # [:, : mesh.edgepoints]},
        # Optionally provide extra data on points, cells, etc.
        point_data=point_data,
        cell_data=cell_data,
    )

    if filename is not None:
        mesh.write(filename)


def force(results, boundary):
    return np.array(
        [
            (((np.split(res.r, res.unstack)[0]).reshape(-1, 3))[boundary.points]).sum(0)
            for res in results
        ]
    )


def moment(results, boundary, point=np.zeros(3)):

    points = results[0].fields[0].region.mesh.points
    points = point.reshape(-1, 3)

    indices = np.array([(1, 2), (2, 0), (0, 1)])

    if points.shape[0] == 1:
        points = np.tile(points, (len(results), 1))

    moments = []

    for pt, res in zip(points, results):
        displacements = res.fields[0].values
        d = ((points + displacements) - pt)[boundary.points]

        force = (np.split(res.r, res.unstack)[0]).reshape(-1, 3)
        f = force[boundary.points]

        moments.append([(f[:, i] * d[:, i[::-1]]).sum() for i in indices])

    return np.array(moments)


def curve(x, y):
    kind = [None, "linear", "quadratic", "cubic"][min(len(y), 4) - 1]
    f = interp1d(x[: len(y)], y, kind=kind)
    xx = np.linspace(x[0], x[: len(y)][-1])
    return np.array([x[: len(y)], y]), np.array([xx, f(xx)])