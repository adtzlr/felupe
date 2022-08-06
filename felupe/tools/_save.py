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

from ..math import dot, transpose, det, eigvalsh
from . import topoints


def save(
    region,
    fields,
    r=None,
    gradient=None,
    converged=True,
    filename="result.vtk",
    cell_data=None,
    point_data=None,
):

    u = fields.fields[0]
    mesh = region.mesh

    offsets = fields.offsets

    if point_data is None:
        point_data = {}

    point_data["Displacements"] = u.values

    if r is not None:
        reactionforces = np.split(r, offsets)[0]
        point_data["ReactionForce"] = reactionforces.reshape(*u.values.shape)

    if gradient is not None:
        # 1st Piola Kirchhoff stress
        F = fields.extract()[0]
        P = gradient[0]

        # cauchy stress at integration points
        s = dot(P, transpose(F)) / det(F)
        sp = np.sort(eigvalsh(s), axis=0)

        # shift stresses to points and average nodal values
        cauchy = topoints(s, region=region, sym=True)
        cauchyprinc = [topoints(sp_i, region=region, mode="scalar") for sp_i in sp]

        point_data["CauchyStress"] = cauchy

        point_data["MaxPrincipalCauchyStress"] = cauchyprinc[2]
        point_data["IntPrincipalCauchyStress"] = cauchyprinc[1]
        point_data["MinPrincipalCauchyStress"] = cauchyprinc[0]

        point_data["MaxPrincipalShearCauchyStress"] = cauchyprinc[2] - cauchyprinc[0]

    import meshio

    mesh = meshio.Mesh(
        points=mesh.points,
        cells=[
            (mesh.cell_type, mesh.cells),
        ],
        point_data=point_data,
        cell_data=cell_data,
    )

    mesh.write(filename)
