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


class Boundary:
    "A Boundary as a collection of prescribed dofs (coordinate components of a field at points of a mesh)."

    def __init__(
        self,
        field,
        name="default",
        fx=lambda v: v == np.nan,
        fy=lambda v: v == np.nan,
        fz=lambda v: v == np.nan,
        value=0,
        skip=(False, False, False),
    ):

        mesh = field.region.mesh
        dof = field.indices.dof

        self.ndim = field.dim  # mesh.ndim
        self.name = name
        self.value = value
        self.skip = np.array(skip).astype(int)[: mesh.ndim]  # self.ndim
        self.fun = [fx, fy, fz][: mesh.ndim]

        # apply functions on the points per coordinate
        # fx(x), fy(y), fz(z) and create a mask for each coordinate
        mask = [f(x) for f, x in zip(self.fun, mesh.points.T)]

        # combine the masks with "logical_or" if ndim > 1
        if mesh.ndim == 1:
            mask = mask[0]

        elif mesh.ndim == 2:
            mask = np.logical_or(mask[0], mask[1])

        elif mesh.ndim == 3:  # and mesh.points.shape[1] == 3:
            tmp = np.logical_or(mask[0], mask[1])
            mask = np.logical_or(tmp, mask[2])

        # tile the mask
        self.mask = np.tile(mask.reshape(-1, 1), self.ndim)

        # check if some axes should be skipped
        if True not in skip:
            pass
        else:
            # exclude mask from axes which should be skipped
            self.mask[:, np.where(self.skip)[0]] = False

        self.dof = dof[self.mask]
        self.points = np.arange(mesh.npoints)[mask]