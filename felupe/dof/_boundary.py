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
    "A Boundary as a collection of prescribed degrees of freedom."

    def __init__(
        self,
        field,
        name="default",
        fx=lambda v: v == np.nan,
        fy=lambda v: v == np.nan,
        fz=lambda v: v == np.nan,
        value=0,
        skip=(False, False, False),
        mask=None,
        mode="or",
    ):
        """A Boundary as a collection of prescribed degrees of freedom
        (numbered coordinate components of a field at points of a mesh).


        Arguments
        ---------
        field : Field
            Field on wich the boundary is created.

        name : str, optional (default is "default")
            Name of the boundary.

        fx : function, optional (default is `lambda v: v == np.nan`)
            Mask-function for x-component of mesh-points which returns
            `True` at points on which the boundary will be applied.

        fy : function, optional (default is `lambda v: v == np.nan`)
            Mask-function for y-component of mesh-points which returns
            `True` at points on which the boundary will be applied.

        fz : function, optional (default is `lambda v: v == np.nan`)
            Mask-function for z-component of mesh-points which returns
            `True` at points on which the boundary will be applied.

        value : int, optional (default is 0)
            Value of the selected (prescribed) degrees of freedom.

        skip : tuple of bool, optional (default is `(False, False, False)`)
            A tuple to define which axes of the selected points should be
            skipped (i.e. not prescribed).

        mode : string, optional (default is `or`)
            A string which defines the logical combination of points per axis.


        Attributes
        ----------
        mask : array
            Mask-array which contains prescribed degrees of freedom.

        dof :array
            Array which contains prescribed degrees of freedom.

        points : array
            Array which contains the points on which one or more degrees of
            freedom are prescribed.

        """

        mesh = field.region.mesh
        dof = field.indices.dof

        self.field = field
        self.dim = field.dim  # mesh.dim
        self.name = name
        self.value = value
        self.skip = np.array(skip).astype(int)[: mesh.dim]  # self.dim

        self.mode = mode

        # check if callable
        _fx = fx if callable(fx) else lambda x: np.isclose(x, fx)
        _fy = fy if callable(fy) else lambda y: np.isclose(y, fy)
        _fz = fz if callable(fz) else lambda z: np.isclose(z, fz)

        self.fun = [_fx, _fy, _fz][: mesh.dim]

        if mask is None:

            # apply functions on the points per coordinate
            # fx(x), fy(y), fz(z) and create a mask for each coordinate
            mask = [f(x) for f, x in zip(self.fun, mesh.points.T)]

            # select the logical combination function "or" or "and"
            combine = {"or": np.logical_or, "and": np.logical_and}[self.mode]

            # combine the masks with "logical_or" if dim > 1
            if mesh.dim == 1:
                mask = mask[0]

            elif mesh.dim == 2:
                mask = combine(mask[0], mask[1])

            elif mesh.dim == 3:  # and mesh.points.shape[1] == 3:
                tmp = np.logical_or(mask[0], mask[1])
                mask = combine(tmp, mask[2])

        # tile the mask
        self.mask = np.tile(mask.reshape(-1, 1), self.dim)

        # check if some axes should be skipped
        if True not in skip:
            pass
        else:
            # exclude mask from axes which should be skipped
            self.mask[:, np.where(self.skip)[0]] = False

        self.dof = dof[self.mask]
        self.points = np.arange(mesh.npoints)[mask]

    def update(self, value):

        self.value = value
