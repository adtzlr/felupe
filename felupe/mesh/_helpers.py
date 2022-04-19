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

from functools import wraps

from ._mesh import Mesh


def mesh_or_data(meshfun):
    """If a ``Mesh`` is passed to a mesh function, extract ``points`` and
    ``cells`` arrays along with the ``cell_type`` and return a ``Mesh``
    as a result."""

    @wraps(meshfun)
    def mesh_or_points_cells_type(*args, **kwargs):

        # init mesh flag
        is_mesh = False

        # check if unnamed args are passed
        if len(args) > 0:

            # meshfun(Mesh)
            if isinstance(args[0], Mesh):

                # set mesh flag
                is_mesh = True

                # get points, cells and cell_type
                points = args[0].points
                cells = args[0].cells
                cell_type = args[0].cell_type

                # remove Mesh from args
                args = args[1:]

        if not is_mesh:

            # meshfun(points:ndarray, cells:ndarray, cell_type:str)
            if "points" in kwargs.keys():
                # get points, cells and cell_type from keyword arguments
                points = kwargs.pop("points")
                cells = kwargs.pop("cells")
                cell_type = kwargs.pop("cell_type")

            # meshfun(ndarray, cells:ndarray, cell_type:str)
            elif "points" not in kwargs.keys() and "cells" in kwargs.keys():
                # get points as first entry of args
                # get cells and cell_type from keyword arguments
                points = args[0]
                cells = kwargs.pop("cells")
                cell_type = kwargs.pop("cell_type")
                args = args[1:]

            # meshfun(ndarray, cells, cell_type:str)
            elif (
                "points" not in kwargs.keys()
                and "cells" not in kwargs.keys()
                and "cell_type" in kwargs.keys()
            ):
                # get points and cells as first and second entries of args
                # get cell_type from keyword arguments
                points = args[0]
                cells = args[1]
                cell_type = kwargs.pop("cell_type")
                args = args[2:]

            # meshfun(points, cells, cell_type)
            elif (
                "points" not in kwargs.keys()
                and "cells" not in kwargs.keys()
                and "cell_type" not in kwargs.keys()
            ):
                # get points, cells and cell_type from unnamed arguments
                points = args[0]
                cells = args[1]
                cell_type = args[2]
                args = args[3:]

        # call mesh manipulation function
        points, cells, cell_type = meshfun(points, cells, cell_type, *args, **kwargs)

        # return a Mesh if a Mesh was passed
        if is_mesh:
            return Mesh(points=points, cells=cells, cell_type=cell_type)

        else:
            # or (points, cells, cell_type) if arrays were given
            return points, cells, cell_type

    return mesh_or_points_cells_type
