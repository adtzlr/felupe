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

from ._mesh import Mesh
from ._container import MeshContainer


def read(
    filename, file_format=None, cellblock=None, dim=None, merge=False, decimals=None
):
    "Read a mesh-file using meshio and create a felupe mesh-container."

    from meshio import read as meshio_read

    m = meshio_read(filename=filename, file_format=file_format)

    if dim is None:
        dim = m.points.shape[1]

    points = m.points[:, :dim]

    if cellblock is None:
        cellblock = slice(None)

    cells = m.cells[cellblock]

    if not isinstance(cells, list):
        cells = [cells]

    meshes = [Mesh(points, c.data, c.type) for c in cells]

    return MeshContainer(meshes, merge=merge, decimals=decimals)
