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

from ._container import MeshContainer
from ._mesh import Mesh


def read(
    filename, file_format=None, cellblock=None, dim=None, merge=False, decimals=None
):
    """Read a mesh from a file using :func:`meshio.read` and create a
    :class:`~felupe.MeshContainer`.

    Parameters
    ----------
    filename : str
        The filename of the mesh file.
    file_format : str or None, optional
        The file format of the mesh file (default is None).
    cellblock : list of int or None, optional
        Read only a subset of the cellblocks from the mesh file (default is None). If
        None, all cell blocks are added to the :class:`~felupe.MeshContainer`.
    dim : int or None, optional
        If provided, the dimension to trim the points array to (default is None). If
        None, the points array is unchanged.
    merge : bool, optional
        Flag to merge duplicate mesh points. This changes the cells arrays of the
        meshes. Default is False.
    decimals : float or None, optional
        Precision decimals for merging duplicated mesh points. Only relevant if
        merge=True. Default is None.

    Returns
    -------
    MeshContainer
        A mesh container created with :func:`meshio.read`.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Rectangle(n=3)
    >>> mesh.write(filename="mesh.xdmf")

    >>> container = fem.mesh.read("mesh.xdmf")
    >>> container
    <felupe mesh container object>
      Number of points: 9
      Number of cells:
        quad: 4

    >>> container.meshes[0]
    <felupe Mesh object>
      Number of points: 9
      Number of cells:
        quad: 4

    See Also
    --------
    meshio.read : Reads an unstructured mesh with added data.
    felupe.Mesh.write : Write the mesh to a file.
    """

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

    if len(cells) > 0:
        meshes = [Mesh(points, c.data, c.type) for c in cells]
    else:
        meshes = [Mesh(points, np.zeros((0, 0), dtype=int), None)]

    return MeshContainer(meshes, merge=merge, decimals=decimals)
