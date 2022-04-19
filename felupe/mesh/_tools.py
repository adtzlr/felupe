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

from ._helpers import mesh_or_data
from ..math import rotation_matrix


@mesh_or_data
def expand(points, cells, cell_type, n=11, z=1):
    """Expand a 1d-Line to a 2d-Quad or a 2d-Quad to a 3d-Hexahedron Mesh.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    n : int, optional (default is 11)
        Number of n-point repetitions or (n-1)-cell repetitions
    z : int, optional (default is 1)
        Total expand dimension (edge length in expand direction is z / n)

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.
    """

    # ensure points, cells as ndarray
    points = np.array(points)
    cells = np.array(cells)

    # get dimension of points array
    # init zero vector of input dimension
    dim = points.shape[1]
    zeros = np.zeros(dim)

    # set new cell-type and the appropriate slice
    cell_type_new, sl = {
        "line": ("quad", slice(None, None, -1)),
        "quad": ("hexahedron", slice(None, None, None)),
    }[cell_type]

    # init new padded points array
    p = np.pad(points, ((0, 0), (0, 1)))

    # generate new points array for every thickness expansion ``h``
    points_new = np.vstack([p + np.array([*zeros, h]) for h in np.linspace(0, z, n)])

    # generate new cells array
    c = [cells + len(p) * a for a in np.arange(n)]
    cells_new = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    return points_new, cells_new, cell_type_new


@mesh_or_data
def rotate(points, cells, cell_type, angle_deg, axis, center=None):
    """Rotate a Mesh.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    angle_deg : int
        Rotation angle in degree.
    axis : int
        Rotation axis.
    center : list or ndarray or None, optional (default is None)
        Center point coordinates.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.
    """

    points = np.array(points)
    dim = points.shape[1]

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.array(center)
    center = center.reshape(1, -1)

    points_new = (
        rotation_matrix(angle_deg, dim, axis) @ (points - center).T
    ).T + center

    return points_new, cells, cell_type


@mesh_or_data
def revolve(points, cells, cell_type, n=11, phi=180, axis=0):
    """Revolve a 2d-Quad to a 3d-Hexahedron Mesh.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    n : int, optional (default is 11)
        Number of n-point revolutions (or (n-1) cell revolutions).
    phi : int, optional (default is 180)
        Revolution angle in degree.
    axis : int, optional (default is 0)
        Revolution axis.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.
    """

    points = np.array(points)
    cells = np.array(cells)

    dim = points.shape[1]

    # set new cell-type and the appropriate slice
    cell_type_new, sl = {
        # "line": ("quad", slice(None, None, -1)),
        "quad": ("hexahedron", slice(None, None, None)),
    }[cell_type]

    if abs(phi) > 360:
        raise ValueError("phi must be within |phi| <= 360 degree.")

    p = np.pad(points, ((0, 0), (0, 1)))
    R = rotation_matrix

    points_new = np.vstack(
        [(R(angle, dim + 1) @ p.T).T for angle in np.linspace(0, phi, n)]
    )

    c = [cells + len(p) * a for a in np.arange(n)]

    if phi == 360:
        c[-1] = c[0]
        points_new = points_new[: len(points_new) - len(points)]

    cells_new = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    return points_new, cells_new, cell_type_new


@mesh_or_data
def sweep(points, cells, cell_type, decimals=None):
    """Sweep duplicated points and update cells of a Mesh.

    **WARNING**: This function sorts points.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    decimals : int or None, optional (default is None)
        Number of decimals for point coordinate comparison.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.
    """

    if decimals is None:
        points_rounded = points
    else:
        points_rounded = np.round(points, decimals)

    points_new, index, inverse, counts = np.unique(
        points_rounded, True, True, True, axis=0
    )

    original = np.arange(len(points))

    mask = inverse != original
    find = original[mask]
    replace = inverse[mask]

    cells_new = cells.copy()

    for i, j in zip(find, replace):
        cells_new[cells == i] = j

    return points_new, cells_new, cell_type


@mesh_or_data
def mirror(
    points, cells, cell_type, normal=[1, 0, 0], centerpoint=[0, 0, 0], axis=None
):
    """Mirror points by plane normal and ensure positive cell volumes for
    `tria`, `tetra`, `quad` and `hexahedron` cell types.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    normal: list or ndarray, optional (default is [1, 0, 0])
        Mirror-plane normal vector.
    centerpoint: list or ndarray, optional (default is [0, 0, 0])
        Center-point coordinates on the mirror plane.
    axis: int or None, optional (default is None)
        Mirror axis.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    """

    points = np.array(points)
    cells = np.array(cells)

    dim = points.shape[1]

    # create normal vector
    if axis is not None:
        normal = np.zeros(dim)
        normal[axis] = 1
    else:
        normal = np.array(normal, dtype=float)[:dim]

        # ensure unit vector
        normal /= np.linalg.norm(normal)

    centerpoint = np.array(centerpoint, dtype=float)[:dim]

    points_new = points - np.einsum(
        "i, k, ...k -> ...i", 2 * normal, normal, (points - centerpoint)
    )

    faces_to_flip = {
        "line": ([0, 1],),
        "tria": ([0, 1, 2],),
        "tetra": ([0, 1, 2],),
        "quad": ([0, 1, 2, 3],),
        "hexahedron": ([0, 1, 2, 3], [4, 5, 6, 7]),
    }[cell_type]

    cells_new = cells.copy()

    for face in faces_to_flip:
        cells_new[:, face] = cells[:, face[::-1]]

    return points_new, cells_new, cell_type
