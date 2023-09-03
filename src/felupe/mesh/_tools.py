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
from scipy.interpolate import griddata

from ..math import rotation_matrix, transpose
from ._helpers import mesh_or_data


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
    n : int, optional
        Number of n-point repetitions or (n-1)-cell repetitions,
        default is 11.
    z : float or ndarray, optional
        Total expand dimension as float (edge length in expand direction is z / n),
        default is 1. Optionally, if an array is passed these entries are
        taken as expansion and `n` is ignored.

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
    if np.isscalar(z):
        points_z = np.linspace(0, z, n)
    else:
        points_z = z
        n = len(z)

    points_new = np.vstack([p + np.array([*zeros, h]) for h in points_z])

    # generate new cells array
    c = [cells + len(p) * a for a in np.arange(n)]
    cells_new = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    return points_new, cells_new, cell_type_new


def fill_between(mesh, other_mesh, n=11):
    """Fill a 2d-Quad Mesh between two 1d-Line Meshes, embedded in 2d-space, or a
    3d-Hexahedron Mesh between two 2d-Quad Meshes, embedded in 3d-space, by expansion.
    Both meshes must have equal number of points and cells. The cells-array is taken
    from the first mesh.

    Parameters
    ----------
    mesh : felupe.Mesh
        The base line- or quad-mesh.
    other_mesh : felupe.Mesh
        The other line- or quad-mesh.
    n : int or ndarray
        Number of n-point repetitions or (n-1)-cell repetitions,
        (default is 11). If an array is given, then its values are used for the
        relative positions in a reference configuration (-1, 1) between the two meshes.

    Returns
    -------
    mesh : felupe.Mesh
        The expanded mesh.
    """

    if not hasattr(n, "__len__"):
        n = np.linspace(-1, 1, n)

    sections = []
    for bottom, top in zip(mesh.points, other_mesh.points):
        sections.append(griddata(points=[-1, 1], values=np.vstack([bottom, top]), xi=n))

    new_mesh = mesh.copy()
    new_mesh.points = new_mesh.points[:, : mesh.points.shape[1] - 1]

    new_mesh = new_mesh.expand(n=len(n))
    new_mesh.points[:] = transpose(sections).reshape(new_mesh.points.shape)

    return new_mesh


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
    center : list or ndarray or None, optional
        Center point coordinates (default is None).

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
    n : int, optional
        Number of n-point revolutions (or (n-1) cell revolutions),
        default is 11.
    phi : float or ndarray, optional
        Revolution angle in degree (default is 180).
    axis : int, optional
        Revolution axis (default is 0).

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
        "line": ("quad", slice(None, None, -1)),
        "quad": ("hexahedron", slice(None, None, None)),
    }[cell_type]

    if np.isscalar(phi):
        points_phi = np.linspace(0, phi, n)
    else:
        points_phi = phi
        n = len(points_phi)

    if abs(points_phi[-1]) > 360:
        raise ValueError("phi must be within |phi| <= 360 degree.")

    p = np.pad(points, ((0, 0), (0, 1)))
    R = rotation_matrix

    points_new = np.vstack(
        [(R(angle, dim + 1, axis=axis) @ p.T).T for angle in points_phi]
    )

    c = [cells + len(p) * a for a in np.arange(n)]

    if points_phi[-1] == 360:
        c[-1] = c[0]
        points_new = points_new[: len(points_new) - len(points)]

    cells_new = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    return points_new, cells_new, cell_type_new


@mesh_or_data
def sweep(points, cells, cell_type, decimals=None):
    """Merge duplicated points and update cells of a Mesh.

    **WARNING**: This function re-sorts points.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    decimals : int or None, optional
        Number of decimals for point coordinate comparison (default is None).

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
def translate(points, cells, cell_type, move, axis):
    """Translate (move) a Mesh.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    move : float
        Translation along given axis.
    axis : int
        Translation axis.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.
    """

    points_new = np.array(points)
    points_new[:, axis] += move

    return points_new, cells, cell_type


@mesh_or_data
def flip(points, cells, cell_type, mask=None):
    """Ensure positive cell volumes for `tria`, `tetra`, `quad` and
    `hexahedron` cell types.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    mask: list or ndarray, optional
        Boolean mask for selected cells to flip.

    Returns
    -------
    points : ndarray
        Point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    """

    if mask is None:
        mask = slice(None)
    else:
        mask = np.where(mask)[0].reshape(-1, 1)

    faces_to_flip = {
        "line": ([0, 1],),
        "triangle": ([0, 1, 2],),
        "tetra": ([0, 1, 2],),
        "quad": ([0, 1, 2, 3],),
        "hexahedron": ([0, 1, 2, 3], [4, 5, 6, 7]),
    }[cell_type]

    cells_new = cells.copy()

    for face in faces_to_flip:
        cells_new[:, face] = cells[:, face[::-1]]

    return points, cells_new, cell_type


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
    normal: list or ndarray, optional
        Mirror-plane normal vector (default is [1, 0, 0]).
    centerpoint: list or ndarray, optional
        Center-point coordinates on the mirror plane (default is [0, 0, 0]).
    axis: int or None, optional
        Mirror axis (default is None).

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

    return flip(points_new, cells, cell_type, mask=None)


def concatenate(meshes):
    "Join a sequence of meshes with identical cell types."

    Mesh = meshes[0].__mesh__

    points = np.vstack([mesh.points for mesh in meshes])
    offsets = np.cumsum(np.insert([mesh.npoints for mesh in meshes][:-1], 0, 0))
    cells = np.vstack(
        [int(offset) + mesh.cells for offset, mesh in zip(offsets, meshes)]
    )
    mesh = Mesh(points=points, cells=cells, cell_type=meshes[0].cell_type)

    return mesh


def stack(meshes):
    "Stack cell-blocks from meshes with identical points-array and cell-types."

    Mesh = meshes[0].__mesh__

    points = meshes[0].points
    cells = np.vstack([mesh.cells for mesh in meshes])
    mesh = Mesh(points=points, cells=cells, cell_type=meshes[0].cell_type)

    return mesh


@mesh_or_data
def triangulate(points, cells, cell_type, mode=3):
    """Triangulate a quad or a hex mesh.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    mode: int, optional
        Choose a mode how to convert hexahedrons to tets [1] (default is 3).

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    References
    ----------
    [1] Dompierre, J., Labbé, P., Vallet, M. G., & Camarero, R. (1999).
    How to Subdivide Pyramids, Prisms, and Hexahedra into Tetrahedra.
    IMR, 99, 195.
    """

    if cell_type == "quad":
        # triangles out of a quad
        i = [0, 3]
        j = [1, 1]
        k = [3, 2]

        cells_new = np.dstack(
            (
                cells[:, i],
                cells[:, j],
                cells[:, k],
            )
        )

        cell_type_new = "triangle"

    elif cell_type == "hexahedron":
        # tets out of a hex
        # mode ... no. of diagional through hex-point 6.
        if mode == 0:
            i = [0, 0, 0, 0, 2]
            j = [1, 2, 2, 5, 7]
            k = [2, 7, 3, 7, 5]
            m = [5, 5, 7, 4, 6]

        elif mode == 3:
            i = [0, 0, 0, 0, 1, 1]
            j = [2, 3, 7, 5, 5, 6]
            k = [3, 7, 4, 6, 6, 2]
            m = [6, 6, 6, 4, 0, 0]

        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

        cells_new = np.dstack(
            (
                cells[:, i],
                cells[:, j],
                cells[:, k],
                cells[:, m],
            )
        )

        cell_type_new = "tetra"

    cells_new = cells_new.reshape(-1, cells_new.shape[-1])

    return points, cells_new, cell_type_new


@mesh_or_data
def runouts(
    points,
    cells,
    cell_type,
    values=[0.1, 0.1],
    centerpoint=[0, 0, 0],
    axis=0,
    exponent=5,
    mask=slice(None),
    normalize=False,
):
    """Add simple rubber-runouts for realistic rubber-metal structures.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    values : list or ndarray, optional
        Relative amount of runouts (per coordinate) perpendicular to the axis
        (default is 10% per coordinate, i.e. [0.1, 0.1]).
    centerpoint : list or ndarray, optional
        Center-point coordinates (default is [0, 0, 0]).
    axis : int or None, optional
        Axis (default is 0).
    exponent : int, optional
        Positive exponent to control the shape of the runout. The higher
        the exponent, the steeper the transition (default is 5).
    mask : list or None, optional
        List of points to be considered (default is None).
    normalize : bool, optional
        Normalize the runouts to create indents, i.e. maintain the original shape at the
        ends (default is False).

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.
    """

    dim = points.shape[1]
    runout_along = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    centerpoint = np.array(centerpoint, dtype=float)[:dim]
    values = np.array(values, dtype=float)[:dim]

    points_new = points - centerpoint
    top = points[:, axis].max()
    bottom = points[:, axis].min()

    # check symmetry
    if top == centerpoint[axis] or bottom == centerpoint[axis]:
        half_height = top - bottom
    else:
        half_height = (top - bottom) / 2

    for i, coord in enumerate(runout_along[axis][: dim - 1]):
        factor = (abs(points_new[mask, axis]) / half_height) ** exponent
        scale = 1 + factor * values[i]

        if normalize:
            scale /= np.max(np.abs(scale))

        points_new[mask, coord] *= scale

    return points_new + centerpoint, cells, cell_type
