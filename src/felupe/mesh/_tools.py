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
def expand(points, cells, cell_type, n=11, z=1, axis=-1, expand_dim=True):
    """Expand a 0d-Point to a 1d-Line, a 1d-Line to a 2d-Quad or a 2d-Quad to a
    3d-Hexahedron Mesh.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    n : int, optional
        Number of n-point repetitions or (n-1)-cell repetitions, default is 11. Must be
        greater or equal 0.
    z : float or ndarray, optional
        Total expansion as float (edge length in expand direction is
        ``z / (n - 1)``), default is 1. Optionally, if an array is passed these entries
        are taken as expansion and ``n`` is ignored.
    axis : int, optional
        Axis of expansion (default is -1).
    expand_dim : bool, optional
        Expand the dimension of the point coordinates (default is True).

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    Expand a rectangle to a cube.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> rect = fem.Rectangle(n=4)
       >>> cube = fem.mesh.expand(rect, n=7, z=2)
       >>>
       >>> cube.plot().show()

    >>> cube
    <felupe Mesh object>
      Number of points: 112
      Number of cells:
        hexahedron: 54

    See Also
    --------
    felupe.Mesh.expand : Expand a 0d-Point to a 1d-Line, a 1d-Line to a 2d-Quad or a
        2d-Quad to a 3d-Hexahedron Mesh.
    """

    thickness = z

    # ensure points, cells as ndarray
    points = np.array(points)
    cells = np.array(cells)

    # get dimension of points array
    dim = points.shape[1]

    # init new padded points array
    dim_new = dim
    if expand_dim:
        dim_new += 1

    points_new = np.pad(points, ((0, 0), (0, dim_new - dim)))[np.newaxis, ...]
    cells_new = cells
    cell_type_new = cell_type

    # set new cell-type and the appropriate slice
    if n > 1:
        cell_type_new, sl = {
            "vertex": ("line", slice(None, None, None)),
            "line": ("quad", slice(None, None, -1)),
            "quad": ("hexahedron", slice(None, None, None)),
            "quad9": ("hexahedron27", slice(None, None, None)),
        }[cell_type]

        if cell_type_new == "hexahedron27":
            n = 2 * n - 1

        if np.isscalar(thickness):
            points_thickness = np.linspace(0, thickness, n)
        else:

            points_thickness = np.array(thickness)
            if cell_type_new == "hexahedron27":

                points_thickness = np.vstack(
                    [
                        points_thickness,
                        points_thickness + np.diff(points_thickness, append=np.nan) / 2,
                    ]
                ).T.ravel()[:-1]

        n = len(points_thickness)

        # init zero vector of input dimension
        layers = np.zeros((n, dim_new))
        layers[:, axis] = points_thickness

        points_new = points_new + layers[:, np.newaxis, ...]

        # generate new cells array
        cells_new = (
            cells[np.newaxis, ...]
            + len(points) * np.arange(n)[..., np.newaxis, np.newaxis]
        )

        if cell_type_new == "hexahedron27":
            verts = [0, 1, 2, 3, 18, 19, 20, 21]
            edges = [4, 5, 6, 7, 22, 23, 24, 25, 9, 10, 11, 12]
            faces = [16, 14, 13, 15, 8, 26]
            volume = [17]

            start = cells_new[:-2:2]
            middle = cells_new[1:-1:2]
            end = cells_new[2::2]

            cells_new = np.concatenate([start, middle, end], axis=-1)[
                ..., [*verts, *edges, *faces, *volume]
            ]

        else:
            start = cells_new[:-1]
            end = cells_new[1:, ..., sl]

            cells_new = np.concatenate([start, end], axis=-1)

        # expand vertex point to line in first direction
        if cell_type_new == "line":
            points_new = points_new[..., 1:]

    return (
        points_new.reshape(-1, points_new.shape[-1]),
        cells_new.reshape(-1, cells_new.shape[-1]),
        cell_type_new,
    )


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

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> inner = fem.mesh.revolve(fem.Point(1)).expand(z=0.4).translate(0.2, axis=2)
       >>> outer = fem.mesh.revolve(fem.Point(2), phi=160).rotate(
       ...     axis=2, angle_deg=20
       ... ).expand(z=1.2)
       >>> mesh = fem.mesh.fill_between(inner, outer, n=6)
       >>>
       >>> mesh.plot().show()

    See Also
    --------
    felupe.Mesh.fill_between : Fill a 2d-Quad Mesh between two 1d-Line Meshes, embedded
        in 2d-space, or a 3d-Hexahedron Mesh between two 2d-Quad Meshes, embedded in
        3d-space, by expansion.

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


def _rotation_minimizing_frames(points, tangents):
    """Return rotation-minimizing frames along a discrete path by the double-reflection
    method. The columns of each frame are the tangent and two normal vectors."""

    # an arbitrary normal vector at the start of the path
    normal = np.cross(tangents[0], np.eye(3)[np.argmin(np.abs(tangents[0]))])
    normal /= np.linalg.norm(normal)

    frames = np.zeros((len(points), 3, 3))
    frames[0] = np.column_stack([tangents[0], normal, np.cross(tangents[0], normal)])

    for i in range(len(points) - 1):
        # reflection at the bisecting plane of the two points
        v1 = points[i + 1] - points[i]
        c1 = v1 @ v1
        normal_left = normal - 2 / c1 * (v1 @ normal) * v1
        tangent_left = tangents[i] - 2 / c1 * (v1 @ tangents[i]) * v1

        # second reflection to align the reflected tangent with the next tangent
        v2 = tangents[i + 1] - tangent_left
        c2 = v2 @ v2
        normal = normal_left
        if c2 > np.finfo(float).eps:
            normal = normal_left - 2 / c2 * (v2 @ normal_left) * v2

        frames[i + 1] = np.column_stack(
            [tangents[i + 1], normal, np.cross(tangents[i + 1], normal)]
        )

    return frames


def extrude(mesh, path, other_mesh=None):
    r"""Extrude a 1d-Line to a 2d-Quad or a 2d-Quad to a 3d-Hexahedron Mesh along a
    given path. Optionally, the section is interpolated between the mesh and another
    mesh.

    Parameters
    ----------
    mesh : felupe.Mesh
        A line- or quad-mesh (the section) in its position at the start of the path.
    path : felupe.Mesh or ndarray
        A line-mesh or an array of point coordinates, which defines the path. The
        points of the path define the positions of the extruded sections. The cells of
        a line-mesh must be ordered consecutively along the path.
    other_mesh : felupe.Mesh or None, optional
        Another line- or quad-mesh with equal cells and number of points, also located
        at the start of the path (default is None). If given, the section is linearly
        interpolated between the mesh (at the start) and the other mesh (at the end of
        the path) w.r.t. the normalized arc length of the path.

    Returns
    -------
    felupe.Mesh
        The extruded mesh.

    Notes
    -----
    The section is transported along the path by rotation-minimizing frames [1]_,
    i.e. the section is not twisted around the path. The section is not required to be
    perpendicular to the path. Instead, the (initial) orientation of the section w.r.t.
    the tangent of the path is preserved. The tangents are evaluated as the normalized
    mean of the two adjacent segment directions at each point of the path. For each
    point :math:`\boldsymbol{p}_i` of the path, the points of the section are given by
    Eq. :eq:`extrude`

    ..  math::
        :label: extrude

        \boldsymbol{x}_i = \boldsymbol{p}_i + \boldsymbol{Q}_i \left(
            \boldsymbol{X} - \boldsymbol{p}_0
        \right)

    where :math:`\boldsymbol{Q}_i` is the rotation matrix from the frame at the start
    to the frame at the :math:`i`-th point of the path.

    ..  note::
        The mesh must be located at the start of the path. The path must not contain
        duplicate consecutive points. Duplicate points of closed paths are not merged.

    If the mesh is a line-mesh and both the mesh and the path are embedded in 2d-space,
    the extruded quad-mesh is also embedded in 2d-space. Otherwise, the extruded mesh
    is embedded in 3d-space. The cells of the extruded mesh are flipped, if required,
    to ensure positive cell volumes (for meshes embedded in 2d-space with quads and for
    hexahedrons).

    Examples
    --------
    Extrude a circular section along a helix. The section is not perpendicular to the
    path, it is tilted by 20 degrees.

    ..  pyvista-plot::
        :force_static:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> phi = np.linspace(0, 4 * np.pi, 97)
        >>> path = fem.mesh.Line(n=97)
        >>> path.update(points=np.vstack([np.cos(phi), np.sin(phi), phi / 8]).T)
        >>>
        >>> section = (
        ...     fem.Circle(radius=0.2, n=4)
        ...     .expand(n=1)
        ...     .rotate(90, axis=0)
        ...     .rotate(20, axis=2)
        ...     .translate(1, axis=0)
        ... )
        >>> mesh = fem.mesh.extrude(section, path)
        >>>
        >>> mesh.plot().show()

    Optionally, the section may be interpolated between two sections with equal cells.
    Both sections are located at the start of the path.

    ..  pyvista-plot::
        :force_static:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> phi = np.linspace(0, np.pi / 2, 21)
        >>> path = fem.mesh.Line(n=21)
        >>> path.update(points=2 * np.vstack([np.cos(phi), np.sin(phi)]).T)
        >>>
        >>> section = fem.mesh.Line(a=1.8, b=2.2, n=5)
        >>> section.update(points=np.pad(section.points, ((0, 0), (0, 1))))
        >>> other_section = fem.mesh.Line(a=1.5, b=2.5, n=5)
        >>> other_section.update(points=np.pad(other_section.points, ((0, 0), (0, 1))))
        >>>
        >>> mesh = fem.mesh.extrude(section, path, other_section)
        >>>
        >>> mesh.plot().show()

    References
    ----------
    .. [1] W. Wang, B. Jüttler, D. Zheng and Y. Liu, "Computation of rotation minimizing
       frames", ACM Transactions on Graphics, vol. 27, no. 1, pp. 1–18, 2008,
       doi: `10.1145/1330511.1330513 <https://doi.org/10.1145/1330511.1330513>`_.

    See Also
    --------
    felupe.Mesh.extrude : Extrude a 1d-Line to a 2d-Quad or a 2d-Quad to a
        3d-Hexahedron Mesh along a given path.
    felupe.mesh.expand : Expand a 0d-Point to a 1d-Line, a 1d-Line to a 2d-Quad or a
        2d-Quad to a 3d-Hexahedron Mesh.
    felupe.mesh.fill_between : Fill a 2d-Quad Mesh between two 1d-Line Meshes, embedded
        in 2d-space, or a 3d-Hexahedron Mesh between two 2d-Quad Meshes, embedded in
        3d-space, by expansion.
    felupe.mesh.interpolate_line : Return an interpolated line mesh from an existing
        line mesh with provided interpolation points on a given axis.

    """

    if mesh.cell_type not in ["line", "quad"]:
        raise TypeError("The cell type of the mesh must be either line or quad.")

    if other_mesh is not None and (
        other_mesh.points.shape != mesh.points.shape
        or not np.array_equal(other_mesh.cells, mesh.cells)
    ):
        raise ValueError("Both meshes must have equal points-shape and cells.")

    # get the ordered points of the path
    if hasattr(path, "cells"):
        if path.cell_type != "line":
            raise TypeError("The cell type of the path must be line.")

        cells = path.cells
        if not np.all(cells[1:, 0] == cells[:-1, 1]):
            raise ValueError("The cells of the path must be ordered consecutively.")

        path_points = path.points[np.append(cells[:, 0], cells[-1, 1])]

    else:
        path_points = np.asarray(path, dtype=float)

    if path_points.ndim != 2 or len(path_points) < 2:
        raise ValueError("The path must contain at least two points.")

    # dimension of the extruded mesh
    dim = 3
    if mesh.cell_type == "line" and mesh.dim == 2 and path_points.shape[1] == 2:
        dim = 2

    def pad(points):
        return np.pad(points, ((0, 0), (0, 3 - points.shape[1])))

    path_points = pad(path_points)
    section = pad(mesh.points)
    other_section = section
    if other_mesh is not None:
        other_section = pad(other_mesh.points)

    # segment lengths and tangents
    segments = np.diff(path_points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)

    if np.any(np.isclose(lengths, 0)):
        raise ValueError("The path must not contain duplicate consecutive points.")

    directions = segments / lengths.reshape(-1, 1)
    tangents = np.zeros_like(path_points)
    tangents[:-1] += directions
    tangents[1:] += directions
    norm = np.linalg.norm(tangents, axis=1)

    if np.any(np.isclose(norm, 0)):
        raise ValueError("The path must not reverse its direction.")

    tangents /= norm.reshape(-1, 1)

    # rotation matrices from the frame at the start to the frames along the path
    frames = _rotation_minimizing_frames(path_points, tangents)
    rotations = np.einsum("nij,kj->nik", frames, frames[0])

    # interpolated sections, relative to the start of the path
    progress = np.insert(np.cumsum(lengths), 0, 0) / np.sum(lengths)
    progress = progress.reshape(-1, 1, 1)
    sections = (1 - progress) * section + progress * other_section - path_points[0]

    points = path_points.reshape(-1, 1, 3) + np.einsum(
        "nij,nkj->nki", rotations, sections
    )

    # take the cells from the expanded mesh
    new_mesh = mesh.copy().expand(n=len(path_points))
    new_mesh = new_mesh.copy(points=points.reshape(-1, 3)[:, :dim])

    # ensure positive cell volumes
    if new_mesh.cell_type == "hexahedron" or dim == 2:
        x = new_mesh.points[new_mesh.cells[: mesh.ncells]]

        if new_mesh.cell_type == "quad":
            dxdr = x[:, [1, 2]].mean(axis=1) - x[:, [0, 3]].mean(axis=1)
            dxds = x[:, [2, 3]].mean(axis=1) - x[:, [0, 1]].mean(axis=1)
            volumes = dxdr[:, 0] * dxds[:, 1] - dxdr[:, 1] * dxds[:, 0]

        else:
            dxdr = x[:, [1, 2, 5, 6]].mean(axis=1) - x[:, [0, 3, 4, 7]].mean(axis=1)
            dxds = x[:, [2, 3, 6, 7]].mean(axis=1) - x[:, [0, 1, 4, 5]].mean(axis=1)
            dxdt = x[:, [4, 5, 6, 7]].mean(axis=1) - x[:, [0, 1, 2, 3]].mean(axis=1)
            volumes = np.einsum("ci,ci->c", dxdr, np.cross(dxds, dxdt))

        if not (np.all(volumes > 0) or np.all(volumes < 0)):
            raise ValueError(
                "The path must not be tangent to the section at the start of the path."
            )

        if np.all(volumes < 0):
            new_mesh = flip(new_mesh)

    return new_mesh


@mesh_or_data
def rotate(points, cells, cell_type, angle_deg, axis, center=None, mask=None):
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
    mask : ndarray or None, optional
        A boolean mask to select points which are rotated (default is None).

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    Rotate a rectangle in the xy-plane by 35 degree.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> rect = fem.Rectangle(b=(3, 1), n=(10, 4))
       >>> mesh = fem.mesh.rotate(rect, angle_deg=35, axis=2, center=[1.5, 0.5])
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 40
      Number of cells:
        quad: 27

    See Also
    --------
    felupe.Mesh.rotate : Rotate a Mesh.
    """

    points = np.array(points)
    dim = points.shape[1]

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.array(center)
    center = center.reshape(1, -1)

    if mask is None:
        mask = slice(None)

    points_rotated = (
        rotation_matrix(angle_deg, dim, axis) @ (points - center).T
    ).T + center

    points_new = points.copy()
    points_new[mask] = points_rotated[mask]

    return points_new, cells, cell_type


@mesh_or_data
def revolve(points, cells, cell_type, n=11, phi=180, axis=0, expand_dim=True):
    """Revolve a 0d-Point to a 1d-Line, a 1d-Line to 2d-Quad or a 2d-Quad to a
    3d-Hexahedron Mesh.

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
    expand_dim : bool, optional
        Expand the dimension of the point coordinates (default is True).

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    Revolve a cylinder from a rectangle.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> rect = fem.Rectangle(a=(0, 4), b=(3, 5), n=(10, 4))
       >>> mesh = fem.mesh.revolve(rect, n=11, phi=180, axis=0)
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 440
      Number of cells:
        hexahedron: 270

    See Also
    --------
    felupe.Mesh.revolve : Revolve a 2d-Quad to a 3d-Hexahedron Mesh.
    """

    points = np.array(points)
    cells = np.array(cells)

    dim = points.shape[1]

    # set new cell-type and the appropriate slice
    cell_type_new, sl = {
        "vertex": ("line", slice(None, None, None)),
        "line": ("quad", slice(None, None, -1)),
        "quad": ("hexahedron", slice(None, None, None)),
        "quad9": ("hexahedron27", slice(None, None, None)),
    }[cell_type]

    if cell_type_new == "hexahedron27":
        n = 2 * n - 1

    if np.isscalar(phi):
        points_phi = np.linspace(0, phi, n)
    else:
        points_phi = np.array(phi)

        if cell_type_new == "hexahedron27":
            points_phi = np.vstack(
                [points_phi, points_phi + np.diff(points_phi, append=np.nan) / 2]
            ).T.ravel()[:-1]

    n = len(points_phi)
    dim_new = dim

    if expand_dim:
        dim_new = dim + 1

    p = np.pad(points, ((0, 0), (0, dim_new - dim)))
    R = rotation_matrix

    points_new = np.vstack(
        [(R(angle, dim_new, axis=axis) @ p.T).T for angle in points_phi]
    )

    c = [cells + len(p) * a for a in np.arange(n)]

    if points_phi[-1] == 360:
        c[-1] = c[0]
        points_new = points_new[: len(points_new) - len(points)]

    if cell_type_new == "hexahedron27":
        one = [0, 1, 2, 3, 18, 19, 20, 21]
        two = [4, 5, 6, 7, 22, 23, 24, 25]
        three = [9, 10, 11, 12, 16, 14, 13, 15, 8, 26, 17]

        cells_new = np.vstack(
            [np.hstack((r, s, t)) for r, s, t in zip(c[:-2:2], c[1:-1:2], c[2::2])]
        )[:, [*one, *two, *three]]

    else:
        cells_new = np.vstack([np.hstack((r, s[:, sl])) for r, s in zip(c[:-1], c[1:])])

    return points_new, cells_new, cell_type_new


@mesh_or_data
def merge_duplicate_points(points, cells, cell_type, decimals=None):
    """Merge duplicate points and update cells of a Mesh.

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

    Notes
    -----
    ..  warning::
        This function re-sorts points.

    ..  note::
        This function does not merge duplicate cells.

    Examples
    --------
    Two quad meshes to be merged overlap some points. Merge these duplicated
    points and update the cells.

    .. pyvista-plot::
       :context:
       :force_static:

       >>> import felupe as fem
       >>>
       >>> rect1 = fem.Rectangle(n=11)
       >>> rect2 = fem.Rectangle(a=(0.9, 0), b=(1.9, 1), n=11)
       >>>
       >>> container = fem.MeshContainer([rect1, rect2])
       >>> stack = fem.mesh.stack(container.meshes)
       >>> mesh = fem.mesh.merge_duplicate_points(stack)
       >>>
       >>> mesh.plot(opacity=0.6).show()

    Each mesh contains 121 points and 100 cells.

    ..  pyvista-plot::
        :context:

        >>> rect1
        <felupe Mesh object>
          Number of points: 121
          Number of cells:
            quad: 100

        >>> rect2
        <felupe Mesh object>
          Number of points: 121
          Number of cells:
            quad: 100

    These two meshes are now stored in a
    :class:`~felupe.MeshContainer`.

    ..  pyvista-plot::
        :context:

        >>> container
        <felupe mesh container object>
          Number of points: 242
          Number of cells:
            quad: 100
            quad: 100

    The meshes of the mesh container are :func:`stacked <felupe.mesh.stack>`.

    ..  pyvista-plot::
        :context:

        >>> stack
        <felupe Mesh object>
          Number of points: 242
          Number of cells:
            quad: 200

    After merging the duplicated points and cells, the number of points is reduced but
    the number of cells is unchanged.

    ..  pyvista-plot::
        :context:

        >>> mesh
        <felupe Mesh object>
          Number of points: 220
          Number of cells:
            quad: 200

    ..  note::
        The :class:`~felupe.MeshContainer` may be directly created with ``merge=True``.
        This enforces :func:`~felupe.mesh.merge_duplicate_points` for the shared points
        array of the container.

    See Also
    --------
    felupe.Mesh.merge_duplicate_points : Merge duplicated points and update cells of a
        Mesh.
    felupe.MeshContainer : A container which operates on a list of meshes with identical
        dimensions.
    """

    if decimals is None:
        points_rounded = points
    else:
        points_rounded = np.round(points, decimals)

    points_new, inverse = np.unique(
        points_rounded,
        return_index=False,
        return_inverse=True,
        return_counts=False,
        axis=0,
    )

    original = np.arange(len(points))

    mask = inverse.ravel() != original
    find = original[mask]
    replace = inverse[mask]

    cells_new = cells.copy()

    for i, j in zip(find, replace):
        cells_new[cells == i] = j

    return points_new, cells_new, cell_type


@mesh_or_data
def merge_duplicate_cells(points, cells, cell_type):
    """Merge duplicate cells of a Mesh.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.

    Returns
    -------
    points : ndarray
        Point coordinates.
    cells : list or ndarray
        Cells with merged duplicate cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Notes
    -----
    ..  warning::
        This function re-sorts cells.

    ..  note::
        This function does not merge duplicate points.

    Examples
    --------
    Two quad meshes to be merged overlap some cells. Merge these duplicated
    points and update the cells.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import felupe as fem
        >>>
        >>> rect1 = fem.Rectangle(n=11)
        >>> rect2 = fem.Rectangle(a=(0.9, 0), b=(1.9, 1), n=11)
        >>>
        >>> container = fem.MeshContainer([rect1, rect2])
        >>> stack = fem.mesh.stack(container.meshes)
        >>> mesh = fem.mesh.merge_duplicate_points(stack)
        >>>
        >>> mesh.plot(opacity=0.6).show()

    Each mesh contains 121 points and 100 cells.

    ..  pyvista-plot::
        :context:

        >>> rect1
        <felupe Mesh object>
          Number of points: 121
          Number of cells:
            quad: 100

        >>> rect2
        <felupe Mesh object>
            Number of points: 121
            Number of cells:
            quad: 100

    These two meshes are now stored in a :class:`~felupe.MeshContainer`.

    ..  pyvista-plot::
        :context:

        >>> container
        <felupe mesh container object>
        Number of points: 242
        Number of cells:
            quad: 100
            quad: 100

    The meshes of the mesh container are :func:`stacked <felupe.mesh.stack>`.

    ..  pyvista-plot::
        :context:

        >>> stack
        <felupe Mesh object>
        Number of points: 242
        Number of cells:
            quad: 200

    After merging the duplicated points and cells, the number of points is reduced but
    the number of cells is unchanged.

    ..  pyvista-plot::
        :context:

        >>> mesh
        <felupe Mesh object>
        Number of points: 220
        Number of cells:
            quad: 200

    ..  note::
        The :class:`~felupe.MeshContainer` may be directly created with ``merge=True``.
        This enforces :func:`~felupe.mesh.merge_duplicate_points` for the shared points
        array of the container.

    The duplicate cells are merged in a second step.

    ..  pyvista-plot::
        :context:

        >>> merged = fem.mesh.merge_duplicate_cells(mesh)
        >>> merged
        <felupe Mesh object>
        Number of points: 220
        Number of cells:
            quad: 190

    See Also
    --------
    felupe.Mesh.merge_duplicate_points : Merge duplicate points of a Mesh.
    felupe.Mesh.merge_duplicate_cells : Merge duplicate cells of a Mesh.
    felupe.MeshContainer : A container which operates on a list of meshes with identical
        dimensions.
    """

    return points, np.unique(cells, axis=0), cell_type


@mesh_or_data
def translate(points, cells, cell_type, move, axis):
    """Translate (move) a Mesh along a given axis.

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

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Circle(n=6)
    >>> mesh.points.min(axis=0), mesh.points.max(axis=0)
    (array([-1., -1.]), array([1., 1.]))

    >>> translated = fem.mesh.translate(mesh, 0.3, axis=1)
    >>> translated.points.min(axis=0), translated.points.max(axis=0)
    (array([-1. , -0.7]), array([1. , 1.3]))

    See Also
    --------
    felupe.Mesh.translate : Translate (move) a Mesh along a given axis.
    """

    points_new = np.array(points)
    points_new[:, axis] += move

    return points_new, cells, cell_type


@mesh_or_data
def flip(points, cells, cell_type, mask=None):
    r"""Ensure positive cell volumes for `tria`, `tetra`, `quad` and `hexahedron` cell
    types.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    mask: list, ndarray or None, optional
        Boolean mask for selected cells to flip (default is None). If None, all cells
        are selected to be flipped.

    Returns
    -------
    points : ndarray
        Point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    A quad mesh with negative cell volumes occurs if one coordinate axis is multiplied
    by -1. The error pops up if a region is created with this mesh.

    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Rectangle(n=3)
    >>> mesh.update(points=mesh.points * np.array([[-1, 1]]))
    >>> region = fem.RegionQuad(mesh)

    The sum of the differential volumes :math:`V = \sum_c \sum_q dV_{qc}` is evaluated
    to -1.0.

    >>> region.dV.sum()
    -1.0

    Let's try to fix the mesh.

    >>> mesh.cells
    array([[0, 1, 4, 3],
           [1, 2, 5, 4],
           [3, 4, 7, 6],
           [4, 5, 8, 7]])

    The cells array is rearranged to ensure positive cell volumes.

    >>> mesh_fixed = fem.mesh.flip(mesh)
    >>> mesh_fixed.cells
    array([[3, 4, 1, 0],
           [4, 5, 2, 1],
           [6, 7, 4, 3],
           [7, 8, 5, 4]])

    A region now correctly evaluates the total volume of the mesh to 1.0.

    >>> region_fixed = fem.RegionQuad(mesh_fixed)
    >>> region_fixed.dV.sum()
    1.0

    See Also
    --------
    felupe.Mesh.flip : Ensure positive cell volumes for `tria`, `tetra`, `quad` and
        `hexahedron` cell types.

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
        cells_new[mask, face] = cells[mask, face[::-1]]

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

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Circle(sections=[0, 90, 180], n=5)
       >>> mesh.plot().show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Circle(sections=[0, 90, 180], n=5)
       >>> fem.mesh.mirror(mesh, normal=[0, 1, 0]).plot().show()

    See Also
    --------
    felupe.Mesh.mirror : Mirror points by plane normal and ensure positive cell volumes
        for `tria`, `tetra`, `quad` and `hexahedron` cell types.

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
    """Join a sequence of meshes with identical cell types.

    Parameters
    ----------
    meshes : list of Mesh
        A list with meshes.

    Returns
    -------
    Mesh
        The joined mesh.

    Notes
    -----
    The ``points``-arrays are vertically stacked. Offsets are added to the  ``cells``-
    arrays of the meshes to refer to the original points.

    Examples
    --------
    Two quad meshes should be joined (merged) into a single mesh.

    >>> import felupe as fem
    >>>
    >>> rect1 = fem.Rectangle(n=11)
    >>> rect2 = fem.Rectangle(a=(0.9, 0), b=(1.9, 1), n=11)
    >>>
    >>> mesh = fem.mesh.concatenate([rect1, rect2])
    >>> mesh.plot(opacity=0.6).show()

    Each mesh contains 121 points and 100 cells.

    >>> rect1
    <felupe Mesh object>
      Number of points: 121
      Number of cells:
        quad: 100

    >>> rect2
    <felupe Mesh object>
      Number of points: 121
      Number of cells:
        quad: 100

    These two meshes are stored in a :class:`~felupe.Mesh`. Note that there are
    duplicate points and cells in the joined mesh.

    >>> mesh
    <felupe Mesh object>
      Number of points: 242
      Number of cells:
        quad: 200
    """

    Mesh = meshes[0].__mesh__

    points = np.vstack([mesh.points for mesh in meshes])
    offsets = np.cumsum(np.insert([mesh.npoints for mesh in meshes][:-1], 0, 0))
    cells = np.vstack(
        [int(offset) + mesh.cells for offset, mesh in zip(offsets, meshes)]
    )
    mesh = Mesh(points=points, cells=cells, cell_type=meshes[0].cell_type)

    return mesh


def stack(meshes):
    """Stack cell-blocks from meshes with identical points-array and cell-types.

    Parameters
    ----------
    meshes : list of Mesh
        A list with meshes. The ``points``-array is taken from the first mesh in the
        list.

    Returns
    -------
    Mesh
        The stacked mesh.

    Notes
    -----
    The ``points``-array is taken from the first mesh. The ``points``-arrays of all
    meshes must be identical. The ``cells``-array of the stacked mesh is created by
    a vertical stack of the ``cells``-arrays.

    Examples
    --------
    Two quad meshes with identical point arrays should be stacked into a single mesh.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle(n=11)
       >>> rect1, rect2 = mesh.copy(), mesh.copy()
       >>> rect1.update(cells=mesh.cells[: 40])
       >>> rect2.update(cells=mesh.cells[-50:])
       >>>
       >>> mesh = fem.mesh.stack([rect1, rect2])
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 121
      Number of cells:
        quad: 90

    See Also
    --------
    felupe.MeshContainer.stack : Stack cell-blocks with same cell-types into a single
        mesh.
    """

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
        Choose a mode how to convert hexahedrons to tets [1]_ (default is 3).

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    Use ``mode=0`` to convert a mesh of hexahedrons into tetrahedrons [1]_.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube(n=6)
       >>> triangulated = fem.mesh.triangulate(mesh, mode=0)
       >>> triangulated.plot().show()

    Use ``mode=3`` to convert a mesh of hexahedrons into tetrahedrons [1]_.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube(n=6)
       >>> triangulated = fem.mesh.triangulate(mesh, mode=3)
       >>> triangulated.plot().show()

    References
    ----------
    .. [1] Dompierre, J., Labbé, P., Vallet, M. G., & Camarero, R. (1999).
       How to Subdivide Pyramids, Prisms, and Hexahedra into Tetrahedra.
       IMR, 99, 195.

    See Also
    --------
    felupe.Mesh.triangulate : Triangulate a quad or a hex mesh.
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
        # mode ... no. of diagonal through hex-point 6.
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

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> rect = fem.Rectangle(a=(-3, -1), b=(3, 1), n=(31, 11))
       >>> mesh = fem.mesh.runouts(rect, axis=1, values=[0.2], normalize=True)
       >>>
       >>> mesh.plot().show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> cube = fem.Cube(a=(-3, -2, -1), b=(3, 2, 1), n=(31, 21, 11))
       >>> mesh = fem.mesh.runouts(cube, axis=2, values=[0.1, 0.3], normalize=True)
       >>>
       >>> mesh.plot().show()

    See Also
    --------
    felupe.Mesh.add_runouts : Add simple rubber-runouts for realistic rubber-metal
        structures.

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
