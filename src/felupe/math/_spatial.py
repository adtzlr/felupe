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


def rotation_matrix(alpha_deg, dim=3, axis=0):
    r"""Rotation matrix with given rotation axis and dimension (2d or 3d).

    Parameters
    ----------
    alpha_deg : int
        Rotation angle in degree.
    dim : int, optional (default is 3)
        Dimension of the rotation matrix.
    axis : int, optional (default is 0)
        Rotation axis.

    Returns
    -------
    rotation_matrix : ndarray
        Rotation matrix of dim 2 or 3 with given rotation axis.

    Notes
    -----
    The two-dimensional rotation axis is denoted in Eq. :eq:`rotation-matrix-2d`.

    ..  math::
        :label: rotation-matrix-2d

        \boldsymbol{R}(\alpha) = \begin{bmatrix}
            \cos(\alpha) & -\sin(\alpha) \\
            \sin(\alpha) &  \cos(\alpha)
        \end{bmatrix}

    A three-dimensional rotation matrix is created by inserting zeros in the row and
    column at the given axis of rotation and one at the intersection, see
    Eq. :eq:`rotation-matrix-3d`. If the axis of rotation is the second axis, the two-
    dimensinal rotation matrix is transposed.

    ..  math::
        :label: rotation-matrix-3d

        \boldsymbol{R}(\alpha) = \begin{bmatrix}
            \cos(\alpha) & -\sin(\alpha) & 0 \\
            \sin(\alpha) &  \cos(\alpha) & 0 \\
                  0      &        0      & 1
        \end{bmatrix}

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> R = fem.math.rotation_matrix(alpha_deg=45, dim=2)
    >>> x = np.array([1., 0.])
    >>> y = R @ x
    >>> y
    array([0.70710678, 0.70710678])
    """

    a = np.deg2rad(alpha_deg)
    rotation_matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    if dim == 3:
        if axis == 1:
            rotation_matrix = rotation_matrix.T
        rotation_matrix = np.insert(rotation_matrix, [axis], np.zeros((1, 2)), axis=0)
        rotation_matrix = np.insert(rotation_matrix, [axis], np.zeros((3, 1)), axis=1)
        rotation_matrix[axis, axis] = 1

    return rotation_matrix


def rotate_points(points, angle_deg, axis, center=None, mask=None):
    """Rotate points along a given axis.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
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

    Examples
    --------
    Rotate the points of a rectangle in the xy-plane by 35 degree.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> points = fem.Rectangle(b=(3, 1), n=(10, 4)).points
       >>> points_new = fem.math.rotate_points(
       ...     points, angle_deg=35, axis=2, center=[1.5, 0.5]
       ... )

    See Also
    --------
    felupe.mesh.rotate : Rotate a Mesh.
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

    return points_new


def revolve_points(points, n=11, phi=180, axis=0, expand_dim=True):
    """Revolve points along a given axis.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
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

    Examples
    --------
    Revolve the points of a cylinder from a rectangle.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> points = fem.Rectangle(a=(0, 4), b=(3, 5), n=(10, 4)).points
       >>> points_new = fem.math.revolve_points(mesh.points, n=11, phi=180, axis=0)

    See Also
    --------
    felupe.mesh.revolve : Revolve a 0d-Point to a 1d-Line, a 1d-Line to 2d-Quad or a
        2d-Quad to a 3d-Hexahedron Mesh.
    felupe.Mesh.revolve : Revolve a 2d-Quad to a 3d-Hexahedron Mesh.
    """

    points = np.array(points)
    dim = points.shape[1]

    if np.isscalar(phi):
        points_phi = np.linspace(0, phi, n)
    else:
        points_phi = phi
        n = len(points_phi)

    dim_new = dim
    if expand_dim:
        dim_new = dim + 1

    p = np.pad(points, ((0, 0), (0, dim_new - dim)))

    points_new = np.vstack(
        [(rotation_matrix(angle, dim_new, axis=axis) @ p.T).T for angle in points_phi]
    )

    if points_phi[-1] == 360:
        points_new = points_new[: len(points_new) - len(points)]

    return points_new
