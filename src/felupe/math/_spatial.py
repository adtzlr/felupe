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
