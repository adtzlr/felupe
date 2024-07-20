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


def interpolate_line(mesh, xi, axis, Interpolator=None, **kwargs):
    r"""Return an interpolated line mesh from an existing line mesh with provided
    interpolation points on a given axis.

    Parameters
    ----------
    mesh : Mesh
        A :class:`~felupe.Mesh` with cell-type ``line``.
    xi : ndarray
        Points at which to interpolate data.
    axis : int
        The axis of the points at which the data will be interpolated.
    Interpolator : callable or None, optional
        An interpolator class (default is :class:`scipy.interpolate.PchipInterpolator`).
    **kwargs : dict, optional
        Optional keyword arguments are passed to the interpolator.

    Returns
    -------
    Mesh
        A new line mesh with interpolated points.

    Examples
    --------
    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> mesh = fem.mesh.Line(n=5).expand(n=1)
        >>> t = mesh.x.copy()
        >>> mesh.points[:, 0] = np.sin(np.pi / 2 * t)
        >>> mesh.points[:, 1] = np.cos(np.pi / 2 * t)
        >>>
        >>> mesh.plot(style="points", color="black").show()

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> mesh_new = interpolate_line(mesh, xi=np.linspace(0, 1, 101), axis=1)
        >>>
        >>> mesh_new.plot(style="points", color="black").show()
    """

    # line connectivity
    # concatenate the first point of each cell and the last point of the last cell
    line = np.concatenate([mesh.cells[:, :-1].ravel(), mesh.cells[-1, -1:]])

    # independent spline variable
    points = mesh.points[line, axis]
    ascending = np.argsort(points)

    # dependent spline variable(s)
    mask = np.ones(mesh.dim, dtype=bool)
    mask[axis] = False

    axes = np.arange(mesh.dim)
    values = mesh.points[line.reshape(-1, 1), axes[mask]]

    # spline interpolator
    if Interpolator is None:
        from scipy.interpolate import PchipInterpolator as Interpolator

    # create a spline
    spline = Interpolator(points[ascending], values[ascending], **kwargs)

    # evaluation points for the independent spline variable
    points_new = np.zeros((len(xi), mesh.dim))
    points_new[:, axis] = xi
    points_new[:, axes[mask]] = spline(xi)

    cells_new = np.repeat(np.arange(len(xi)), 2)[1:-1].reshape(-1, 2)

    return type(mesh)(points_new, cells_new, cell_type="line")
