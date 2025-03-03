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


def interpolate_line(mesh, xi, axis=None, Interpolator=None, **kwargs):
    r"""Return an interpolated line mesh from an existing line mesh with provided
    interpolation points on a given axis.

    Parameters
    ----------
    mesh : Mesh
        A :class:`~felupe.Mesh` with cell-type ``line``.
    xi : ndarray
        Points at which to interpolate data. If axis is None, a normalized curve-
        progress must be provided (:math:`0 \le \xi \le 1`).
    axis : int or None, optional
        The axis of the points at which the data will be interpolated. If None, a
        normalized curve-progress is used. Default is None.
    Interpolator : callable or None, optional
        An interpolator class (default is :class:`scipy.interpolate.PchipInterpolator`).
    **kwargs : dict, optional
        Optional keyword arguments are passed to the Interpolator.

    Returns
    -------
    Mesh
        A new line mesh with interpolated points. The attribute ``points_derivative``
        holds the derivatives of the independent variable w.r.t. the dependent
        variable(s).

    Examples
    --------
    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import felupe as fem
        >>> import numpy as np
        >>> from scipy.interpolate import CubicSpline
        >>>
        >>> mesh = fem.mesh.Line(b=1.0, n=5).expand(n=1)
        >>> t = mesh.x.copy()
        >>> mesh.points[:, 0] = np.sin(2 * np.pi * t)
        >>> mesh.points[:, 1] = np.cos(2 * np.pi * t)
        >>>
        >>> mesh_new = fem.mesh.interpolate_line(
        ...     mesh, xi=np.linspace(0, 1), Interpolator=CubicSpline, bc_type="periodic"
        ... )
        >>>
        >>> mesh_new.plot(
        ...     plotter=mesh.plot(style="points", color="red", point_size=15),
        ...     color="black",
        ... ).show()

    """

    if Interpolator is None:
        from scipy.interpolate import PchipInterpolator as Interpolator

    if axis is None:  # progress-based interpolation

        distances = np.linalg.norm(np.diff(mesh.points, axis=0), axis=1)
        progress = np.insert(np.cumsum(distances), 0, 0)
        progress /= progress.max()

        spline = Interpolator(progress, mesh.points, **kwargs)

        points_new = spline(xi)
        cells_new = np.repeat(np.arange(len(xi)), 2)[1:-1].reshape(-1, 2)

        mesh_new = type(mesh)(points_new, cells_new, cell_type="line")
        mesh_new.points_derivative = spline.derivative()(xi)

    else:  # independent- and dependent-variables

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

        # create a spline
        spline = Interpolator(points[ascending], values[ascending], **kwargs)

        # evaluation points for the independent spline variable
        points_new = np.zeros((len(xi), mesh.dim))
        points_new[:, axis] = xi
        points_new[:, axes[mask]] = spline(xi)

        cells_new = np.repeat(np.arange(len(xi)), 2)[1:-1].reshape(-1, 2)

        mesh_new = type(mesh)(points_new, cells_new, cell_type="line")
        mesh_new.points_derivative = np.zeros_like(mesh_new.points)
        mesh_new.points_derivative[:, axes[mask]] = spline.derivative()(xi)

    return mesh_new
