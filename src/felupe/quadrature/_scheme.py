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


class Scheme:
    r"""A quadrature scheme with integration points :math:`x_q` and weights :math:`w_q`.
    It approximates the integral of a function over a region :math:`V` by a weighted sum
    of function values :math:`f_q = f(x_q)`, evaluated on the quadrature-points.

    Notes
    -----

    The approximation is given by

    ..  math::

        \int_V f(x) dV \approx \sum f(x_q) w_q

    with quadrature points :math:`x_q` and corresponding weights :math:`w_q`.
    """

    def __init__(self, points, weights):
        self.points = points
        self.weights = weights

        self.npoints, self.dim = self.points.shape

    def plot(self, plotter=None, point_size=20, weighted=False, **kwargs):
        """Plot the quadrature points, scaled by their weights, into a (optionally
        provided) PyVista plotter.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        """

        if plotter is None:
            from ..mesh import Mesh

            mesh = Mesh(self.points, np.zeros((0, 2), dtype=int), "line")
            plotter = mesh.plot(**kwargs)

        for weight, point in zip(self.weights, self.points):
            # plotter requires 3d-point coordinates
            points = np.pad([point], ((0, 0), (0, 3 - self.dim)))

            if weighted:
                point_weight = weight / self.weights.max()
            else:
                point_weight = 1.0

            plotter.add_points(
                points=points,
                point_size=point_size * point_weight,
                color="grey",
            )

        return plotter

    def screenshot(
        self,
        *args,
        filename=None,
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the quadrature.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        if filename is None:
            filename = "quadrature.png"

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )
