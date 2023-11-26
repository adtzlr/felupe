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
    "A quadrature scheme."

    def __init__(self, points, weights):
        "Quadrature scheme with integration `points` and `weights`."

        self.points = points
        self.weights = weights

        self.npoints, self.dim = self.points.shape

    def plot(self, plotter=None, add_axes=True, point_size=100, **kwargs):
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

            plotter.add_points(
                points=points,
                point_size=point_size * weight / self.weights.max(),
                render_points_as_spheres=True,
                opacity=0.8,
                color="grey",
            )

        if add_axes:
            plotter.add_axes(xlabel="r", ylabel="s", zlabel="t")

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
