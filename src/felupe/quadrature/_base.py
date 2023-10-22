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

    def plot(self, plotter, **kwargs):
        """Plot the quadrature points, scaled by their weights, into a given PyVista
        plotter.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        """

        for weight, point in zip(self.weights, self.points):
            plotter.add_points(
                points=np.pad([point], ((0, 0), (0, 3 - self.points.shape[1]))),
                point_size=100 * weight,
                render_points_as_spheres=True,
                color="grey",
            )

        return plotter
