# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from ..view._backends import backend_of, take_screenshot


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
        """Plot the quadrature points, optionally scaled by their weights, into a
        (optionally provided) plotter of the plotting backend.

        Parameters
        ----------
        plotter : plotter or None, optional
            A plotter of the plotting backend (default is None). If None, a new plotter
            is created.
        point_size : float, optional
            The size of the points (default is 20).
        weighted : bool, optional
            A flag to scale the size of the points by the magnitude of their weights,
            relative to the largest magnitude of the weights (default is False). The
            points with negative weights, e.g. of a :class:`~felupe.TriangleQuadrature`
            of order 3, are drawn in red.
        **kwargs : dict, optional
            Additional keyword-arguments for the plot of the points of the quadrature
            as a mesh, if no plotter is given.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        """

        if plotter is None:
            from ..mesh import Mesh

            mesh = Mesh(self.points, np.zeros((0, 2), dtype=int), "line")
            plotter = mesh.plot(**kwargs)

        backend = backend_of(plotter)
        largest = np.abs(self.weights).max()

        for weight, point in zip(self.weights, self.points):
            point_weight = 1.0
            color = "grey"

            # the size of a point is scaled by the magnitude of its weight (a point
            # size must be positive), negative weights are drawn in red
            if weighted:
                point_weight = abs(weight) / largest
                color = "red" if weight < 0 else "grey"

            plotter = backend.add_points(
                plotter,
                [point],
                point_size=point_size * point_weight,
                color=color,
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

        return take_screenshot(
            self.plot,
            *args,
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
            **kwargs,
        )
