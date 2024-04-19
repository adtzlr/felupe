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

from ..tools import force
from ._job import Job


class CharacteristicCurve(Job):
    r"""A job with a list of steps and a method to evaluate them. Force-displacement
    curve data is tracked during evaluation for a given :class:`~felupe.Boundary` by
    a built-in ``callback``.

    Parameters
    ----------
    steps : list of Step
        A list with steps, where each step subsequently depends on the solution of the
        previous step.
    items : list of SolidBody, SolidBodyNearlyIncompressible, SolidBodyPressure, SolidBodyGravity, PointLoad, MultiPointConstraint, MultiPointContact or None, optional
        A list of items with methods for the assembly of sparse vectors/matrices which
        are used to evaluate the sum of reaction forces. If None, the total reaction
        forces from the :class:`~felupe.tools.NewtonResult` of the substep are used.
    callback : callable, optional
        A callable which is called after each completed substep. Function signature must
        be ``lambda stepnumber, substepnumber, substep, **kwargs: None``, where
        ``substep`` is an instance of :class:`~felupe.tools.NewtonResult`. The field
        container of the completed substep is available as ``substep.x``. Default
        is ``callback=lambda stepnumber, substepnumber, substep, **kwargs: None``.
    **kwargs : dict, optional
        Optional keyword-arguments for the ``callback`` function.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> boundaries = dict()
        >>> boundaries["fixed"] = fem.Boundary(field[0], fx=0, skip=(False, False, False))
        >>> boundaries["clamped"] = fem.Boundary(field[0], fx=1, skip=(True, False, False))
        >>> boundaries["move"] = fem.Boundary(field[0], fx=1, skip=(False, True, True))
        >>>
        >>> umat = fem.NeoHooke(mu=1, bulk=2)
        >>> solid = fem.SolidBody(umat, field)
        >>>
        >>> move = fem.math.linsteps([0, 1], num=5)
        >>> step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
        >>>
        >>> job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"]).evaluate()
        >>> fig, ax = job.plot(
        ...    xlabel=r"Displacement $u_1$ in mm $\rightarrow$",
        ...    ylabel=r"Normal Force in $F_1$ in N $\rightarrow$",
        ...    marker="o",
        ... )
        >>> solid.plot("Principal Values of Cauchy Stress").show()

    See Also
    --------
    Step : A Step with multiple substeps, subsequently depending on the solution
        of the previous substep.
    Job : A job with a list of steps and a method to evaluate them.
    tools.NewtonResult : A data class which represents the result found by
        Newton's method.

    """

    def __init__(
        self,
        steps,
        boundary,
        items=None,
        callback=lambda stepnumber, substepnumber, substep, **kwargs: None,
        **kwargs,
    ):
        super().__init__(steps, self._callback, **kwargs)

        self.items = items
        self.boundary = boundary
        self.x = []
        self.y = []
        self.res = None
        self._cb = callback

    def _callback(self, stepnumber, substepnumber, substep, **kwargs):
        if self.items is not None:
            fun = sum([item.results.force for item in self.items])
        else:
            fun = substep.fun

        self.x.append(substep.x[0].values[self.boundary.points[0]])
        self.y.append(force(substep.x, fun, self.boundary))
        self.res = substep

        self._cb(stepnumber, substepnumber, substep, **kwargs)

    def plot(
        self,
        x=None,
        y=None,
        xaxis=0,
        yaxis=0,
        xlabel=None,
        ylabel=None,
        xscale=1.0,
        yscale=1.0,
        xoffset=0.0,
        yoffset=0.0,
        gradient=False,
        swapaxes=False,
        ax=None,
        items=None,
        **kwargs,
    ):
        """Plot force-displacement characteristic curves on a pre-evaluated job,
        tracked on a given :class:`~felupe.Boundary`.

        Parameters
        ----------
        x : list of ndarray or None, optional
            A list with arrays of displacement data. If None, the displacement is taken
            from the first field of the field container from each completed substep. The
            displacement data is then taken from the first point of the tracked
            :class:`~felupe.Boundary`. Default is None.
        y : list of ndarray or None, optional
            A list with arrays of reaction force data. If None, the force is taken
            from the :class:`~felupe.tools.NewtonResult` of each completed substep.
            Default is None.
        xaxis : int, optional
            The axis for the displacement data (default is 0).
        yaxis : int, optional
            The axis for the reaction force data (default is 0).
        xlabel : str or None, optional
            The label of the x-axis (default is None).
        ylabel : str or None, optional
            The label of the y-axis (default is None).
        xscale : float, optional
            A scaling factor for the displacement data (default is 1.0).
        yscale : float, optional
            A scaling factor the reaction force data (default is 1.0).
        xoffset : float, optional
            An offset for the displacement data (default is 0.0).
        yoffset : float, optional
            An offset for the reaction force data (default is 0.0).
        gradient : bool, optional
            A flag to plot the gradient of the y-data. Uses
            ``numpy.gradient(edge_order=2)``. The gradient data is set to ``np.nan`` for
            absolute values greater than the mean value plus two times the standard
            deviation. Default is False.
        swapaxes : bool, optional
            A flag to flip the plot (x, y) to (y, x). Also changes the labels.
        ax : matplotlib.axes.Axes
            An axes object where the plot is placed in.
        items : slice, ndarray or None
            Indices or a range of data points to plot. If None, all data points are
            plotted (default is None).
        **kwargs : dict
            Additional keyword arguments for plotting in ``ax.plot(**kwags)``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object where the plot is placed in.
        ax : matplotlib.axes.Axes
            The axes object where the plot is placed in.

        """

        if self.res is None:
            raise ValueError(
                "Results are empty. Run `job.evaluate()` and call `job.plot()` again."
            )

        import matplotlib.pyplot as plt

        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if items is None:
            items = slice(None)

        x = np.array(x)[items]
        y = np.array(y)[items]

        if gradient:
            y = np.gradient(y, x[:, xaxis], edge_order=2, axis=0)
            z = np.gradient(y, x[:, xaxis], edge_order=2, axis=0)
            cuttoff = np.mean(abs(z[:, yaxis])) + 2 * np.std(abs(z[:, yaxis]))
            y[abs(z) > cuttoff] = np.nan

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if swapaxes:
            x, y = y, x
            xlabel, ylabel = ylabel, xlabel
            xaxis, yaxis = yaxis, xaxis
            xscale, yscale = yscale, xscale

        ax.plot(
            xoffset + x[:, xaxis] * xscale, yoffset + y[:, yaxis] * yscale, **kwargs
        )

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        return fig, ax
