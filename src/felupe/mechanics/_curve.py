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

from ._job import Job
from .plugins import CharacteristicCurvePlugin


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
        :context:
        :force_static:

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
        >>> solid.plot("Principal Values of Cauchy Stress").show()
        >>> fig, ax = job.plot(
        ...    xlabel=r"Displacement $u_1$ in mm $\rightarrow$",
        ...    ylabel=r"Normal Force in $F_1$ in N $\rightarrow$",
        ...    marker="o",
        ... )

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

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
        self._curve = CharacteristicCurvePlugin(boundary=boundary, items=items)
        super().__init__(steps, plugins=[self._curve], callback=callback, **kwargs)

        self.items = self._curve.items
        self.boundary = self._curve.boundary
        self.x = self._curve.x
        self.y = self._curve.y
        self.res = self._curve.res

        self.plot = self._curve.plot
