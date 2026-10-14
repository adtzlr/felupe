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

from ..plugins import CharacteristicCurvePlugin
from ._job import Job


class CharacteristicCurve(Job):
    r"""A job with a list of steps and a method to evaluate them. Force-displacement
    curve data is tracked during evaluation for a given :class:`~felupe.Boundary` by
    a built-in :class:`~felupe.CharacteristicCurvePlugin`.

    Parameters
    ----------
    steps : list of Step
        A list with steps, where each step subsequently depends on the solution of the
        previous step.
    boundary : Boundary
        The boundary condition, whose displacements and reaction forces are tracked.
    items : list of SolidBody, SolidBodyNearlyIncompressible, SolidBodyPressure, SolidBodyGravity, PointLoad, MultiPointConstraint, MultiPointContact or None, optional
        A list of items with methods for the assembly of sparse vectors/matrices which
        are used to evaluate the sum of reaction forces. If None, the total reaction
        forces from the :class:`~felupe.tools.NewtonResult` of the substep are used.
    plugins : list or None, optional
        A list of plugins, see :class:`~felupe.Job` (default is None). The
        :class:`~felupe.CharacteristicCurvePlugin` is added to the end of the list.

        ..  versionchanged:: 12.0.0
            ``plugins`` is a keyword-only argument. The arguments ``callback`` and
            ``**kwargs`` (for the callback) are removed, a callable plugin replaces the
            callback, see the notes of :class:`~felupe.Job`.

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

    def __init__(self, steps, boundary, items=None, *, plugins=None):
        self._curve = CharacteristicCurvePlugin(boundary=boundary, items=items)

        plugins = [*(plugins or []), self._curve]
        super().__init__(steps, plugins=plugins)

        self.items = self._curve.items
        self.boundary = self._curve.boundary

        self.plot = self._curve.plot

    @property
    def x(self):
        return self._curve.x

    @property
    def y(self):
        return self._curve.y

    @property
    def res(self):
        return self._curve.res
