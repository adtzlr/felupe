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
from scipy.sparse import csr_matrix

from ..mechanics import Assemble, Results


class TimeStep:
    r"""
    A time step item.

    Parameters
    ----------
    items : list of felupe.thermal.SolidBodyThermal
        List of items to be updated at each time step.
    time_old : float, optional
        Initial time (default is 0.0).
    time_new : float or None, optional
        New time (default is None).
    time_step_min : float, optional
        Minimum time step to avoid numerical issues (default is
        :math:`\sqrt{\epsilon}`).

    Notes
    -----
    This class is used to update the time step for each item in the list of items. The
    time step is calculated as the difference between the new time and the old time, and
    is assigned to each item in the list.

    .. note::

       It is important to include this :class:`~felupe.thermal.TimeStep` item in a step
       as **first item**, so that the time step is updated before the thermal solid body
       is evaluated.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=11).triangulate()
        >>> region = fem.RegionTriangle(mesh)
        >>> temperature = fem.Field(region, dim=1)
        >>> field = fem.FieldContainer([temperature])
        >>>
        >>> solid = fem.thermal.SolidBodyThermal(
        ...     field,
        ...     mass_density=1.0,
        ...     specific_heat_capacity=1.0,
        ...     thermal_conductivity=1.0,
        ... )
        >>>
        >>> time = fem.thermal.TimeStep(items=[solid])
        >>> table = fem.math.linsteps([0, 1], num=100)
        >>>
        >>> step = fem.Step(items=[time, solid], ramp={time: table})

    See Also
    --------
    felupe.thermal.SolidBodyThermal : A thermal solid body.

    """

    def __init__(
        self,
        items,
        time_old=0.0,
        time_new=None,
        time_step_min=np.finfo(float).eps ** 0.5,
    ):
        self.field = items[0].field  # get dummy field from first item
        self.assemble = Assemble()
        self.results = Results()
        self.time_old = time_old
        self.time_new = time_new
        self.items = items
        self.time_step_min = time_step_min

    def update(self, time_new):
        time_step = time_new - self.time_old
        time_step = np.maximum(self.time_step_min, time_step)

        for item in self.items:
            item.time_step = time_step

        self.time_old = time_new
