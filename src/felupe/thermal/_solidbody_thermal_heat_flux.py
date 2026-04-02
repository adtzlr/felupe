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

from ..assembly import IntegralForm
from ..mechanics import Assemble, Results


class SolidBodyThermalHeatFlux:
    r"""A thermal heat flux boundary condition for a thermal solid body.

    Parameters
    ----------
    field : felupe.FieldContainer
        The field container with the temperature as first field.
    coefficient : float
        The heat flux coefficient :math:`q` in W/m^2.

    Notes
    -----
    This class represents a thermal heat flux boundary condition for a thermal solid
    body, which is used to model heat flux at the boundary of a solid material.
    The heat flux coefficient is used to calculate the heat flux at the boundary.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=11)
        >>> region = fem.RegionQuad(mesh)
        >>> temperature = fem.Field(region, dim=1)
        >>> field = fem.FieldContainer([temperature])
        >>>
        >>> region_flux = fem.RegionQuadBoundary(mesh, mask=mesh.y == 1.0)
        >>> temperature_flux = fem.Field(region_flux, dim=1)
        >>> field_flux = fem.FieldContainer([temperature_flux])
        >>>
        >>> boundaries = fem.BoundaryDict(
        ...     left=fem.Boundary(temperature, fx=0),
        ...     right=fem.Boundary(temperature, fx=1),
        ... )
        >>>
        >>> solid = fem.thermal.SolidBodyThermal(
        ...     field=field,
        ...     mass_density=1.0,  # kg/m^3
        ...     specific_heat_capacity=1.0,  # J/(kg*K)
        ...     time_step=0.01,  # s
        ...     thermal_conductivity=1.0,  # W/(m*K)
        ... )
        >>> heat_flux = fem.thermal.SolidBodyThermalHeatFlux(
        ...     field=field_flux,
        ...     heat_flux=1.0,  # W/m^2
        ... )
        >>> time = fem.thermal.TimeStep([solid])
        >>> table = fem.math.linsteps([0, 1], num=10)
        >>> ramp = {
        ...     boundaries["right"]: 10 * table,
        ...     time: 0.1 * table,
        ...     heat_flux: 10 * table,
        ... }
        >>> step = fem.Step(
        ...     items=[time, solid, heat_flux], ramp=ramp, boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate(
        ...     filename="result.xdmf",  # result file for Paraview
        ...     point_data={"Temperature": lambda field, substep: temperature.values},
        ...     point_data_default=False,
        ...     cell_data_default=False,
        ... )
        >>>
        >>> mesh.view(
        ...     point_data={"Temperature in K": temperature.values}
        ... ).plot("Temperature in K").show()

    See Also
    --------
    felupe.thermal.TimeStep : A time step item.
    felupe.thermal.SolidBodyThermal : A thermal solid body for heat conduction.
    felupe.thermal.SolidBodySurfaceHeatTransfer : A thermal solid body for heat
        convection.

    """

    def __init__(self, field, heat_flux=None):
        self.field = field
        self.time_step = 0

        self.assemble = Assemble(vector=self._vector, matrix=None, multiplier=-1.0)
        self.results = Results()

        if heat_flux is not None:
            self.results.heat_flux = heat_flux

    def update(self, heat_flux):
        self.results.heat_flux = heat_flux

    def _vector(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        if self.time_step is not None and self.time_step == 0:  # inactive time step
            return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))

        fun = [-self.results.heat_flux * np.ones((1, 1))]

        self.results.force = IntegralForm(
            fun=fun, v=self.field, dV=self.field.region.dV, grad_v=[False]
        ).assemble(**kwargs)

        return self.results.force
