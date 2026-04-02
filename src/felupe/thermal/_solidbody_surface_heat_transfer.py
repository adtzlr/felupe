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
from ..mechanics import Assemble, Results, UpdateItem


class SolidBodySurfaceHeatTransfer:
    r"""A surface boundary condition for a thermal solid body.

    Parameters
    ----------
    field : felupe.FieldContainer
        The field container with the temperature as first field.
    coefficient : float
        The convection coefficient :math:`h` in W/(m^2 K).
    temperature : float
        The ambient temperature :math:`T_\infty` in °C or K.

    Notes
    -----
    This class represents a boundary condition for a thermal solid body, which
    is used to model heat transfer (convection, radiation) at the boundary of a
    solid material. The coefficient is used to calculate the heat flux at the
    boundary based on the difference between the temperature at the boundary
    and the ambient temperature.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> mesh = fem.Rectangle(n=11)
        >>> region = fem.RegionQuad(mesh)
        >>> temperature = fem.Field(region, dim=1)
        >>> field = fem.FieldContainer([temperature])
        >>>
        >>> region_heat_transfer = fem.RegionQuadBoundary(mesh, mask=mesh.x == 1.0)
        >>> temperature_heat_transfer = fem.Field(region_heat_transfer, dim=1)
        >>> field_heat_transfer = fem.FieldContainer([temperature_heat_transfer])
        >>>
        >>> boundaries = fem.BoundaryDict(
        ...     left=fem.Boundary(temperature, fx=0),
        ... )
        >>>
        >>> solid = fem.thermal.SolidBodyThermal(
        ...     field=field,
        ...     mass_density=1400.0,  # kg/m^3
        ...     specific_heat_capacity=1000.0,  # J/(kg*K)
        ...     time_step=720.0,  # s
        ...     thermal_conductivity=1.0,  # W/(m*K)
        ... )
        >>> heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
        ...     field=field_heat_transfer,
        ...     coefficient=7.69,  # W/(m^2 K)
        ...     temperature=10.0,  # °C
        ... )
        >>> time = fem.thermal.TimeStep([solid])
        >>> table = fem.math.linsteps([0, 1], num=15)
        >>> air_temperature = fem.math.linsteps([0, 40], num=15)  # air temperature
        >>> coefficient = fem.math.linsteps([7.0, 8.0], num=15)  # heat transfer coeff.
        >>> ramp = {
        ...     boundaries["left"]: 10 * table,  # surface temperature
        ...     time: 18000 * table,  # five hours
        ...     heat_transfer["temperature"]: air_temperature,
        ...     heat_transfer["coefficient"]: coefficient,
        ... }
        >>> step = fem.Step(
        ...     items=[time, solid, heat_transfer], ramp=ramp, boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate(
        ...     # filename="result.xdmf",  # result file for Paraview
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
    felupe.thermal.SolidBodyHeatFlux : A thermal solid body for heat flux.

    """

    def __init__(self, field, coefficient, temperature):
        self.field = field
        self.time_step = None

        self.results = Results()
        self.results.temperature = temperature
        self.results.coefficient = coefficient

        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )

    def __getitem__(self, key):
        return UpdateItem(self, key)

    def update(self, temperature):
        self._update_temperature(temperature)

    def _update_temperature(self, temperature):
        self.results.temperature = temperature

    def _update_coefficient(self, coefficient):
        self.results.coefficient = coefficient

    def _vector(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        if self.time_step is not None and self.time_step == 0:  # inactive time step
            return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))

        temperature = self.field.extract(grad=False)[0]
        fun = [-self.results.coefficient * (temperature - self.results.temperature)]

        self.results.force = IntegralForm(
            fun=fun, v=self.field, dV=self.field.region.dV, grad_v=[False]
        ).assemble(**kwargs)

        return self.results.force

    def _matrix(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        if self.time_step is not None and self.time_step == 0:  # inactive time step
            return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))

        dim = self.field[0].dim
        fun = [-self.results.coefficient * np.eye(dim).reshape(dim, dim, 1, 1)]

        self.results.stiffness = IntegralForm(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
            grad_u=[False],
        ).assemble(**kwargs)

        return self.results.stiffness
