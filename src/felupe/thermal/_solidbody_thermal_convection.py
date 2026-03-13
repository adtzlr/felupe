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

from ..assembly import IntegralForm
from ..mechanics import Assemble, Results


class SolidBodyThermalConvection:
    r"""A thermal convection boundary condition for a thermal solid body.

    Parameters
    ----------
    field : felupe.FieldContainer
        The field container with the temperature as first field.
    coefficient : float
        The convection coefficient :math:`h` in W/(m^2*K).
    temperature : float
        The ambient temperature :math:`T_\infty` in K.

    Notes
    -----
    This class represents a thermal convection boundary condition for a thermal solid
    body, which is used to model heat convection at the boundary of a solid material.
    The convection coefficient is used to calculate the heat flux at the boundary based
    on the difference between the temperature at the boundary and the ambient
    temperature.

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
        >>> convection = fem.thermal.SolidBodyThermalConvection(
        ...     field=field_convection,
        ...     coefficient=1.0,  # W/(m^2*K)
        ...     temperature=10.0,  # K
        ... )
        >>> time = fem.thermal.TimeStep([solid])
        >>> table = fem.math.linsteps([0, 1], num=10)
        >>> ramp = {
        ...     boundaries["right"]: 10 * table,
        ...     time: 0.1 * table,
        ...     convection: 100 * table,
        ... }
        >>> step = fem.Step(
        ...     items=[time, solid, convection], ramp=ramp, boundaries=boundaries
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

    """

    def __init__(self, field, coefficient, temperature):
        self.field = field

        self.results = Results()
        self.results.temperature = temperature

        if coefficient is not None:
            self.results.coefficient = coefficient

        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )

    def update(self, temperature):
        self.results.temperature = temperature

    def _vector(self, field=None, parallel=False, resize=None):
        if field is not None:
            self.field = field

        temperature = self.field.extract(grad=False)[0]
        fun = [-self.results.coefficient * (temperature - self.results.temperature)]

        self.results.force = IntegralForm(
            fun=fun, v=self.field, dV=self.field.region.dV, grad_v=[False]
        ).assemble(parallel=parallel)

        if resize is not None:
            self.results.force.resize(*resize.shape)

        return self.results.force

    def _matrix(self, field=None, parallel=False, resize=None):
        if field is not None:
            self.field = field

        dim = self.field[0].dim
        fun = [-self.results.coefficient * np.eye(dim).reshape(dim, dim, 1, 1)]

        self.results.stiffness = IntegralForm(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
            grad_u=[False],
        ).assemble(parallel=parallel)

        if resize is not None:
            self.results.stiffness.resize(*resize.shape)

        return self.results.stiffness
