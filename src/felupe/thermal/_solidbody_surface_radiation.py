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
from scipy.constants import sigma
from scipy.sparse import csr_matrix

from ..assembly import IntegralForm
from ..mechanics import Assemble, Results, UpdateItem


class SolidBodySurfaceRadiation:
    r"""Long wave radiative heat transfer on the surface of a thermal solid body.

    Parameters
    ----------
    field : felupe.FieldContainer
        Field container with the temperature in °C as first field.
    emissivity : float
        Emissivity :math:`\varepsilon` of the surface (dimensionless,
        :math:`0 \le \varepsilon \le 1`).
    temperature : float
        The surrounding temperature :math:`\theta_{sur}` in °C.

    Notes
    -----
    This class represents a boundary condition for a thermal solid body, which
    is used to model long wave radiative heat transfer between the boundary of a
    solid material and a surrounding hemispherical body with an emissivity of
    :math: `\epsilon_{sur}` = 1.0 and temperature :math:`\theta_{sur}` in °C.

    The hemispherical long wave surface emissivity :math: `\epsilon`is used to
    define the corresponding solid surface property.

    The the heat flux at the boundary is calculated according to Eq.
    :eq:`long-wave-radiation-flux` where 'T' denotes temperatures in K.
    
    .. math::
       :label: long-wave-radiation-flux

       q_r = \epsilon\,\sigma_B\,\left(T_s^4 - T_{sur}^4\right)

    ..  note:

        A thermal analysis must use temperatures in °C, if
        :class:`~felupe.thermal.SolidBodySurfaceRadiation` is included.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> mesh = fem.Rectangle(n=11)
        >>> region = fem.RegionQuad(mesh)
        >>> temperature = fem.Field(region, dim=1, values=20.0)
        >>> field = fem.FieldContainer([temperature])
        >>>
        >>> region_radiation = fem.RegionQuadBoundary(mesh, mask=mesh.x == 1.0)
        >>> temperature_radiation = fem.Field(region_radiation, dim=1)
        >>> field_radiation = fem.FieldContainer([temperature_radiation])
        >>>
        >>> boundaries = fem.BoundaryDict(
        ...     left=fem.Boundary(temperature, fx=0, value=20.0),
        ... )
        >>>
        >>> solid = fem.thermal.SolidBodyThermal(
        ...     field=field,
        ...     mass_density=1400.0,  # kg / m^3
        ...     specific_heat_capacity=1000.0,  # J / (kg K)
        ...     time_step=720.0,  # s
        ...     thermal_conductivity=1.0,  # W / (m K)
        ... )
        >>> radiation = fem.thermal.SolidBodySurfaceRadiation(
        ...     field=field_radiation,
        ...     emissivity=0.8,
        ...     temperature=20.0,  # °C
        ... )
        >>> time = fem.thermal.TimeStep([solid])
        >>> table = fem.math.linsteps([0, 1], num=15)
        >>> air_temperature = fem.math.linsteps([20, 40], num=15)  # air temperature
        >>> emissivity = fem.math.linsteps([0.6, 0.8], num=15)  # a value between 0 ... 1
        >>> ramp = {
        ...     time: 18000 * table,  # five hours
        ...     radiation["temperature"]: air_temperature,
        ...     radiation["emissivity"]: emissivity,
        ... }
        >>> step = fem.Step(
        ...     items=[time, solid, radiation], ramp=ramp, boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>>
        >>> mesh.view(
        ...     point_data={"Temperature in °C": temperature.values}
        ... ).plot("Temperature in °C").show()

    See Also
    --------
    felupe.thermal.TimeStep : A time step item.
    felupe.thermal.SolidBodyThermal : A thermal solid body for heat conduction.

    """

    def __init__(self, field, emissivity, temperature):
        self.field = field
        self.time_step = None

        self.results = Results()
        self.results.temperature = temperature  # ambient temperature in °C
        self.results.emissivity = emissivity
        self.results.radiation_coefficient =\
                          4 * emissivity * sigma * (temperature + 273.15) ** 3

        self._sigma = sigma  # Stefan-Boltzmann constant

        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )

    def __getitem__(self, key):
        return UpdateItem(self, key)

    def update(self, temperature):
        self._update_temperature(temperature)

    def _update_temperature(self, temperature):
        self.results.temperature = temperature

    def _update_emissivity(self, emissivity):
        self.results.emissivity = emissivity

    def _vector(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        if self.time_step is not None and self.time_step == 0:  # inactive time step
            return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))

        temperature = self.field.extract(grad=False)[0]
        fun = [
            -self.results.emissivity
            * self._sigma
            * ((temperature + 273.15) ** 4 - (self.results.temperature + 273.15) ** 4)
        ]

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
        temperature = self.field.extract(grad=False)[0]
        self.results.radiation_coefficient = 4 * self.results.emissivity\
            * self._sigma\
            * ((temperature + self.results.temperature)/2 + 273.15) ** 3

        fun = [
            -self.results.radiation_coefficient
            # -self.results.emissivity
            # * self._sigma
            # * 4
            # * (temperature + 273.15) ** 3
            * np.eye(dim).reshape(dim, dim, 1, 1)
        ]

        self.results.stiffness = IntegralForm(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
            grad_u=[False],
        ).assemble(**kwargs)

        return self.results.stiffness
