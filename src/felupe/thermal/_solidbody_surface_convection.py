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


class SolidBodySurfaceConvection:
    r"""Convective heat transfer on the surface of a thermal solid body.

    Parameters
    ----------
    field : felupe.FieldContainer
        Field container with the temperature in °C as first field.
    convection_coefficient : float | callable
        Convection heat transfer coefficient :math: `h_c` in W/(m^2 K). A
        callable requires the parameters 'surface temperature' and 'ambient
        temperature'.
        Additional parameters should be passed by **kwargs (and are not
        adapted during simulation).
    temperature : float
        The ambient air temperature :math:`\theta_\infty` in °C.

    Notes
    -----
    This class represents a boundary condition for a thermal solid body, which
    is used to model convective heat transfer between the boundary of a
    solid material and the adjacent ambient air with temperature
    :math:`\theta_\infty` in °C.

    The the heat flux at the boundary is calculated according to Eq.
    :eq:`convective-flux`.

    Eq. :eq: `example-horizontal-plate` gives an example for the detailed
    calculation of the convective heat transfer coefficient for a warm
    horizontal plate with heat flux upward. `A` is the plate area, `P` is the
    plate perimeter length. `T` denotes temperatures in K, :math:`T_m` is
    the 'film temperature' for which fluid properties are evaluated. `Ra` is
    the Rayleigh number, `Pr` is the Prandtl number (air) and `Nu` is the
    Nusselt number.

    .. math::
       :label: convective-flux

       q_c = h_c\,\left(\theta_s - \theta_\infty\right)

    .. math::
       :label: example-horizontal-plate

       L &= \frac{A}{P}

       T_m &= 0.5 \left(T_s + T_\infty\right)

       \alpha\left(T_m\right) &= \frac{\lambda_\text{air}}{\rho_\text{air}\,cp_{air}}

       Ra &= \frac{g \frac{1}{T_m} \left|\theta_s - \theta_\infty\right| L^3}{\alpha\nu}

       Nu(10^4\leq Ra \leq 10^7) &= 0.54 Ra^{1/4} \text{(Pr > 0.7)  and}

       Nu(10^7 < Ra \leq 10^11) &= 0.15 Ra^{1/3}

       h_c &= \frac{Nu \lambda_\text{air}}{L}


    Examples
    --------
    ..  pyvista-plot::

        >>> import math
        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> def hc_fun(ts, tamb):
        ...     l = 0.25  # slab 1 x 1 m^2
        ...     alpha = 2.25E-05 # m^2/s, air at 300 K
        ...     lam_air = 0.0263  # W/(m K), air at 300 K
        ...     pr = 0.707  # air at 300 K
        ...     t_m = 0.5*(ts + tamb) + 273.15  # K
        ...     ra = (9.81 / t_m * abs(ts - tamb) * l**3)/alpha/1.59E-7
        ...     if (1E04 <= ra <= 1E07) and pr > 0.7:
        ...         nu = 0.54 * math.pow(ra, 0.25)
        ...     elif (1E07 < ra <= 1E11):
        ...         nu = 0.15 * math.pow(ra, 0.33)
        ...     else:
        ...         nu = 0.15 * math.pow(1E11, 0.33)
        ...     return(nu*lam_air/l)
        >>>
        >>> mesh = fem.Rectangle(b=(1.0, 0.25), n=(11, 11))  # rectangle w/ 10x10 cells
        >>> region = fem.RegionQuad(mesh)
        >>> temperature = fem.Field(region, dim=1, values=30.0)
        >>> field = fem.FieldContainer([temperature])
        >>>
        >>> region_convection = fem.RegionQuadBoundary(mesh, mask=mesh.y == 0.25)
        >>> temperature_convection = fem.Field(region_convection, dim=1)
        >>> field_convection = fem.FieldContainer([temperature_convection])
        >>>
        >>> boundaries = fem.BoundaryDict(
        ...     bottom=fem.Boundary(temperature, fy=0, value=30.0),
        ... )
        >>>
        >>> solid = fem.thermal.SolidBodyThermal(
        ...     field=field,
        ...     mass_density=1400.0,  # kg / m^3
        ...     specific_heat_capacity=1000.0,  # J / (kg K)
        ...     time_step=720.0,  # s
        ...     thermal_conductivity=1.0,  # W / (m K)
        ... )
        >>> convection_function = fem.thermal.SolidBodySurfaceConvection(
        ...     field=field_convection,
        ...     convection_coefficient=hc_fun,
        ...     temperature=20.0,  # °C
        ... )
        >>> time = fem.thermal.TimeStep([solid])
        >>> table = fem.math.linsteps([0, 1], num=10)
        >>> air_temperature = fem.math.linsteps([15, 25], num=10)
        >>> hc_const = fem.math.linsteps([4, 4], num=10)
        >>> ramp = {
        ...     time: 18000 * table,  # five hours
        ...     convection_function["temperature"]: air_temperature,
        ...     convection_function["convection_coefficient"]: hc_const,
        ... }
        >>> step = fem.Step(
        ...     items=[time, solid, convection_function], ramp=ramp, boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>>
        >>> mesh.view(
        ...     point_data={"Temperature 2 in °C": temperature.values}
        ... ).plot("Temperature 2 in °C").show()


    See Also
    --------
    felupe.thermal.TimeStep : A time step item.
    felupe.thermal.SolidBodyThermal : A thermal solid body for heat conduction.

    """

    def __init__(self, field, convection_coefficient, temperature):
        self.field = field
        self.convection_coefficient = convection_coefficient  # value or callable
        self.time_step = None

        self.results = Results()
        self.results.temperature = temperature  # ambient temperature in °C

        if callable(convection_coefficient):
            self.results.convection_coefficient =\
                convection_coefficient(
                    self.field.extract(grad=False)[0],  # ts
                    temperature  # tamb
                )
        else:
            self.results.convection_coefficient = convection_coefficient

        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )

    def __getitem__(self, key):
        return UpdateItem(self, key)

    def update(self, temperature):
        self._update_temperature(temperature)
        # self._update_convection_coefficient()  # adapt hc using cur. temp.

    def _update_temperature(self, temperature):
        self.results.temperature = temperature

    def _update_convection_coefficient(self, convection_coefficient):
        if callable(convection_coefficient):
            self.results.convection_coefficient =\
                convection_coefficient(
                    self.field.extract(grad=False)[0],  # ts
                    self.results.temperature  # tamb
                )
        else:
            self.results.convection_coefficient = convection_coefficient

    def _vector(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        if self.time_step is not None and self.time_step == 0:  # inactive time step
            return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))

        temperature = self.field.extract(grad=False)[0]
        fun = [
            -self.results.convection_coefficient
            * (temperature - self.results.temperature)
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
        # temperature = self.field.extract(grad=False)[0]
        fun = [
            -self.results.convection_coefficient
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

# import math

# def hc_fun(ts, tamb):
#     l = 0.25  # slab 1 x 1 m^2
#     alpha = 2.25E-05 # m^2/s, air at 300 K
#     pr = 0.707  # air at 300 K
#     t_m = 0.5*(ts + tamb) + 273.15  # K
#     ra = (9.81 / t_m * abs(ts - tamb) * l**3)/alpha/1.59E-5
#     if (10**4 <= ra <= 10**7) and pr > 0.7:
#         nu = 0.54 * math.pow(ra, 0.25)
#     elif (10**7 < ra <= 10**11):
#         nu = 0.15 * math.pow(ra, 0.33)
#     else:
#         nu = 1
#     return(nu*0.0263/l)
