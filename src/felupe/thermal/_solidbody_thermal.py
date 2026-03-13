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
from scipy.sparse import csr_array, csr_matrix, diags

from ..assembly import IntegralForm
from ..constitution import Laplace
from ..mechanics import SolidBody


class SolidBodyThermal(SolidBody):
    r"""
    A thermal solid body.

    Parameters
    ----------
    field : felupe.FieldContainer
        The field container containing the temperature field.
    mass_density : float
        The mass density :math:`\rho` of the material.
    specific_heat_capacity : float
        The specific heat capacity :math:`c_p` of the material.
    thermal_conductivity : float
        The thermal conductivity :math:`k` of the material.
    time_step : float or None, optional
        The time step :math:`\Delta t` (default is None).
    model : None or felupe.Laplace, optional
        A model for the thermal conductivity. Default is None, which defaults to
        :class:`~felupe.Laplace`.
    lumped_capacity : bool, optional
        A flag to use a lumped instead of a consistent capacity matrix (default is
        True).

    Notes
    -----
    This class represents a thermal solid body, which is used to model heat conduction
    in a solid material. The thermal conductivity is modeled using the specified model,
    which defaults to the Laplace model. The time step is used to update the temperature
    field at each time step, and the capacity matrix is assembled based on the mass
    density and specific heat capacity of the material.

    An implicit time integration scheme is used, where the stiffness matrix is updated
    at each time step to include the contribution from the capacity matrix. The force
    vector is also updated to include the contribution from the temperature rate, which
    is calculated as the difference between the new temperature and the old temperature
    divided by the time step, see Eq. :eq:`thermal-solid-body`.

    ..  math::
        :label: thermal-solid-body

        \boldsymbol{K} \boldsymbol{T} &= -\boldsymbol{r} \\
    
        \boldsymbol{r} &= \boldsymbol{r}_{\text{conductivity}}
            + \boldsymbol{C} \frac{\boldsymbol{T} - \boldsymbol{T}_n}{\Delta t} \\

        \boldsymbol{K} &= \boldsymbol{K}_{\text{conductivity}}
            + \frac{\boldsymbol{C}}{\Delta t}

    
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
        >>> time = fem.thermal.TimeStep([solid])
        >>> table = fem.math.linsteps([0, 1], num=10)
        >>> ramp = {boundaries["right"]: 10 * table, time: 0.1 * table}
        >>> step = fem.Step(items=[time, solid], ramp=ramp, boundaries=boundaries)
        >>> job = fem.Job(steps=[step]).evaluate(
        ...     filename="result.xdmf",  # result file for Paraview
        ...     point_data={"Temperature": lambda field, substep: temperature.values},
        ...     point_data_default=False,
        ...     cell_data_default=False,
        ... )
        >>>
        >>> view = mesh.view(point_data={"Temperature in K": temperature.values})
        >>> view.plot("Temperature in K").show()

    See Also
    --------
    felupe.thermal.TimeStep : A time step item.
    felupe.thermal.SolidBodyThermalConvection : A thermal convection boundary condition.

    """

    def __init__(
        self,
        field,
        mass_density,
        specific_heat_capacity,
        thermal_conductivity,
        time_step=None,
        model=None,
        lumped_capacity=True,
    ):
        if model is None:
            model = Laplace

        super().__init__(
            umat=model(thermal_conductivity),
            field=field,  # the field container containing the temperature field
            density=mass_density * specific_heat_capacity,  # volumetric heat capacity
            statevars=field[0].values.copy(),  # initial temperature values
        )
        self.time_step = time_step

        # assemble capacity matrix
        self.capacity = self._mass()

        self.evaluate.heat_flux = self.evaluate.stress

        if lumped_capacity:
            self.capacity = diags(csr_array(self.capacity).sum(axis=1))

    def _vector(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        self.results.stress = self.results.heat_flux = self._gradient(field)
        self.results._statevars = self.field[0].values.copy()  # new temperature

        self.results.force = IntegralForm(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        ).assemble(**kwargs)

        if self.time_step > 0:
            temperature_old = self.results.statevars  # old temperature
            temperature_new = self.results._statevars  # new temperature

            temperature_rate = (temperature_new - temperature_old) / self.time_step

            self.results.force += csr_matrix(
                self.capacity @ temperature_rate.reshape(-1, 1)
            )

        return self.results.force

    def _matrix(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        self.results.elasticity = self._hessian(field)
        form = IntegralForm(
            fun=self.results.elasticity,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        self.results.stiffness_values = form.integrate(
            out=self.results.stiffness_values, **kwargs
        )

        self.results.stiffness = form.assemble(values=self.results.stiffness_values)

        if self.time_step > 0:
            self.results.stiffness += self.capacity / self.time_step

        return self.results.stiffness
