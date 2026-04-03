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
from scipy.sparse import csr_array, csr_matrix, diags

from ..assembly import IntegralForm
from ..constitution import Laplace
from ..math import dot, norm
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
        The time step :math:`\Delta t` If None, the stationary solution will be
        calculated. If a float is provided, an implicit time integration scheme will be
        used. Default is None.
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

        \boldsymbol{r}
            + \frac{\partial \boldsymbol{r}}{\partial \boldsymbol{T}}
            \delta \boldsymbol {T} = \boldsymbol{0}

        \boldsymbol{K} \delta \boldsymbol{T} &= -\boldsymbol{r}

        \boldsymbol{r} &= \boldsymbol{r}_{\text{conductivity}}
            + \boldsymbol{C} \frac{\boldsymbol{T} - \boldsymbol{T}_n}{\Delta t}

        \boldsymbol{K} &= \boldsymbol{K}_{\text{conductivity}}
            + \frac{\boldsymbol{C}}{\Delta t}


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
        >>> mesh.view(
        ...     point_data={"Temperature in K": temperature.values}
        ... ).plot("Temperature in K").show()
        >>>
        >>> mesh.view(
        ...     cell_data={"Heat Flux": solid.results.heat_flux[0][0].mean(axis=-2).T}
        ... ).plot("Heat Flux", component=0).show()
        >>>
        >>> flux = solid.heat_flux_boundary(
        ...     region=fem.RegionQuadBoundary(mesh, mask=mesh.x == 1.0),
        ...     normal=True,  # normal component of heat flux
        ...     total=True,
        ...     mean=True,
        ... )
        >>> assert np.isclose(flux.round(1), -30.5)

    See Also
    --------
    felupe.thermal.TimeStep : A time step item.
    felupe.thermal.SolidBodySurfaceHeatTransfer : A thermal convection boundary condition.
    felupe.thermal.SolidBodyHeatFlux : A thermal heat flux boundary condition.

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
        )

        self.time_step = time_step

        # assemble capacity matrix
        self.capacity = self._mass()

        if lumped_capacity:
            self.capacity = diags(csr_array(self.capacity).sum(axis=1))

    def _vector(self, field=None, **kwargs):
        # check if state variables need to be initialized
        # a felupe.mechanics.SolidBody creates a zero-sized array by default
        init_statevars_required = self.results.statevars.size == 0

        if field is not None:
            self.field = field
            if init_statevars_required:  # initial temperature
                self.results.statevars = self.field[0].values.copy()

        else:  # field is None
            if init_statevars_required:
                raise ValueError("Provide a field to obtain the initial temperature.")

        if self.time_step == 0:  # inactive time step, only capacity contribution
            self.results._statevars = self.field[0].values.copy()  # new temperature
            temperature_old = self.results.statevars  # old temperature
            temperature_new = self.results._statevars  # new temperature
            temperature_diff = (temperature_new - temperature_old).reshape(-1, 1)

            self.results.force = csr_matrix(self.capacity @ temperature_diff)
            return self.results.force

        self.results.stress = self.results.heat_flux = self._gradient(field)
        self.results._statevars = self.field[0].values.copy()  # new temperature

        self.results.force = IntegralForm(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        ).assemble(**kwargs)

        temperature_old = self.results.statevars  # old temperature
        temperature_new = self.results._statevars  # new temperature
        temperature_diff = (temperature_new - temperature_old).reshape(-1, 1)

        print(temperature_old.min(), temperature_new.min())

        if self.time_step is not None and self.time_step > 0:
            temperature_rate = temperature_diff / self.time_step
            self.results.force += csr_matrix(self.capacity @ temperature_rate)

        return self.results.force

    def _matrix(self, field=None, **kwargs):
        if field is not None:
            self.field = field

        if self.time_step == 0:  # inactive time step, only capacity contribution
            self.results.stiffness = self.capacity
            return self.results.stiffness

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

        if self.time_step is not None and self.time_step > 0:
            self.results.stiffness += self.capacity / self.time_step

        return self.results.stiffness

    def heat_flux(self, field=None, **kwargs):
        return -self.evaluate.stress(field=field, **kwargs)[0]

    def heat_flux_boundary(
        self,
        field=None,
        region=None,
        integrate=True,
        mean=True,
        **kwargs,
    ):
        """Calculate the heat flux or the integrated heat transfer rate on a boundary
        region.

        Parameters
        ----------
        field : felupe.FieldContainer or None, optional
            The field container with the temperature field as first field on which to
            calculate the heat flux. If None, a field will be created on the provided
            region and linked to the body's field (default is None).
        region : felupe.RegionBoundary or None, optional
            The boundary region on which to calculate the heat flux. If None, the heat
            flux will be calculated on the provided field's region (default is None).
        integrate : bool, optional
            If True, evaluate the integrated heat transfer rate. Note that if ``mean``
            is also True, the mean heat flux over the boundary is returned. If ``mean``
            is False, return the integrated heat transfer rate is returned. Default is
            True.
        mean : bool, optional
            If True, return the mean heat flux over the boundary. If ``integrate`` is
            also True, the mean heat flux is calculated as the integrated heat transfer
            rate divided by the total area of the boundary. If ``integrate`` is False,
            the mean heat flux is calculated as the heat transfer rate at each
            quadrature point divided by the area at each quadrature point. Default is
            True.
        **kwargs
            Additional keyword arguments to be passed to the gradient function.

        Returns
        -------
        heat_flux : numpy.ndarray
            The heat flux or heat transfer rate on the boundary.

        """
        if (field is None and region is None) or (
            field is not None and region is not None
        ):
            raise ValueError("Either provide a field or a region.")

        if field is None:
            Field = self.field[0].__class__
            field = Field(region, values=self.field[0].values).as_container()

        flux = -self.umat.gradient([*field.extract(), None], **kwargs)[0][0]
        area = field.region.dA  # dA differential area at the boundary
        area_norm = norm(area, axis=0)  # |dA| norm of differential area

        transfer = dot(flux, area, mode=(1, 1))  # -q·dA normal heat transfer

        if integrate:
            res = transfer.sum()  # -∫ q·dA heat transfer rate

            if mean:
                res = transfer.sum() / area_norm.sum()  # -1/A ∫ q·dA  mean heat flux

        else:
            # normal heat flux per quadrature point per cell
            res = transfer / area_norm  # -q·dA / |dA|

            if mean:  # mean heat flux per cell
                res = transfer.sum(axis=0) / area_norm.sum(axis=0)

        return res
