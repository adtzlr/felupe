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
from ..math import dot
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

        \boldsymbol{K} \boldsymbol{T} &= -\boldsymbol{r}

        \boldsymbol{r} &= \boldsymbol{r}_{\text{conductivity}}
            + \boldsymbol{C} \frac{\boldsymbol{T} - \boldsymbol{T}_n}{\Delta t}

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
        >>> mesh.view(
        ...     point_data={"Temperature in K": temperature.values}
        ... ).plot("Temperature in K").show()
        >>>
        >>> mesh.view(
        ...     cell_data={"Heat Flux": solid.results.heat_flux[0][0].mean(axis=-2).T}
        ... ).plot("Heat Flux", component=0).show()

    See Also
    --------
    felupe.thermal.TimeStep : A time step item.
    felupe.thermal.SolidBodyThermalConvection : A thermal convection boundary condition.
    felupe.thermal.SolidBodyThermalHeatFlux : A thermal heat flux boundary condition.

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
        self.heat_flux = self.evaluate.stress

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

        temperature_old = self.results.statevars  # old temperature
        temperature_new = self.results._statevars  # new temperature

        if self.time_step is not None:
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

        if self.time_step is not None:
            self.results.stiffness += self.capacity / self.time_step

        return self.results.stiffness

    def heat_flux_boundary(
        self,
        field=None,
        region=None,
        normal=True,
        total=False,
        mean=False,
        **kwargs,
    ):
        """Calculate the heat flux on a boundary region.

        Parameters
        ----------
        field : felupe.FieldContainer or None, optional
            The field container with the temperature field as first field on which to
            calculate the heat flux. If None, a field will be created on the provided
            region and linked to the body's field (default is None).
        region : felupe.RegionBoundary or None, optional
            The boundary region on which to calculate the heat flux. If None, the heat
            flux will be calculated on provided field's region (default is None).
        normal : bool, optional
            If True, return the normal component of the heat flux (default is True).
        total : bool, optional
            If True, return the total heat transfer rate (default is False).
        mean : bool, optional
            If True, return the mean heat flux over the boundary (default is False).
        **kwargs
            Additional keyword arguments to be passed to the gradient function.

        Returns
        -------
        flux_normal : numpy.ndarray
            The normal component of the heat flux on the boundary, or the total heat
            transfer rate if `total` is True, or the mean heat flux if `mean` is True.

        Examples
        --------
        Evaluate the normal heat flux on the right boundary of a rectangular region for
        the final time step of a transient thermal analysis.

        ..  pyvista-plot::

            >>> import felupe as fem
            >>>
            >>> mesh = fem.Rectangle(n=6)
            >>> region = fem.RegionQuad(mesh)
            >>> temperature = fem.Field(region, dim=1)
            >>> field = temperature.as_container()
            >>>
            >>> solid = fem.thermal.SolidBodyThermal(
            ...     field,
            ...     mass_density=1.0,
            ...     specific_heat_capacity=1.0,
            ...     thermal_conductivity=1.0,
            ... )
            >>> boundaries = fem.BoundaryDict(
            ...     left=fem.Boundary(temperature, fx=0, value=0.0),
            ...     right=fem.Boundary(temperature, fx=1, value=100.0),
            ... )
            >>>
            >>> time = fem.thermal.TimeStep(items=[solid])
            >>> table = fem.math.linsteps([0, 0.01], num=[10])
            >>> step = fem.Step(
            ...     items=[time, solid],
            ...     ramp={time: table},
            ...     boundaries=boundaries,
            ... )
            >>>
            >>> job = fem.Job(steps=[step]).evaluate()
            >>>
            >>> my_region = fem.RegionQuadBoundary(mesh, mask=mesh.x == 1)
            >>> flux = solid.heat_flux_boundary(
            ...     region=my_region, normal=True, total=True, mean=True
            ... )
            >>> flux.round(1)
            np.float64(402.4)

        """
        if (field is None and region is None) or (
            field is not None and region is not None
        ):
            raise ValueError("Either provide a field or a region.")

        if field is None:
            Field = self.field[0].__class__
            field = Field(region, dim=self.field[0].dim).as_container()
            field.link(self.field)

        flux = self.umat.gradient([*field.extract(), None], **kwargs)[0][0]
        area = field.region.dV  # differential areas for dV = |dA| at the boundary
        normals = field.region.normals  # outward normal vectors at the boundary

        if normal:  # normal flux over boundary
            flux = dot(flux, normals, mode=(1, 1))

        if total:
            flux = (flux * area).sum()  # total heat transfer rate

            if mean:
                flux /= area.sum()  # mean total heat flux

        else:
            if mean:  # mean heat flux per cell
                flux = (flux * area).sum(axis=0) / area.sum(axis=0)

        return flux
