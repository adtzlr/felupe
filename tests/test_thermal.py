# -*- coding: utf-8 -*-
"""
This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""
import pytest

import felupe as fem


def test_thermal():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    temperature = fem.Field(region, dim=1)
    field = fem.FieldContainer([temperature])

    region_heat_transfer = fem.RegionQuadBoundary(mesh, mask=mesh.y == 1.0)
    temperature_heat_transfer = fem.Field(region_heat_transfer, dim=1)
    field_heat_transfer = fem.FieldContainer([temperature_heat_transfer])

    region_flux = fem.RegionQuadBoundary(mesh, mask=mesh.y == 1.0)
    temperature_flux = fem.Field(region_flux, dim=1)
    field_flux = fem.FieldContainer([temperature_flux])

    boundaries = fem.BoundaryDict(
        left=fem.Boundary(temperature, fx=0),
        right=fem.Boundary(temperature, fx=1),
    )

    solid = fem.thermal.SolidBodyThermal(
        field=field,
        mass_density=1.0,  # kg/m^3
        specific_heat_capacity=1.0,  # J/(kg*K)
        time_step=0.01,  # s
        thermal_conductivity=1.0,  # W/(m*K)
        lumped_capacity=False,
    )

    solid = fem.thermal.SolidBodyThermal(
        field=field,
        mass_density=1.0,  # kg/m^3
        specific_heat_capacity=1.0,  # J/(kg*K)
        time_step=0.01,  # s
        thermal_conductivity=1.0,  # W/(m*K)
    )

    heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
        field=field_heat_transfer,
        coefficient=1.0,  # W/(m^2*K)
        temperature=10.0,  # K
    )

    heat_flux = fem.thermal.SolidBodyThermalHeatFlux(
        field=field_flux,
        heat_flux=1.0,  # W/m^2
    )

    solid.assemble.vector(field)
    solid.assemble.matrix(field)
    heat_transfer.assemble.vector(field)
    heat_transfer.assemble.matrix(field)
    heat_flux.assemble.vector(field)
    heat_flux.assemble.matrix(field)

    time = fem.thermal.TimeStep([solid])
    table = fem.math.linsteps([0, 0, 1], num=2)
    ramp = {
        boundaries["right"]: 10 * table,
        time: 0.1 * table,
        heat_transfer: 100 * table,
        heat_flux: 10 * table,
    }
    step = fem.Step(
        items=[time, solid, heat_transfer, heat_flux], ramp=ramp, boundaries=boundaries
    )
    job = fem.Job(steps=[step]).evaluate(
        filename="result.xdmf",  # result file for Paraview
        point_data={"Temperature": lambda field, substep: temperature.values},
        point_data_default=False,
        cell_data_default=False,
    )


def test_thermal_axi():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    temperature = fem.Field(region, dim=1)
    field = fem.FieldContainer([temperature])

    region_heat_transfer = fem.RegionQuadBoundary(mesh, mask=mesh.y == 1.0)
    temperature_heat_transfer = fem.FieldAxisymmetric(region_heat_transfer, dim=1)
    field_heat_transfer = fem.FieldContainer([temperature_heat_transfer])

    region_flux = fem.RegionQuadBoundary(mesh, mask=mesh.y == 1.0)
    temperature_flux = fem.FieldAxisymmetric(region_flux, dim=1)
    field_flux = fem.FieldContainer([temperature_flux])

    boundaries = fem.BoundaryDict(
        left=fem.Boundary(temperature, fx=0),
        right=fem.Boundary(temperature, fx=1),
    )

    solid = fem.thermal.SolidBodyThermal(
        field=field,
        mass_density=1.0,  # kg/m^3
        specific_heat_capacity=1.0,  # J/(kg*K)
        time_step=0.01,  # s
        thermal_conductivity=1.0,  # W/(m*K)
        lumped_capacity=False,
    )

    solid = fem.thermal.SolidBodyThermal(
        field=field,
        mass_density=1.0,  # kg/m^3
        specific_heat_capacity=1.0,  # J/(kg*K)
        time_step=0.01,  # s
        thermal_conductivity=1.0,  # W/(m*K)
    )

    heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
        field=field_heat_transfer,
        coefficient=1.0,  # W/(m^2*K)
        temperature=10.0,  # K
    )

    heat_flux = fem.thermal.SolidBodyThermalHeatFlux(
        field=field_flux,
        heat_flux=1.0,  # W/m^2
    )

    solid.assemble.vector(field)
    solid.assemble.matrix(field)
    heat_transfer.assemble.vector(field)
    heat_transfer.assemble.matrix(field)
    heat_flux.assemble.vector(field)
    heat_flux.assemble.matrix(field)

    solid.heat_flux(field)

    my_region = fem.RegionQuadBoundary(mesh, mask=mesh.x == 1)
    solid.heat_flux_boundary(region=my_region)
    solid.heat_flux_boundary(region=my_region, mean=True)
    solid.heat_flux_boundary(region=my_region, total=True)
    solid.heat_flux_boundary(region=my_region, total=True, mean=True)

    solid.heat_flux_boundary(region=my_region, normal=False)
    solid.heat_flux_boundary(region=my_region, normal=False, mean=True)
    solid.heat_flux_boundary(region=my_region, normal=False, total=True)
    solid.heat_flux_boundary(region=my_region, normal=False, total=True, mean=True)

    my_field = fem.Field(my_region, dim=1).as_container()
    solid.heat_flux_boundary(field=my_field)
    solid.heat_flux_boundary(field=my_field, mean=True)
    solid.heat_flux_boundary(field=my_field, total=True)
    solid.heat_flux_boundary(field=my_field, total=True, mean=True)

    solid.heat_flux_boundary(field=my_field, normal=False)
    solid.heat_flux_boundary(field=my_field, normal=False, mean=True)
    solid.heat_flux_boundary(field=my_field, normal=False, total=True)
    solid.heat_flux_boundary(field=my_field, normal=False, total=True, mean=True)

    with pytest.raises(ValueError):
        solid.heat_flux_boundary(field=None, region=None)

    with pytest.raises(ValueError):
        solid.heat_flux_boundary(field=my_field, region=my_region)

    time = fem.thermal.TimeStep([solid])
    table = fem.math.linsteps([0, 0, 1], num=2)
    table_2 = fem.math.linsteps([1, 1, 1], num=2)
    ramp = {
        boundaries["right"]: 10 * table,
        time: 0.1 * table,
        heat_transfer: 100 * table,
        heat_transfer["coefficient"]: 100 * table_2,
        heat_flux: 10 * table,
    }
    step = fem.Step(
        items=[time, solid, heat_transfer, heat_flux], ramp=ramp, boundaries=boundaries
    )
    job = fem.Job(steps=[step]).evaluate(
        filename="result.xdmf",  # result file for Paraview
        point_data={"Temperature": lambda field, substep: temperature.values},
        point_data_default=False,
        cell_data_default=False,
    )


def test_timestep():

    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    temperature = fem.Field(region, dim=1)
    field = fem.FieldContainer([temperature])

    solid = fem.thermal.SolidBodyThermal(
        field=field,
        mass_density=1.0,  # kg/m^3
        specific_heat_capacity=1.0,  # J/(kg*K)
        thermal_conductivity=1.0,  # W/(m*K)
    )

    time = fem.thermal.TimeStep(items=[solid])

    time.update(0.1)
    assert solid.time_step == 0.1

    with pytest.raises(ValueError):
        time.update(-0.1)


if __name__ == "__main__":
    test_thermal()
    test_thermal_axi()
    test_timestep()
