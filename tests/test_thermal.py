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
import felupe as fem


def test_thermal():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    temperature = fem.Field(region, dim=1)
    field = fem.FieldContainer([temperature])

    region_convection = fem.RegionQuadBoundary(mesh, mask=mesh.y == 1.0)
    temperature_convection = fem.Field(region_convection, dim=1)
    field_convection = fem.FieldContainer([temperature_convection])

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

    convection = fem.thermal.SolidBodyThermalConvection(
        field=field_convection,
        coefficient=1.0,  # W/(m^2*K)
        temperature=10.0,  # K
    )

    heat_flux = fem.thermal.SolidBodyThermalHeatFlux(
        field=field_flux,
        heat_flux=1.0,  # W/m^2
    )

    solid.assemble.vector(field)
    solid.assemble.matrix(field)
    convection.assemble.vector(field)
    convection.assemble.matrix(field)
    heat_flux.assemble.vector(field)
    heat_flux.assemble.matrix(field)

    time = fem.thermal.TimeStep([solid])
    table = fem.math.linsteps([0, 1], num=2)
    ramp = {
        boundaries["right"]: 10 * table,
        time: 0.1 * table,
        convection: 100 * table,
        heat_flux: 10 * table,
    }
    step = fem.Step(
        items=[time, solid, convection, heat_flux], ramp=ramp, boundaries=boundaries
    )
    job = fem.Job(steps=[step]).evaluate(
        filename="result.xdmf",  # result file for Paraview
        point_data={"Temperature": lambda field, substep: temperature.values},
        point_data_default=False,
        cell_data_default=False,
    )


if __name__ == "__main__":
    test_thermal()
