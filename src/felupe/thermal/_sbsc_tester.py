# -*- coding: utf-8 -*-
"""
Tester version for _solid_body_surface_convection.py example case.
"""
import math

from pyfluids import HumidAir, InputHumidAir
from scipy.constants import g

import felupe as fem


T0 = 273.15
P0 = 101325

def pyfluids_units():
    check = HumidAir().factory()
    if str(check.units_system) == 'SIWithCelsiusAndPercents':
        dt_ = 0  # use °C
        rh_ = 1  # use %
    else:
        dt_ = 273.15  # use K
        rh_ = 100  # use absolute value
    return dt_, rh_


def rayleigh(ts_c, ti_c, length_, rh=10):
    """
    Calculate dimensionless Rayleigh number Ra.
    """
    dtk, rhf = pyfluids_units()
    tm_c = (ts_c + ti_c)/2

    # Humid air properties at p0 and Tm (indoors).
    air = HumidAir().with_state(
              InputHumidAir.pressure(P0),
              InputHumidAir.temperature(tm_c + dtk),
              InputHumidAir.relative_humidity(rh/rhf),
          )
    rho = air.density
    cp = air.specific_heat
    uv = air.kinematic_viscosity  # m^2/s
    k = air.conductivity  # W/(m K)
    alpha = k/(rho*cp)  # m^2/s thermal diffusivity
    beta = 1/(tm_c + T0)

    # Eqn. 9.25, page 571.
    ra = g*beta*abs(ts_c - ti_c)*length_*length_*length_/alpha/uv

    return ra

def nusselt_horizontal(ra, pr, hflux='z+'):
    """
    Calculate dimensionless Nusselt number Nu for horizontal plates for various
    cases of heat flux direction.
    """
    if hflux == 'z+':  # warm plate, top face or cold plate, bottom face
        if ra < 1E04:
            nu = 0.54*math.pow(1E04,0.25)
        elif (1E04 <= ra <= 1E07) and (pr >= 0.7):
            nu = 0.54*math.pow(ra,0.25)
        elif 1E07 < ra <= 1E11:
            nu = 0.15*math.pow(ra,0.33333)
        else:
            nu = 0.15*math.pow(1E11,0.33333)
    else:  # warm plate, bottom face or cold plate, top face
        if ra < 1E04:
            nu = 0.52*math.pow(1E04,0.2)
        elif 1E04 <= ra <= 1E09 and pr >= 0.7:
            nu = 0.52*math.pow(ra,0.2)
        else:
            nu = 0.52*math.pow(1E09,0.2)
    return nu


def hc_fun(ts, tamb):
    """
    Calculate convection coefficient for horizontal plate.
    """
    dtk, rhf = pyfluids_units()
    tm_c = (ts + tamb)/2
    rh = 50

    air = HumidAir().with_state(
              InputHumidAir.pressure(P0),
              InputHumidAir.temperature(tm_c + dtk),
              InputHumidAir.relative_humidity(rh/rhf),
          )

    l = 1.25  # slab 1 x 1 m^2
    ra = rayleigh(ts, tamb, l)
    # alpha = 2.25E-05 # m^2/s, air at 300 K
    # lam_air = 0.0263  # W/(m K), air at 300 K
    # pr = 0.707  # air at 300 K
    # t_m = 0.5*(ts + tamb) + 273.15  # K
    # ra = (9.81 / t_m * abs(ts - tamb) * math.pow(l, 3.0))/alpha/1.59E-7
    if ts > tamb:
        nu = nusselt_horizontal(ra, air.prandtl, hflux='z+')
    else:
        nu = nusselt_horizontal(ra, air.prandtl, hflux='z-')

    return(nu*air.conductivity/l)

mesh = fem.Rectangle(b=(1.0, 0.25), n=(11, 11))  # rectangle w/ 10x10 cells
region = fem.RegionQuad(mesh)
temperature = fem.Field(region, dim=1, values=30.0)
field = fem.FieldContainer([temperature])

region_convection = fem.RegionQuadBoundary(mesh, mask=mesh.y == 0.25)
temperature_convection = fem.Field(region_convection, dim=1)
field_convection = fem.FieldContainer([temperature_convection])

boundaries = fem.BoundaryDict(
    bottom=fem.Boundary(temperature, fy=0, value=30.0),
)

solid = fem.thermal.SolidBodyThermal(
    field=field,
    mass_density=1400.0,  # kg / m^3
    specific_heat_capacity=1000.0,  # J / (kg K)
    time_step=720.0,  # s
    thermal_conductivity=1.0,  # W / (m K)
)

convection_function = fem.thermal.SolidBodySurfaceConvection(
    field=field_convection,
    convection_coefficient=hc_fun(30, 20),
    temperature=20.0,  # °C
)

def callback(stepnumber, substepnumber, substep, flux_data):
    """Save mean surface heat flux at top (convective) boundary.
    """
    tamb = list(ramp.values())[1][substepnumber]
    ts = list(convection_function.field.extract(grad=False)[0])[0][0:1][0].mean()
    heat_flux = solid.heat_flux_boundary
    qc = heat_flux(region=region_convection)
    flux_data["top.W.m-2"].append(qc)
    flux_data["tamb.degC"].append(tamb)
    flux_data["ts_top.degC"].append(ts)
    flux_data["hc_top.W.m-2.K-1"].append(convection_function.results.convection_coefficient)
    # flux_data["hc_top.W.m-2.K-1"].append(qc/(abs(ts-tamb)))
    # flux_data["top_t"].append(heat_flux(region=internal_region))
    # solid.results.statevars.data.tolist()  # current temperature (mesh-points)
    # region_convection.mask.data.tolist()  # surface points True(!)

n_steps = 20
time = fem.thermal.TimeStep([solid])
table = fem.math.linsteps([0, 1], num=n_steps)
air_temperature = fem.math.linsteps([15, 25], num=n_steps)

ramp = {
    time: 18000 * table,  # five hours
    convection_function["temperature"]: air_temperature,
}

step = fem.Step(
    items=[time, solid, convection_function], ramp=ramp, boundaries=boundaries
)

flux_data = {"tamb.degC": [], "ts_top.degC": [], "hc_top.W.m-2.K-1": [], "top.W.m-2": []}

job = fem.Job(steps=[step], callback=callback, flux_data=flux_data).evaluate(
    verbose=False
)

# mesh.view(
#     point_data={"Temperature 2 in °C": temperature.values}
# ).plot("Temperature 2 in °C").show()
