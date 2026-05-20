# -*- coding: utf-8 -*-
"""
Tester version for _solid_body_surface_convection.py example case.
"""
import math
import numpy as np

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


@np.vectorize
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

@np.vectorize
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


@np.vectorize
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

    l = 1.25  # assume slab size 5 x 5 m^2 for h_c
    ra = rayleigh(ts, tamb, l)
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
    convection_coefficient=hc_fun, #(30, 20),
    temperature=20.0,  # °C
)

def callback(stepnumber, substepnumber, substep, tstep_data):
    """Save time step data at top (convective) boundary.
    """
    tamb = list(ramp.values())[1][substepnumber]
    ts = list(convection_function.field.extract(grad=False)[0])[0][0:1][0].mean()
    heat_flux = solid.heat_flux_boundary
    qc = heat_flux(region=region_convection)
    tstep_data["tstep.s"].append(list(ramp.values())[0][substepnumber])
    tstep_data["qc_top.W.m-2"].append(qc)
    tstep_data["tamb.degC"].append(tamb)
    tstep_data["ts_top.degC"].append(ts)
    tstep_data["hc_top.W.m-2.K-1"].append(
        convection_function.results.convection_coefficient.mean())
    tstep_data["hc_fun_top.W.m-2.K-1"].append(hc_fun(ts, tamb).mean())
    tstep_data["hc_calc_top.W.m-2.K-1"].append(abs(qc/(ts-tamb)))


n_steps = 20
time = fem.thermal.TimeStep([solid])
table = fem.math.linsteps([0, 1], num=n_steps)
air_temperature = fem.math.linsteps([15, 25], num=n_steps)

ramp = {
    time: 18000 * table,  # five hours
    convection_function: air_temperature,
}

step = fem.Step(
    items=[time, solid, convection_function], ramp=ramp, boundaries=boundaries
)

tstep_data = {"tstep.s": [], "tamb.degC": [], "ts_top.degC": [],
              "hc_top.W.m-2.K-1": [], "qc_top.W.m-2": [],
              "hc_fun_top.W.m-2.K-1": [],
              "hc_calc_top.W.m-2.K-1": []}

job = fem.Job(steps=[step], callback=callback, tstep_data=tstep_data).evaluate(
    verbose=False
)

# mesh.view(
#     point_data={"Temperature 2 in °C": temperature.values}
# ).plot("Temperature 2 in °C").show()

# Plot h_c / surface temp. / air temp. vs. time.
# https://matplotlib.org/stable/gallery/axes_grid1/parasite_simple.html
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

host = host_subplot(111)
par = host.twinx()

host.set_xlabel("Time (s)")
host.set_ylabel("Convection coefficient (W/(m2 K))")
par.set_ylabel("Temperature (°C)")

p1, = host.plot(tstep_data["tstep.s"],
                tstep_data["hc_top.W.m-2.K-1"],
                label="hc")
p2, = par.plot(tstep_data["tstep.s"],
               tstep_data["tamb.degC"],
               label="t_amb")
p3, = par.plot(tstep_data["tstep.s"],
               tstep_data["ts_top.degC"],
               label="ts_top")
p3.set_color(p2.get_color())

host.legend(labelcolor="linecolor")

host.yaxis.label.set_color(p1.get_color())
par.yaxis.label.set_color(p2.get_color())

plt.savefig("_test.png")
