# -*- coding: utf-8 -*-
"""
Tester version for _solid_body_surface_convection.py example case.
"""

import numpy as np
import felupe as fem

hc_fun = np.vectorize(fem.FreeConvection(5, 5).hc_fun)


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
    convection_coefficient=hc_fun,
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
# or
# https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
import matplotlib.pyplot as plt

# host = host_subplot(111)
# par = host.twinx()
# par2 = host.twinx()


fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
twin2 = ax.twinx()

# twin2.spines.right.set_position(("axes", 1.2))

ax.set_xlabel("Time (s)")
ax.set_ylabel("Convection coefficient (W/(m2 K))")
twin1.set_ylabel("Temperature (°C)")
twin2.set_ylabel("Heat flux (W/m2)")

p1 = ax.plot(tstep_data["tstep.s"], tstep_data["hc_top.W.m-2.K-1"],
                label="hc")
p2 = ax.plot(tstep_data["tstep.s"], tstep_data["hc_fun_top.W.m-2.K-1"],
                label="hc_fun(ts_top - t_amb)")
p3 = twin1.plot(tstep_data["tstep.s"], tstep_data["tamb.degC"],
                 label="t_amb")
p4 = twin1.plot(tstep_data["tstep.s"], tstep_data["ts_top.degC"],
                 label="ts_top")
# p3, = host.plot(tstep_data["tstep.s"], tstep_data["hc_calc_top.W.m-2.K-1"],
#                 label="hc from q")
# p3.set_color(p2.get_color())
p5 = twin2.plot(tstep_data["tstep.s"], tstep_data["qc_top.W.m-2"],
                 label="qc_top")

ax.legend(handles=p1+p2+p3+p4+p5, loc='best')

ax.legend(labelcolor="linecolor")

twin2.spines['right'].set_position(('outward', 45))

# host.yaxis.label.set_color(p1.get_color())
# par.yaxis.label.set_color(p2.get_color())

plt.savefig("_test.png")
