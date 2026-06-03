r"""
Thermal Analysis
----------------

.. topic:: Analysis of a thermally activated slab setup.

   * use :class:`~felupe.thermal.SolidBodyThermal`,
     :class:`~felupe.thermal.SolidBodyHeatFlux`,
     :class:`~felupe.thermal.SolidBodySurfaceRadiation` and
     :class:`~felupe.thermal.SolidBodySurfaceConvection`

   * evaluate the surface heat flux at internal and external boundaries
     with a job :class:`~felupe.Plugin`

   * view the temperature field


This example describes a thermally activated concrete slab using a simplified
model and geometry. The model is two-dimensional. The system is set up with two
:class:`solids <felupe.thermal.SolidBodyThermal>`. The temperature boundary
conditions include the floor temperature (constant), the ceiling temperature
(constant) and the room air temperature with a :math:`\pm 2` K sinusoidal
variation around its average value with a period of 24 h.

The heat injection via the pipe layer is constant at 295 W/m2 and directly
injected at the internal concrete surfaces (no pipe material is modelled).

Surface heat transfer is modelled separately for convection and radiation.
"""
import matplotlib.pyplot as plt
import numpy as np

import felupe as fem

# %%
# Define material properties as lists for (reinforced) concrete and insulation.
# This includes mass density, specific heat capacity and thermal conductivity.
density = [2100, 20]  # kg/m^3
specific_heat = [1000, 1450]  # J/(kg K)
thermal_conductivity = [2.1, 0.035]  # W/(m K)

# %%
# Set up one mesh per material. If a material consists of multiple areas, these
# are collected in a :class:`mesh container <felupe.MeshContainer>` and are
# merged into one mesh per material. These meshes per material are then added
# to a mesh container for the construction.
concrete_1a = fem.Rectangle(a=(0.0, 0.0), b=(0.18, 0.22), n=(19, 23))  # left / right
concrete_1b = fem.Rectangle(a=(0.0, 0.0), b=(0.02, 0.10), n=(3, 11))  # pipe bottom / top
concrete_1 = fem.MeshContainer(
    [
        concrete_1a.translate(0.02, axis=0),  # left
        concrete_1b.translate(0.20, axis=0),  # pipe 1, bottom
        concrete_1b.translate(0.20, axis=0).translate(0.12, axis=1),  # pipe1, top
    ],
    merge=True,
    decimals=6,
).stack()

concrete = fem.MeshContainer(
    [
        concrete_1,  # left
        concrete_1.translate(0.2, axis=0),  #
        concrete_1.translate(0.4, axis=0),  #
        concrete_1.translate(0.6, axis=0),  #
        concrete_1a.translate(0.82, axis=0),  # right
    ],
    merge=True,
    decimals=6,
).stack()

insulation_1 = fem.Rectangle(a=(0.0, 0.0), b=(0.02, 0.22), n=(3, 23))  # left / right
insulation = fem.MeshContainer(
    [
        insulation_1,
        insulation_1.translate(1.0, axis=0),
    ],
    merge=True,
    decimals=6,
).stack()

container = fem.MeshContainer([concrete, insulation], merge=True, decimals=6)

# container.plot(
#     colors=["lightgrey", "sepia"],
#     labels=["Concrete", "Insulation"],
#     show_edges=False,
# ).show()

# %%
# A top-level temperature field is defined on the whole construction with an initial
# temperature value of 10 °C, and separate fields are defined for each material. The
# surface heat transfer coefficients and ambient temperatures are defined for the
# internal and external boundaries. Thermal solid bodies are created for each material.
regions = [fem.RegionQuad(m) for m in container]
fields = [fem.Field(r, dim=1).as_container() for r in regions]

# top level temperature field
mesh = container.stack()
region = fem.RegionQuad(mesh)
temperature = fem.Field(region, dim=1, values=20.0)  # initial temperature 10 °C
field = fem.FieldContainer([temperature])

materials = []
for mfield, rho, cp, k in zip(fields, density, specific_heat, thermal_conductivity):
    materials.append(
        fem.thermal.SolidBodyThermal(
            field=mfield,
            mass_density=rho,
            specific_heat_capacity=cp,
            thermal_conductivity=k,
        )
    )

# %%
# The surface heat transfer coefficients and ambient temperatures are defined
# for the side, top and bottom surfaces.
side_region1 = fem.RegionQuadBoundary(mesh, mask=mesh.x == mesh.x.min())
side_temperature1 = fem.Field(side_region1, dim=1)

side_region2 = fem.RegionQuadBoundary(mesh, mask=mesh.x == mesh.x.max())
side_temperature2 = fem.Field(side_region2, dim=1)

side_field = fem.FieldContainer([side_temperature1, side_temperature2])

bottom_region = fem.RegionQuadBoundary(mesh, mask=mesh.y == mesh.y.min())
bottom_temperature = fem.Field(bottom_region, dim=1)
bottom_field = fem.FieldContainer([bottom_temperature])

top_region = fem.RegionQuadBoundary(mesh, mask=mesh.y == mesh.y.max())
top_temperature = fem.Field(top_region, dim=1)
top_field = fem.FieldContainer([top_temperature])

side_heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
    field=side_field,
    coefficient=7.69,  # W/(m^2 K)
    temperature=20.0,  # °C
)
top_heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
    field=top_field,
    coefficient=10.0,  # W/(m^2 K)
    temperature=20.0,  # °C
)
bottom_heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
    field=bottom_field,
    coefficient=5.0,  # W/(m^2 K)
    temperature=20.0,  # °C
)

# %%
# Heat flux on pipe walls is defined.
center_points = np.asarray([[0.21, 0.11], [0.41, 0.11], [0.61, 0.11], [0.81, 0.11]])

pipe_region = []
pipe_field = []
pipe_flux = []
for idx, p in enumerate(center_points):
    mask = np.isclose(mesh.points[:, None, :], p[:], rtol=0.05, atol=0.0101).all(axis=2).any(axis=1)
    pipe_region.append(fem.RegionQuadBoundary(mesh, mask=mask))
    pipe_field.append(fem.FieldContainer([fem.Field(pipe_region[idx], dim=1)]))
    pipe_flux.append(fem.thermal.SolidBodyHeatFlux(
        field=pipe_field[idx],
        heat_flux=-231.25,  # W / m^2, 74/(4*4*0.02)
    ))

# %%
# A callback-function records the mean surface heat flux at the internal and external
# boundaries after each completed time step. The mean surface heat flux is calculated
# by the :meth:`~felupe.thermal.SolidBodyThermal.heat_flux_boundary` method of the
# thermal solid body, which returns the integrated surface heat flux for a given
# boundary region and time step. The mean surface heat flux is stored in the
# ``flux_data`` dictionary, which is passed to the callback function as an argument.
def callback(stepnumber, substepnumber, substep, flux_data):
    """Save mean surface heat flux at internal and external boundaries."""

    heat_flux = materials[0].heat_flux_boundary
    flux_data["top"].append(heat_flux(region=top_region))
    flux_data["bottom"].append(heat_flux(region=bottom_region))
    pflux = 0
    for p_ in pipe_region:
        pflux += heat_flux(region=p_)
    flux_data["pipes"].append(pflux)

time_steps = fem.math.linsteps([0, 24 * 3600], num=int(24 * 3600 / 720))[1:]

t_air = 20 + 2 * np.sin(2 * np.pi * time_steps / 86400)
# t_int = 20 + 2 * np.sin(2 * np.pi * time_steps / 86400)


# %%
# The time step item is created with the thermal solid bodies. It must be located as the
# first item in the step to properly update the time step in the materials. The internal
# and external heat transfer item values are defined in the ramp, which specifies how
# their values change over time. Finally, a job is created with the step and the
# callback function, and evaluated with the top-level temperature field. A result file
# is created for visualization in Paraview, and the temperature field is saved as point-
# data in the result file.
model_list = [*materials, top_heat_transfer, bottom_heat_transfer]
# model_list = [*materials, side_heat_transfer, top_heat_transfer, bottom_heat_transfer]

time = fem.thermal.TimeStep(model_list)
ramp = {
    time: time_steps,
    side_heat_transfer: t_air,
    top_heat_transfer: t_air,
    bottom_heat_transfer: t_air,
}
step = fem.Step(
    items=[time] + model_list + pipe_flux,
    ramp=ramp,
)

flux_data = {"top": [], "bottom": [], "pipes": []}

job = fem.Job(steps=[step], callback=callback, flux_data=flux_data).evaluate(
    x0=field,
    filename="result.xdmf",  # create a result file for Paraview
    point_data={"Temperature": lambda field, substep: temperature.values},
    point_data_default=False,
    cell_data_default=False,
)

# %%
# Internal and external surface heat flux values are plotted over time.
#
# .. note::
#
#    The heat flux is **positive** when **heat leaves the construction** (here, on the
#    external surface), and **negative** when **heat enters the construction** (here, on
#    the internal surface).
#
fig, ax = plt.subplots()
ax.plot(time_steps / 3600, flux_data["top"], color="C3", label="top")
ax.plot(time_steps / 3600, flux_data["bottom"], color="C0", label="bottom")

tmin, tmax = ax.get_xlim()
ax.plot([tmin, tmax], np.zeros(2), "black", lw=0.5)

text_kwargs = dict(transform=ax.transAxes, ha="center", va="center")
ax.text(0.5, 0.97, "heat leaves construction", **text_kwargs)
ax.text(0.5, 0.03, "heat enters construction", **text_kwargs)

ax.legend()
ax.set(xlim=(tmin, tmax), xlabel="time in h", ylabel=r"surface heat flux in W/m$^2$")


# %%
# A view on the temperature field at the end of the simulation period visualizes the
# temperature distribution.
field.plot("Field", scalar_bar_vertical=True).show()
