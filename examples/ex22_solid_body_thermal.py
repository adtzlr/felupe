r"""
Thermal Analysis
------------------------

.. topic:: Thermal analysis of simple construction.

   * use :class:`~felupe.thermal.SolidBodyThermal` and
     :class:`~felupe.thermal.SolidBodySurfaceHeatTransfer`

   * view the temperature field


This example describes a simple two-dimensional light-weight construction system set up
with nine :class:`solids <felupe.thermal.SolidBodyThermal>`. The temperature boundary
conditions have a :math:`\pm 1` K sinusoidal variation around their average value with a
period of 24 h.
"""

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

import felupe as fem

# %%
# Define material properties as lists ``[plasterboard, insulation, wood]``.
density = [1000, 20, 500]  # kg/m^3
specific_heat = [1125, 1450, 1600]  # J/(kg K)
thermal_conductivity = [0.4, 0.035, 0.16]  # W/(m K)

material_index = [0, 0, 0, 1, 2, 1, 0, 0, 0]

# %%
# Set up one mesh per material area, split horizontally and vertically leading
# to nine mesh areas total.
mesh_list = [
    fem.mesh.Grid(np.linspace(0, 0.018, 6), np.linspace(0, 0.47, 18)),  # plasterboard
    fem.mesh.Grid(np.linspace(0, 0.018, 6), np.linspace(0.47, 0.53, 8)),
    fem.mesh.Grid(np.linspace(0, 0.018, 6), np.linspace(0.53, 1.0, 18)),
    fem.mesh.Grid(
        np.linspace(0.018, 0.268, 12), np.linspace(0, 0.47, 18)  # insulation
    ),
    fem.mesh.Grid(np.linspace(0.018, 0.268, 12), np.linspace(0.47, 0.53, 8)),  # wood
    fem.mesh.Grid(
        np.linspace(0.018, 0.268, 12), np.linspace(0.53, 1.0, 18)  # insulation
    ),
    fem.mesh.Grid(
        np.linspace(0.268, 0.286, 6), np.linspace(0, 0.47, 18)  # plasterboard
    ),
    fem.mesh.Grid(np.linspace(0.268, 0.286, 6), np.linspace(0.47, 0.53, 8)),
    fem.mesh.Grid(np.linspace(0.268, 0.286, 6), np.linspace(0.53, 1.0, 18)),
]

# %%
# Beside points and cells we have to define temperature boundary conditions, and the
# materials for the solid bodies.
mesh_container = fem.MeshContainer(mesh_list, merge=True)
regions = [fem.RegionQuad(m) for m in mesh_container]
field_list = [fem.Field(r, dim=1).as_container() for r in regions]

# top level
mesh = mesh_container.stack()
region = fem.RegionQuad(mesh)
field = fem.Field(region, dim=1).as_container()

temperature = field[0]  # define top-level field values as temperature

external_loc = fem.RegionQuadBoundary(mesh, mask=mesh.x == mesh.x.min())
external_temp = fem.Field(external_loc, dim=1)
external_fld = fem.FieldContainer([external_temp])

internal_loc = fem.RegionQuadBoundary(mesh, mask=mesh.x == mesh.x.max())
internal_temp = fem.Field(internal_loc, dim=1)
internal_fld = fem.FieldContainer([internal_temp])

external_heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
    field=external_fld,
    coefficient=25.0,  # W/(m^2 K)
    temperature=20.0,  # °C
)
internal_heat_transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
    field=internal_fld,
    coefficient=7.69,  # W/(m^2 K)
    temperature=20.0,  # °C
)

materials = []
for imat, fld in enumerate(field_list):
    cur_mat = material_index[imat]
    materials.append(
        fem.thermal.SolidBodyThermal(
            field=fld,
            mass_density=density[cur_mat],
            specific_heat_capacity=specific_heat[cur_mat],
            thermal_conductivity=thermal_conductivity[cur_mat],
        )
    )


# %%
# Define helper and callback functions for surface flux value retrieval.
def boundary_flux(bmesh, material, coord_index=0, coord_value=0.0):
    """
    Extract boundary flux from mesh/material. Code by adtzlr."""
    points = bmesh.points[bmesh.cells]  # shape(n_cells, 4, 2)
    mask = np.isclose(points[..., coord_index], coord_value).T  # shape(4, n_cells)
    flux = material.evaluate.stress()[0][coord_index, mask]  # shape(38)
    return flux


def callback(stepnumber, substepnumber, substep, flux_data):
    """
    Extract stress data (flux), shape(n_cells, 4) per substep."""
    flux_data[0].append(
        boundary_flux(mesh_list[0], materials[0], coord_value=field.region.mesh.x.min())
    )

    flux_data[1].append(
        boundary_flux(
            mesh_list[-1], materials[-1], coord_value=field.region.mesh.x.max()
        )
    )


time_steps = fem.math.linsteps([0, 24 * 3600], num=int(24 * 3600 / 720))

t_int = np.around(20 + 1 * np.sin(2 * np.pi * time_steps / 86400), 5)
t_ext = np.around(0 + 1 * np.sin(2 * np.pi * time_steps / 86400), 5)

# %%
# Set up boundary values, job description and solve.
time = fem.thermal.TimeStep(materials)

ramp = {internal_heat_transfer: t_int, external_heat_transfer: t_ext, time: time_steps}

step = fem.Step(
    items=[time] + materials + [internal_heat_transfer, external_heat_transfer],
    ramp=ramp,
)

flux_data = {0: [], 1: []}

job = fem.Job(steps=[step], callback=callback, flux_data=flux_data).evaluate(
    x0=field,
    filename="result.xdmf",  # create a result file for Paraview
    point_data={"Temperature": lambda field, substep: temperature.values},
    point_data_default=False,
    cell_data_default=False,
)

# %%
# Internal and external surface heat flux vs. time.
q_ext = np.mean(flux_data[0], axis=1)  # W/m2 => len=1
q_int = np.mean(flux_data[1], axis=1)  # W/m2

fig, ax = plt.subplots()
ax.plot(time_steps.T / 3600, q_int.T)
ax.plot(time_steps / 3600, q_ext)
ax.set(xlabel="time (h)", ylabel="surface heat flux (W/(m^2 K))")

# %%
# Temperature field at the end of the simulation period.
view = mesh.view(point_data={"Field": temperature.values})
view.plot("Field").show()
