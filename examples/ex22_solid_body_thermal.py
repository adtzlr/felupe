r"""
Thermal Analysis
------------------------

.. topic:: Thermal analysis of simple construction.

   * use SolidBodyThermal

   * view the temperature field


This example describes a simple two-dimensional light-weight construction
system set up with nine SolidBodyThermal solids.
"""

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

import felupe as fem

# %%
# Define material properties as lists [plasterboard, insulation, wood].
density=[700, 20, 500]  # kg/m^3
specific_heat=[1125, 1450, 1000]  # J/(kg K)
thermal_conductivity=[0.4, 0.032, 0.13]  # W/(m K)

material_index = [0, 0, 0, 1, 2, 1, 0, 0, 0]

# %%
# Set up one mesh per material area, split horizontally and vertically leading
# to nine mesh areas total.
mesh_list = [fem.mesh.Grid(np.linspace(0, 0.018, 6),  # plasterboard
                           np.linspace(0, 0.47, 18)),
             fem.mesh.Grid(np.linspace(0, 0.018, 6),
                           np.linspace(0.47, 0.53, 8)),
             fem.mesh.Grid(np.linspace(0, 0.018, 6),
                           np.linspace(0.53, 1.0, 18)),
             fem.mesh.Grid(np.linspace(0.018, 0.268, 12),  # insulation
                           np.linspace(0, 0.47, 18)),
             fem.mesh.Grid(np.linspace(0.018, 0.268, 12),  # wood
                           np.linspace(0.47, 0.53, 8)),
             fem.mesh.Grid(np.linspace(0.018, 0.268, 12),  # insulation
                           np.linspace(0.53, 1.0, 18)),
             fem.mesh.Grid(np.linspace(0.268, 0.286, 6),  # plasterboard
                           np.linspace(0, 0.47, 18)),
             fem.mesh.Grid(np.linspace(0.268, 0.286, 6),
                           np.linspace(0.47, 0.53, 8)),
             fem.mesh.Grid(np.linspace(0.268, 0.286, 6),
                           np.linspace(0.53, 1.0, 18)),
]

# %%
# Beside points and cells we have to define temperature boundary conditions,
# and the materials for the solid bodies.
mesh_container = fp.MeshContainer(mesh_list, merge=True)
regions = [fp.RegionQuad(m) for m in mesh_container]
field_list = [fp.Field(r, dim=1).as_container() for r in regions]

# top level
mesh = mesh_container.stack()
region = fp.RegionQuad(mesh)
field = fp.Field(region, dim=1).as_container()

temperature = field[0]  # define top-level field values as temperature

external_loc = fp.RegionQuadBoundary(mesh, mask=mesh.x == x.min())
external_temp = fp.Field(external_loc, dim=1)
external_fld = fp.FieldContainer([external_temp])

internal_loc = fp.RegionQuadBoundary(mesh, mask=mesh.x == x.max())
internal_temp = fp.Field(internal_loc, dim=1)
internal_fld = fp.FieldContainer([internal_temp])

boundaries = { # Heat transfer coefficients.
    "external" : SolidBodySurfaceHeatTransfer(
        field=external_fld,
        coefficient=25.0,  # W/(m^2 K)
        temperature=20.0,  # °C
    ),
    "internal" : SolidBodySurfaceHeatTransfer(
        field=internal_fld,
        coefficient=7.69,  # W/(m^2 K)
        temperature=20.0,  # °C
    )
}

materials = []
for imat, fld in enumerate(field_list):
    cur_mat = material_index[imat]
    materials.append(
        SolidBodyThermal(
            fld,
            # field_list[idx],
            density[cur_mat],
            specific_heat[cur_mat],
            thermal_conductivity[cur_mat])
        )

# %%
# Define helper and callback functions for surface flux value retrieval.
def boundary_flux(bmesh, material, coord_index=0, coord_value=0.):
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
            boundary_flux(mesh_list[0], materials[0],
                          coord_value=field.region.mesh.x.min())
    )

    flux_data[1].append(
            boundary_flux(mesh_list[-1], materials[-1],
                          coord_value=field.region.mesh.x.max())
    )

time_steps = fem.math.linsteps([0, 24*3600], num=int(24*3600/720))

t_int = np.around(20 + 1 * np.sin(2*np.pi * time_steps / 86400), 5)
t_ext = np.around(0 + 1 * np.sin(2*np.pi * time_steps / 86400), 5)

# %%
# Set up boundary values, job description and solve.
time = TimeStep(materials)

ramp = {boundaries["internal"]: t_int,
        boundaries["external"]: t_ext,
        time: time_steps}

step = fp.Step(items=[time] + materials,
                ramp=ramp,
                boundaries=boundaries)

flux_data = {0: [], 1: []}

job = fp.Job(
    steps=[step],
    callback=callback,
    flux_data=flux_data).evaluate(
        x0=field,
        filename="result.xdmf",  # result file for Paraview
        point_data={"Temperature": lambda field, substep: temperature.values},
        point_data_default=False,
        cell_data_default=False,
    )

# %%
# Internal and external surface heat flux vs. time.

# %%
# Temperature field at end of simulation period.

