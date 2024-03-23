r"""
Voxelized Foot Bone
-------------------

.. admonition:: This example requires external packages.
   :class: hint
   
   .. code-block::
      
      pip install pypardiso
"""

import numpy as np
import pypardiso
import pyvista as pv
from pyvista import examples

import felupe as fem

surface = examples.download_foot_bones()
voxels = pv.voxelize(surface, density=surface.length / 50)

mesh = fem.Mesh(
    points=voxels.points,
    cells=voxels.cell_connectivity.reshape(-1, 8),
    cell_type="hexahedron",
).rotate(90, axis=0)
mesh.points *= 25
mesh.update(points=np.vstack([mesh.points, [0, 0, mesh.z.min() - 2]]))

region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])
boundaries = {"fixed": fem.Boundary(field[0], fx=lambda x: x <= -110)}

umat = fem.LinearElastic(E=1000, nu=0.3)
solid = fem.SolidBody(umat, field)
gravity = fem.SolidBodyGravity(field, gravity=[0, 0, -7e-2])
bottom = fem.MultiPointContact(
    field, points=np.arange(mesh.npoints), centerpoint=-1, skip=(1, 1, 0)
)

step = fem.Step(items=[solid, gravity, bottom], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate(solver=pypardiso.spsolve, parallel=True)
plotter = solid.plot(
    "Principal Values of Cauchy Stress",
    show_edges=False,
    show_undeformed=False,
    clim=[0, 10],
)
bottom.plot(plotter=plotter, color="white", opacity=1).show()
