r"""
Voxelized Foot Bones
--------------------
A :class:`~felupe.Region` on a voxel-based mesh with uniform hexahedrons should be
created with ``uniform=True`` to enhance performance.

.. admonition:: This example requires external packages.
   :class: hint

   .. code-block::

      pip install pypardiso
"""

import numpy as np
import pypardiso

import felupe as fem

mesh = fem.mesh.read("ex12_foot-bones_mesh-voxels.vtu")[0]

region = fem.RegionHexahedron(mesh, uniform=True)
field = fem.FieldContainer([fem.Field(region, dim=3)])
boundaries = {"fixed": fem.Boundary(field[0], fx=lambda x: x <= -110)}

umat = fem.LinearElastic(E=1000, nu=0.3)
solid = fem.SolidBody(umat, field)
gravity = fem.SolidBodyForce(field, values=[0, 0, -7e-2])
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
