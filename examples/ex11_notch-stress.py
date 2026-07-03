r"""
Notch Stress
------------

.. topic:: Three-dimensional linear-elastic analysis.

   * create a hexahedron mesh

   * define a linear-elastic solid body

   * project the linear-elastic stress tensor to the mesh-points

   * plot the max. principal stress component

   * evaluate the fatigue life

.. admonition:: This example requires external packages.
   :class: hint

   .. code-block::

      pip install pypardiso

A linear-elastic notched plate is subjected to uniaxial tension. The cell-based mean of
the stress tensor is projected to the mesh-points and its maximum principal value is
plotted.
"""

# sphinx_gallery_thumbnail_number = -1
import pypardiso

import felupe as fem

meshes = []

radius = fem.mesh.Point(a=-2.5).revolve(n=9, phi=90).translate(5, axis=1)
radius = fem.mesh.flip(radius)
middle = fem.mesh.Line(a=-7.5, b=0, n=9).expand(n=0)
meshes.append(middle.fill_between(radius, n=6))

left = fem.mesh.Line(-7.5, 5, n=11).expand(n=0).rotate(90, axis=2, center=[-7.5, 0])
right = (
    fem.mesh.Line(a=-2.5, b=5, n=11)
    .expand(n=0)
    .rotate(90, axis=2, center=[-2.5, 0])
    .translate(5, axis=1)
)
meshes.append(right.fill_between(left, n=6))
meshes.append(fem.Rectangle(a=(-50, 0), b=(-7.5, 12.5), n=(36, 11)))

mesh = fem.MeshContainer(meshes, merge=True).stack()
mesh = fem.MeshContainer([mesh, mesh.mirror(axis=0)], merge=True, decimals=6).stack()
mesh = fem.MeshContainer([mesh, mesh.mirror(axis=1)], merge=True, decimals=6).stack()
mesh = mesh.expand(n=3, z=2.5)

region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

boundaries = fem.dof.uniaxial(
    field, clamped=True, sym=False, move=0.02, return_loadcase=False
)
solid = fem.SolidBody(umat=fem.LinearElastic(E=2.1e5, nu=0.30), field=field)
step = fem.Step(items=[solid], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate(parallel=True, solver=pypardiso.spsolve)

solid.plot(
    "Principal Values of Stress",
    show_edges=False,
    view="xy",
    project=fem.topoints,
    show_undeformed=False,
).show()

# %%
# The number of maximum endurable cycles between zero and the applied displacement is
# evaluated with a SN-curve as denoted in Eq. :eq:`sn-curve`. The range of the maximum
# principal value of the stress tensor is used to evaluate the fatigue life.
# For simplicity, the stress is evaluated for the total solid body. To consider only
# stresses on points which lie on the surface of the solid body, the cells on faces
# :meth:`~felupe.RegionHexahedronBoundary.mesh.cells_faces` must be determined
# first.
#
# .. math::
#    :label: sn-curve
#
#    \frac{N}{N_D} = \left( \frac{S}{S_D} \right)^{-k}
#
S_D = 100  # MPa
N_D = 2e6  # cycles
k = 5  # slope

S = fem.topoints(fem.math.eigvalsh(solid.evaluate.stress())[-1], region)
N = N_D * (abs(S) / S_D) ** -k

view = solid.view(point_data={"Endurable Cycles": N})
plotter = view.plot(
    "Endurable Cycles",
    show_undeformed=False,
    show_edges=False,
    log_scale=True,
    flip_scalars=True,
    clim=[N.min(), 2e6],
    above_color="lightgrey",
)
plotter.camera.zoom(6)
plotter.show()
