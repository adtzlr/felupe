r"""
Notch Stress
------------

.. topic:: Three-dimensional linear-elastic analysis.

   * read a mesh file
   
   * define a linear-elastic solid body
   
   * project the linear-elastic stress tensor to the mesh-points
   
   * plot the longitudinal stress component

.. admonition:: This example requires external packages.
   :class: hint
   
   .. code-block::
      
      pip install pypardiso

A linear-elastic notched plate is subjected to uniaxial tension. The cell-based mean of
the stress tensor is projected to the mesh-points and the longitudinal normal component
:math:`\sigma_{xx}` is plotted. FElupe has no quadratic wedge element formulation
implemented and hence, the quadratic wedges in the mesh are converted to quadratic
hexahedrons.
"""
# sphinx_gallery_thumbnail_number = -1
import numpy as np
import pyvista as pv
import pypardiso
import felupe as fem

m = pv.examples.download_notch_displacement()

hex20 = [0, 2, 1, 1, 3, 5, 4, 4, 8, 7, 1, 6, 11, 10, 4, 9, 12, 14, 13, 13]
mesh = fem.Mesh(
    m.points * 250,
    np.vstack([m.cells_dict[25], m.cells_dict[26][:, hex20]]),
    "hexahedron20",
)
region = fem.RegionQuadraticHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, sym=False, move=0.03375)
solid = fem.SolidBody(umat=fem.LinearElastic(E=2.1e5, nu=0.30), field=field)
step = fem.Step(items=[solid], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate(parallel=True, solver=pypardiso.spsolve)

solid.view(
    point_data={"Stress": fem.project(solid.results.gradient, region, mean=True)}
).plot("Stress", component=0, show_edges=False, show_undeformed=False, view="xy").show()
