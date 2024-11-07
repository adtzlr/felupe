r"""
Notch Stress
------------

.. topic:: Three-dimensional linear-elastic analysis.

   * read a mesh file
   
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
plotted. FElupe has no wedge element formulation implemented and hence, the wedges in
the mesh are converted to hexahedrons.
"""

# sphinx_gallery_thumbnail_number = -1
import numpy as np
import pypardiso
import pyvista as pv

import felupe as fem

m = pv.examples.download_notch_displacement()

hex8 = [0, 2, 1, 1, 3, 5, 4, 4]
mesh = fem.Mesh(
    m.points * 250,
    np.vstack([m.cells_dict[25][:, :8], m.cells_dict[26][:, hex8]]),
    "hexahedron",
)
point_ids = np.zeros(mesh.npoints, dtype=int)
point_ids[mesh.points_with_cells] = np.arange(len(mesh.points_with_cells))
mesh.update(points=mesh.points[mesh.points_with_cells], cells=point_ids[mesh.cells])

region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, sym=False, move=0.02)
solid = fem.SolidBody(umat=fem.LinearElastic(E=2.1e5, nu=0.30), field=field)
step = fem.Step(items=[solid], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate(parallel=True, solver=pypardiso.spsolve)

solid.plot(
    "Principal Values of Cauchy Stress",
    show_edges=False,
    view="xy",
    project=fem.topoints,
    show_undeformed=False,
).show()

# %%
# The number of maximum endurable cycles between zero and the applied displacement is
# evaluated with a SN-curve as denoted in Eq. :eq:`sn-curve`. The range of the maximum
# principal value of the Cauchy stress tensor is used to evaluate the fatigue life.
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

S = fem.topoints(fem.math.eigvalsh(solid.evaluate.cauchy_stress())[-1], region)
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
