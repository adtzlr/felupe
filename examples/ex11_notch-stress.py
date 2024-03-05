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

A linear-elastic notched plate is subjected to uniaxial tension. The stress tensor is
projected to the mesh-points and the longitudinal normal component :math:`\sigma_{xx}`
is plotted.

A mesh file is provided for this example (taken from the docs of
`pyvista <https://docs.pyvista.org/>`_): 
    
* `mesh <../_static/ex11_notch-stress_mesh.vtu>`_
"""
# sphinx_gallery_thumbnail_number = -1
import felupe as fem
import pypardiso

mesh = fem.mesh.read("ex11_notch-stress_mesh.vtu")[0]
region = fem.RegionTriQuadraticHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, sym=False, move=0.03375)
solid = fem.SolidBody(umat=fem.LinearElastic(E=2.1e5, nu=0.30), field=field)
step = fem.Step(items=[solid], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate(parallel=True, solver=pypardiso.spsolve)

solid.view(point_data={"Stress": fem.project(solid.results.gradient, region)}).plot(
    "Stress", component=0, show_edges=False, show_undeformed=False, view="xy"
).show()
