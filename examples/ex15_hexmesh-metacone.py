r"""
Script-based Hex-meshing with revolve
-------------------------------------

.. topic:: Create a 3d dynamic mesh for a metacone component out of quads and revolve
   it to hexahedrons.

   * apply :ref:`mesh-tools <felupe-api-mesh>`

   * use :class:`~felupe.MeshContainer` to combine multiple meshes

   * run an axisymmetric simulation and revolve the results to a 3d simulation
"""

# sphinx_gallery_thumbnail_number = -1
import numpy as np
import pypardiso

import felupe as fem

layers = [2, 11, 2, 11, 2]
lines = [
    fem.mesh.Line(a=0, b=13, n=21).expand(n=1).translate(2, axis=1),
    fem.mesh.Line(a=0, b=13, n=21).expand(n=1).translate(2.5, axis=1),
    fem.mesh.Line(a=-0.2, b=10, n=21).expand(n=1).translate(4.5, axis=1),
    fem.mesh.Line(a=-0.2, b=10, n=21).expand(n=1).translate(5, axis=1),
    fem.mesh.Line(a=-0.4, b=7, n=21).expand(n=1).translate(6.5, axis=1),
    fem.mesh.Line(a=-0.4, b=7, n=21).expand(n=1).translate(7, axis=1),
]
faces = fem.MeshContainer(
    [
        first.fill_between(second, n=n)
        for first, second, n in zip(lines[:-1], lines[1:], layers)
    ]
)
point = lambda m: m.points[np.unique(m.cells)].mean(axis=0) - np.array([2, 0])
mask = lambda m: np.unique(m.cells)
kwargs = dict(axis=1, exponent=2.5, normalize=True)
faces.points[:] = (
    faces[1]
    .add_runouts([3], centerpoint=point(faces[1]), mask=mask(faces[1]), **kwargs)
    .points
)
faces.points[:] = (
    faces[3]
    .add_runouts([7], centerpoint=point(faces[3]), mask=mask(faces[3]), **kwargs)
    .points
)
faces.points[:21] = faces.points[21 : 2 * 21]
faces.points[-21:] = faces.points[-2 * 21 : -21]
faces.points[:] = faces[0].rotate(15, axis=2).points
faces[0].y[:21] = 2
faces[-1].y[-21:] = 8.5
mesh = fem.MeshContainer(
    [faces.stack([1, 3]), faces.stack([0, 2, 4])], merge=True, decimals=4
)

# %%
# Sub-fields and a top-level field are created by the merge-method of a field container.
# Two solid bodies are created, one for the rubber and one for the metal. The top-level
# field is passed as the ``x0``-argument to the evaluate-method of the job. The part is
# displaced along the rotation axis.
fields, x0 = fem.FieldContainer(
    [fem.FieldAxisymmetric(fem.RegionQuad(m), dim=2) for m in mesh]
).merge()

rubber = fem.NeoHooke(mu=1)
metal = fem.LinearElasticLargeStrain(E=2.1e5, nu=0.3)

solid1 = fem.SolidBodyNearlyIncompressible(rubber, fields[0], bulk=5000)
solid2 = fem.SolidBody(metal, fields[1])

boundaries = fem.dof.uniaxial(x0, clamped=True, sym=False, return_loadcase=False)
ramp = {boundaries["move"]: fem.math.linsteps([0, 1], num=5) * -1.5}
step = fem.Step(items=[solid1, solid2], ramp=ramp, boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate(x0=x0, solver=pypardiso.spsolve)

solid1.plot(
    "Principal Values of Cauchy Stress",
    plotter=solid2.plot(color="white", show_edges=False),
).show()

# %%
# In order to obtain a 3d-model, the top-level field and the solid bodies are revolved.
# For simplicity, the same load case is applied on the 3d-model. This may be used as a
# starting point for further non-axisymmetric loads applied on the model.
solid1_3d = solid1.revolve(n=6, phi=90)
solid2_3d = solid2.revolve(n=6, phi=90)
x0_3d = x0.revolve(n=6, phi=90)

boundaries = fem.dof.uniaxial(
    x0_3d, clamped=True, sym=(0, 1, 1), move=-1.5, return_loadcase=False
)
step = fem.Step(items=[solid1_3d, solid2_3d], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate(x0=x0_3d, solver=pypardiso.spsolve)

solid1_3d.plot(
    "Principal Values of Cauchy Stress",
    plotter=solid2_3d.plot(color="white", show_edges=False),
).show()
