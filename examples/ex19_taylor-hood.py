r"""
Mixed-field hyperelasticity with quadratic triangles
----------------------------------------------------
A 90° section of a plane-strain circle is subjected to frictionless uniaxial compression
by a vertically moved rigid top plate. A mixed-field formulation is used with quadratic 
triangles.
"""

# sphinx_gallery_thumbnail_number = -1
from functools import partial

import numpy as np

import felupe as fem

# create a 90° section of a circle with quadratic triangles
mesh = fem.Circle(n=6, sections=[0]).triangulate().add_midpoints_edges()

# shift the midpoints to the outer radius
mask = np.isclose(mesh.x**2 + mesh.y**2, 1, atol=0.05)
mesh.points[mask] /= np.linalg.norm(mesh.points[mask], axis=1).reshape(-1, 1)

# add a center-point for the rigid top plate
mesh.update(points=np.vstack([mesh.points, [0, 1.1]]))

# create a region and a mixed-field
region = fem.RegionQuadraticTriangle(mesh)
field = fem.FieldsMixed(region, n=3, planestrain=True, disconnect=True)

# define boundary conditions
boundaries = fem.dof.symmetry(field[0])
boundaries["move"] = fem.Boundary(field[0], fy=1.1, skip=(1, 0))

# create a nearly-incompressible hyperelastic solid body and the rigid top plate
umat = fem.NearlyIncompressible(material=fem.NeoHooke(mu=1), bulk=5000)
solid = fem.SolidBody(umat=umat, field=field)
top = fem.MultiPointContact(
    field=field,
    points=np.arange(mesh.npoints)[np.isclose(mesh.x**2 + mesh.y**2, 1)],
    centerpoint=-1,
    skip=(1, 0),
)
mesh.plot(nonlinear_subdivision=4, plotter=top.plot(line_width=5, opacity=1)).show()

move = fem.math.linsteps([0, -0.4], num=4)
ramp = {boundaries["move"]: move}
step = fem.Step(items=[solid, top], ramp=ramp, boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate()

solid.plot(
    "Principal Values of Cauchy Stress",
    nonlinear_subdivision=4,
    plotter=top.plot(line_width=5, opacity=1),
    project=partial(fem.project, mean=True),
).show()
