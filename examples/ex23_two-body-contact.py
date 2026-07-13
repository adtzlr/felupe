r"""
Two-Body Contact
----------------

.. topic:: Node-to-segment penalty contact between two deformable solid bodies.

   * create a mesh container with two meshes and a common point numbering

   * define two solid bodies and their contact surfaces on boundary regions

   * add a :class:`~felupe.SolidBodyContact` as an item to a :class:`~felupe.Step`

   * plot the deformed solid bodies

This example demonstrates a frictionless contact between a curved body (a circular
section) and a rectangular block. The curved body is pressed onto the block. The contact
is formulated as a node-to-segment penalty contact by a
:class:`~felupe.SolidBodyContact`, which takes two field containers whose fields are
defined on a :class:`~felupe.RegionBoundary`. Coulomb friction is available via the
``friction``-argument.

First, two meshes with a small initial gap are created and merged into one
:class:`~felupe.MeshContainer`. The ``merge=True`` argument ensures a common point
numbering, which is required to assemble one global system of equations. The initial gap
prevents the two contact surfaces from being bonded (merged).
"""

import felupe as fem

bottom = fem.Rectangle(a=(0.0, 0.0), b=(2.0, 1.0), n=(11, 6))
top = fem.Circle(radius=1, centerpoint=(1.0, 2.05), n=4, sections=[180, 270])
container = fem.MeshContainer([bottom, top], merge=True)

# %%
# For each mesh of the container, a quad region and a plane-strain displacement field
# are created. Both bodies use the same Neo-Hookean material formulation.
regions = [fem.RegionQuad(mesh) for mesh in container.meshes]
fields = [
    fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)]) for region in regions
]

umats = [fem.NeoHooke(mu=1.0, bulk=5.0), fem.NeoHooke(mu=1.0, bulk=5.0)]
solids = [fem.SolidBody(umat, field) for umat, field in zip(umats, fields)]

# %%
# The whole outline of each body is used as a contact surface. For each body a boundary
# region (a :class:`~felupe.RegionQuadBoundary` without a mask, i.e. the complete
# boundary) and a boundary field are created. With ``symmetric=True``, the contact
# search is performed in two passes, where the roles of the slave (contactor) and master
# (target) surface are swapped. This is useful here, because a curved surface contacts a
# flat one and neither surface fully covers the other.
boundary_bottom = fem.RegionQuadBoundary(container.meshes[0])
field_bottom = fem.FieldContainer([fem.FieldPlaneStrain(boundary_bottom, dim=2)])

boundary_top = fem.RegionQuadBoundary(container.meshes[1])
field_top = fem.FieldContainer([fem.FieldPlaneStrain(boundary_top, dim=2)])

contact = fem.SolidBodyContact(
    field_bottom,
    field_top,
    items=solids,
    symmetric=True,
    multiplier=5.0,
)

# %%
# A top-level plane-strain field on the stacked mesh includes all unknowns and is
# required for the selection of the prescribed degrees of freedom as well as for
# Newton's method. The block is fixed at its bottom edge and the curved body is pressed
# downwards by a prescribed vertical displacement on its top edge.
region = fem.RegionQuad(container.stack())
field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

boundaries = fem.BoundaryDict(
    fixed=fem.Boundary(field[0], fy=0.0),
    move=fem.Boundary(field[0], fy=2.05),
)

# %%
# The prescribed vertical displacement is ramped up in several substeps. The top-level
# field has to be passed as the ``x0``-argument of the job.
move = fem.math.linsteps([0, -0.4], num=40, axis=1, axes=2)

step = fem.Step(
    items=[*solids, contact],
    ramp={boundaries["move"]: move},
    boundaries=boundaries,
)
job = fem.Job(steps=[step]).evaluate(x0=field, verbose=2, tol=1e-2)

# %%
# The deformed configuration of both solid bodies is plotted.
plotter = solids[0].plot("Principal Values of Cauchy Stress", show_undeformed=False)
solids[1].plot(
    "Principal Values of Cauchy Stress",
    show_undeformed=False,
    plotter=plotter,
).show()
