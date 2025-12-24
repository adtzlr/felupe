r"""
Getting Started
---------------

.. topic:: Your very first steps with FElupe.

   * create a meshed cube with hexahedron cells

   * define a numeric region along with a displacement field

   * load a Neo-Hookean material formulation

   * apply a uniaxial loadcase

   * solve the problem

   * export the displaced mesh

This tutorial covers the essential high-level parts of creating and solving problems
with FElupe. As an introductory example, a quarter model of a solid cube with
hyperelastic material behaviour is subjected to a uniaxial elongation applied at a
clamped end-face. First, let's import FElupe and create a meshed cube out of hexahedron
cells with ``n`` points per axis. A numeric region, pre-defined for hexahedrons, is
created on the mesh. A vector-valued displacement field is initiated on the region.
Next, a field container is created on top of the displacement field.
"""

import felupe as fem

mesh = fem.Cube(n=6)
region = fem.RegionHexahedron(mesh=mesh)
displacement = fem.Field(region=region, dim=3)
field = fem.FieldContainer(fields=[displacement])

# %%
# A uniaxial load case is applied on the displacement field stored inside the field
# container. This involves setting up symmetry planes as well as the absolute value of
# the prescribed displacement at the mesh-points on the right-end face of the cube. The
# right-end face is *clamped*: only displacements in direction x are allowed. The dict
# of boundary conditions for this pre-defined load case are returned as ``boundaries``
# and the partitioned degrees of freedom as well as the external displacements are
# stored within the returned dict ``loadcase``.
boundaries, loadcase = fem.dof.uniaxial(
    field,
    move=0.2,
    right=1,
    clamped=True,
    return_loadcase=True,
)

# %%
# The material behaviour is defined through a built-in Neo-Hookean material formulation.
# The constitutive isotropic hyperelastic material formulation is applied on the
# displacement field by the definition of a solid body.
umat = fem.NeoHooke(mu=1.0, bulk=2.0)
solid = fem.SolidBody(umat=umat, field=field)

# %%
# The problem is solved by an iterative :func:`Newton-Rhapson <felupe.newtonrhapson>`
# procedure. A verbosity level of 2 enables a detailed text-based logging.
res = fem.newtonrhapson(items=[solid], verbose=2, **loadcase)

# %%
# Results may be viewed in an interactive window.
field.plot("Displacement", component=0).show()
