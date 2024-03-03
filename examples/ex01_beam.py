r"""
Cantilever beam under gravity
-----------------------------

.. admonition:: Apply a gravity load on a solid body.
   :class: note

   * create a solid body and apply the gravity load
   
   * linear-elastic analysis

The displacement due to gravity of a cantilever beam with young's modulus
:math:`E=206000 N/mm^2`, poisson ratio :math:`\nu=0.3`, length :math:`L=2000 mm` and
cross section area :math:`A=a \cdot a` with :math:`a=100 mm` is to be evaluated within
a linear-elastic analysis[1]_.

.. image:: ../../examples/ex01_beam_sketch.png

First, let's create a meshed cube out of hexahedron cells with ``n=(181, 9, 9)`` points
per axis. A numeric region created on the mesh represents the cantilever beam. A three-
dimensional vector-valued displacement field is initiated on the region.
"""
import felupe as fem

cube = fem.Cube(a=(0, 0, 0), b=(2000, 100, 100), n=(101, 6, 6))
region = fem.RegionHexahedron(cube)
displacement = fem.Field(region, dim=3)
field = fem.FieldContainer([displacement])

# %%
# A fixed boundary condition is applied on the left end of the beam.
boundaries = {"fixed": fem.dof.Boundary(displacement, fx=0)}

# %%
# The material behaviour is defined through a built-in isotropic linear-elastic material
# formulation.
umat = fem.LinearElastic(E=206000, nu=0.3)
solid = fem.SolidBody(umat=umat, field=field)

# %%
# The body force is defined by a (constant) gravity field on a solid body.
#
# .. math::
#    \delta W_{ext} = \int_v \delta \boldsymbol{u} \cdot \rho \boldsymbol{g} ~ dv
gravity = fem.SolidBodyGravity(field, gravity=[0, 0, 9810], density=7850 * 1e-12)

# %%
# Inside a Newton-Rhapson procedure, the weak form of linear elasticity is assembled
# into the stiffness matrix and the applied gravity field is assembled into the body
# force vector.
step = fem.Step(items=[solid, gravity], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate()

# %%
# The maximum displacement are plotted on a 100x scaled deformed configuration.
field.plot("Displacement", component=None, factor=100).show()

# %%
# References
# ~~~~~~~~~~
# .. [1] Glenk C. et al., *Consideration of Body Forces within Finite Element Analysis*,
#    Strojniški vestnik - Journal of Mechanical Engineering, Faculty of Mechanical
#    Engineering, 2018, |DOI|.
#
# .. |DOI| image:: https://zenodo.org/badge/DOI/10.5545/sv-jme.2017.5081.svg
#    :target: https://www.doi.org/10.5545/sv-jme.2017.5081
