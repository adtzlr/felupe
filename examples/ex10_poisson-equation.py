r"""
Poisson Equation
----------------
The `Poisson equation <https://en.wikipedia.org/wiki/Poisson%27s_equation>`_

.. math::
    
   \text{div}(\boldsymbol{\nabla} u) + f = 0 \quad \text{in} \quad \Omega

with fixed boundaries on the bottom, top, left and right end-edges

.. math::
   
   u = 0 \quad \text{on} \quad \Gamma_u

and a unit load

.. math::
   
   f = 1 \quad \text{in} \quad \Omega

is solved on a unit rectangle with triangles.
"""

# %%
# The Poisson equation is transformed into integral form representation by the
# `divergence (Gauss's) theorem <https://en.wikipedia.org/wiki/Divergence_theorem>`_.
#
# .. math::
#
#    \int_\Omega \boldsymbol{\nabla} (\delta u) \cdot \boldsymbol{\nabla} (\Delta u)
#        \ d\Omega = \int_\Omega  \delta u \cdot f \ d\Omega
#

import felupe as fem

mesh = fem.Rectangle(n=2**5).triangulate()
region = fem.RegionTriangle(mesh)
u = fem.Field(region, dim=1)
field = fem.FieldContainer([u])

boundaries = dict(
    bottom=fem.Boundary(u, fy=0),
    top=fem.Boundary(u, fy=1),
    left=fem.Boundary(u, fx=0),
    right=fem.Boundary(u, fx=1),
)

solid = fem.SolidBody(umat=fem.Laplace(), field=field)
unit_load = fem.SolidBodyForce(field=field, values=1.0)

step = fem.Step([solid, unit_load], boundaries=boundaries)
job = fem.Job([step]).evaluate()

view = mesh.view(point_data={"Field": u.values})
view.plot("Field").show()
