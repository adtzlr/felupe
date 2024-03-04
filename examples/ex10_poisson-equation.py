r"""
Poisson Equation
----------------
The `Poisson equation <https://en.wikipedia.org/wiki/Poisson%27s_equation>`_

.. math::
    
   \text{div}(\boldsymbol{\nabla} v) + f = 0 \quad \text{in} \quad \Omega

with fixed boundaries on the bottom, top, left and right end-edges

.. math::
   
   v = 0 \quad \text{on} \quad \Gamma_v

and a unit load

.. math::
   
   f = 1 \quad \text{in} \quad \Omega

is solved on a unit rectangle with triangles.
"""

import felupe as fem

mesh = fem.Rectangle(n=2**5).triangulate()
region = fem.RegionTriangle(mesh)
scalar = fem.Field(region)
field = fem.FieldContainer([scalar])


# %%
# The Poisson equation is transformed into integral form representation by the
# `divergence (Gauss's) theorem <https://en.wikipedia.org/wiki/Divergence_theorem>`_.
#
# .. math::
#
#    \int_\Omega \boldsymbol{\nabla} v \cdot \boldsymbol{\nabla} u \ d\Omega
#        = \int_\Omega  f \cdot v \ d\Omega
#
# For the :func:`~felupe.newtonrhapson` to converge, the *linear form* of the Poisson
# equation is also required.


@fem.Form(v=field, u=field, grad_v=[True], grad_u=[True])
def a():
    "Container for a bilinear form."
    return [lambda gradv, gradu: fem.math.ddot(gradv, gradu)]


@fem.Form(v=field, grad_v=[True])
def L():
    "Container for a linear form."
    return [lambda gradv: fem.math.ddot(gradv, field[0].grad())]


@fem.Form(v=field, grad_v=[False])
def Lext():
    "Container for a linear form."
    return [lambda v: -1.0 * v]


poisson = fem.FormItem(bilinearform=a, linearform=L)
load = fem.FormItem(linearform=Lext)

boundaries = {
    "bottom-or-left": fem.Boundary(field[0], fx=0, fy=0, mode="or"),
    "top-or-right": fem.Boundary(field[0], fx=1, fy=1, mode="or"),
}

step = fem.Step([poisson, load], boundaries=boundaries)
job = fem.Job([step]).evaluate()

view = mesh.view(point_data={"Field": field[0].values})
view.plot("Field", show_undeformed=False).show()
