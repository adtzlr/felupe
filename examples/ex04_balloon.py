r"""
Inflation of a hyperelastic balloon
-----------------------------------

.. topic:: Numeric continuation of a hyperelastic balloon under inner pressure.

   * use FElupe with contique

   * plot pressure-displacement curve
    
   * view the deformed balloon

.. admonition:: This example requires external packages.
   :class: hint
   
   .. code-block::
      
      pip install contique

With the help of `contique <https://github.com/adtzlr/contique>`_ it is possible to
apply a numerical parameter continuation algorithm on any system of equilibrium
equations. This advanced tutorial demonstrates the usage of FElupe in conjunction with
`contique <https://github.com/adtzlr/contique>`_. The unstable inflation of a
rectangular hyperelastic balloon demonstrates this powerful approach. The deformed model
and the pressure - displacement curve is plotted.

.. image:: ../../examples/ex04_balloon_sketch.png
   :width: 400px

First, setup a problem in FElupe as usual (mesh, region, field, boundaries, umat, solid
and a pressure boundary). For the material definition we use the **Neo-Hooke** built-in
hyperelastic material formulation.
"""
# sphinx_gallery_thumbnail_number = -1
import contique
import numpy as np

import felupe as fem

mesh = fem.Cube(b=(25, 25, 1), n=(4, 4, 2))
region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])
bounds = fem.dof.symmetry(field[0], axes=(True, True, False))
bounds["fix-z"] = fem.Boundary(field[0], fx=25, fy=25, mode="or", skip=(1, 1, 0))
dof0, dof1 = fem.dof.partition(field, bounds)

umat = fem.Hyperelastic(fem.neo_hooke, mu=1)
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

region_for_pressure = fem.RegionHexahedronBoundary(mesh, mask=(mesh.points[:, 2] == 0))
field_for_pressure = fem.FieldContainer([fem.Field(region_for_pressure, dim=3)])

pressure = fem.SolidBodyPressure(field_for_pressure)


# %%
# The next step involves the problem definition for contique. For details have a look at
# its `README <https://github.com/adtzlr/contique>`_.
def fun(x, lpf, *args):
    "The system vector of equilibrium equations."

    field[0].values.ravel()[dof1] = x
    pressure.update(lpf)

    return fem.tools.fun([solid, pressure], field)[dof1]


def dfundx(x, lpf, *args):
    """The jacobian of the system vector of equilibrium equations w.r.t. the
    primary unknowns."""

    K = fem.tools.jac([solid, pressure], field)

    return fem.solve.partition(field, K, dof1, dof0)[2]


def dfundl(x, lpf, *args):
    """The jacobian of the system vector of equilibrium equations w.r.t. the
    load proportionality factor."""

    pressure.update(1)

    return fem.tools.fun([pressure], field)[dof1]


# %%
# Next we have to init the problem and specify the initial values of unknowns (the
# undeformed configuration). After each completed step of the numeric continuation the
# results are saved.
Res = contique.solve(
    fun=fun,
    jac=[dfundx, dfundl],
    x0=field[0][dof1],
    lpf0=0,
    dxmax=1,
    dlpfmax=1e-3,
    maxsteps=150,
    rebalance=True,
)
X = np.array([res.x for res in Res])

# %%
# The unstable pressure-controlled equilibrium path is plotted as pressure-displacement
# curve.
import matplotlib.pyplot as plt

plt.plot(X[:, 2], X[:, -1], lw=3)
plt.xlabel(r"Max. Displacement $u_3(X_1=X_2=X_3=0)$ $\longrightarrow$")
plt.ylabel(r"Load-Proportionality-Factor $\lambda$ $\longrightarrow$")

# %%
# The deformed configuration of the solid body is plotted.
solid.plot("Principal Values of Cauchy Stress").show()
