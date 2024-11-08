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
and a pressure boundary). For the material definition we use the
:class:`Neo-Hooke <felupe.NeoHookeCompressible>` built-in
hyperelastic material formulation, see Eq. :eq:`neo-hookean-strain-energy`.

.. math::
   :label: neo-hookean-strain-energy

   \psi = \frac{\mu}{2} \left( 
       \text{tr}(\boldsymbol{C}) - \ln(\det(\boldsymbol{C})) 
   \right)
"""

# sphinx_gallery_thumbnail_number = -1
import contique
import numpy as np

import felupe as fem

mesh = fem.Rectangle(b=(1, 25), n=(2, 4))
region = fem.RegionQuad(mesh)
field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
boundaries = fem.dof.symmetry(field[0], axes=(0, 1))
boundaries["fix-y"] = fem.Boundary(field[0], fy=mesh.y.max(), mode="or", skip=(0, 1))
dof0, dof1 = fem.dof.partition(field, boundaries)

umat = fem.NeoHookeCompressible(mu=1)
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

region_for_pressure = fem.RegionQuadBoundary(mesh, mask=(mesh.x == 0), ensure_3d=True)
field_for_pressure = fem.FieldContainer(
    [fem.FieldAxisymmetric(region_for_pressure, dim=2)]
)

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
    control0=(0, 1),
    dxmax=1.0,
    dlpfmax=0.0075,
    maxsteps=65,
    rebalance=True,
    tol=1e-3,
    decrease=1.2,
    increase=0.4,
)
X = np.array([res.x for res in Res])

# %%
# The unstable pressure-controlled equilibrium path is plotted as pressure-displacement
# curve.
import matplotlib.pyplot as plt

plt.plot(X[:, 0], X[:, -1], "x-", lw=3)
plt.xlabel(r"Max. Displacement $u_1(X_1=X_2=0)$ $\longrightarrow$")
plt.ylabel(r"Load-Proportionality-Factor $\lambda$ $\longrightarrow$")

# %%
# The 3d-deformed configuration of the solid body is plotted.
mesh_3d = mesh.revolve(phi=90, n=6)
region_3d = fem.RegionHexahedron(mesh_3d)
values = mesh.copy(points=field[0].values).revolve(phi=90, n=6).points
u_3d = fem.Field(region_3d, values=values, dim=3)
field_3d = fem.FieldContainer([u_3d])
solid_3d = fem.SolidBodyNearlyIncompressible(umat, field_3d, bulk=5000)
solid_3d.plot("Principal Values of Cauchy Stress", project=fem.topoints).show()
