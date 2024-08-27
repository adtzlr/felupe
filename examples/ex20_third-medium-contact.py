"""
Third Medium Contact
--------------------

.. topic:: Frictionless contact method simulated by a third medium.

   * create a mesh container with multiple meshes
   
   * define multiple solid bodies and create a top-level field
   
   * plot the Cauchy stress component :math:`\sigma_{22}`

This contact method uses a third medium for two solid contact bodies.

..  note::
    
    This example starts with a transversal isotropic hyperelastic strain energy function
    for the third medium [1]_. Instead of a transversal isotropic hyperelastic strain
    energy function, a Hessian-based regularization is also possible ([2]_, [3]_).

First, let's create a rectangular mesh with quad cells for the upper solid body. The
:meth:`~felupe.Mesh.add_runouts`-method is used to modify the geometry according to
[1]_. Another rectangle is used for the elastic foundation. As both solid bodies will
use the same material model formulation and properties, both meshes are merged by
stacking the meshes of the :class:`mesh container <felupe.MeshContainer>` into a
:class:`mesh <felupe.Mesh>`.
"""

# sphinx_gallery_thumbnail_number = -1
import numpy as np
import tensortrax.math as tm

import felupe as fem
from felupe.math import dddot, hess

top = fem.Rectangle(a=(-1, 1.1), b=(1, 2), n=(11, 4))
block = top.add_runouts([-1 / 3], normalize=True, centerpoint=[0, 2], exponent=2)
assert np.isclose(block.y[block.x == 1].min(), 1.4)

rect = fem.Rectangle(a=(-2, 0), b=(2, 1), n=(21, 6))
structure = fem.MeshContainer([block, rect], merge=True).stack()

# %%
# The mesh for the background material (third medium) is filled with quads between two
# line meshes with identical number of points. The points inside are shifted slightly
# to reproduce the mesh from [1]_.
line = fem.mesh.Line(n=11)
lower = line.copy(
    points=rect.points[
        np.logical_and.reduce([rect.y == rect.y.max(), rect.x >= -1, rect.x <= 1])
    ],
)
upper = line.copy(points=block.points[top.y == top.y.min()])
medium = lower.fill_between(upper, n=3)

# %%
# The left- and right-rectangles are also added to the third medium to visualize the
# deformation of the background mesh in a region where no contact occurs.
air = [
    fem.Rectangle(
        a=(-2, block.y[block.x == -1].min()), b=(-1, 2), n=(6, 4)
    ),  # top-left
    fem.Rectangle(
        a=(-2, 1), b=(-1, block.y[block.x == 1].min()), n=(6, 3)
    ),  # bottom-l.
    fem.Rectangle(a=(1, block.y[block.x == 1].min()), b=(2, 2), n=(6, 4)),  # top-right
    fem.Rectangle(a=(1, 1), b=(2, block.y[block.x == 1].min()), n=(6, 3)),  # bottom-r.
]
medium = fem.MeshContainer([medium, *air], merge=True).stack()
medium.points[
    np.logical_and(medium.y == 2, np.logical_or(medium.x < -1, medium.x > 1)), 1
] -= 0.02

# %%
# Both meshes are added to a mesh container. For each mesh of the container, quad
# regions and plane-strain vector-fields for the displacements are created.
container = fem.MeshContainer([structure, medium], merge=True, decimals=6)

plotter = container.plot(colors=["darkgrey", "lightgrey"])
plotter.show_bounds(xtitle="", ytitle="")
plotter.add_legend([("solid", "darkgrey"), ("medium", "lightgrey")], bcolor="white")
plotter.show()

regions = [
    fem.RegionQuad(container.meshes[0]),
    fem.RegionQuad(container.meshes[1]),
]
fields = [fem.FieldContainer([fem.FieldPlaneStrain(r, dim=2)]) for r in regions]

# %%
# A top-level plane-strain field is created on a stacked mesh. This field includes all
# unknowns and is required for the selection of prescribed degrees of freedom as well
# as for Newton's method.
mesh = container.stack()
region = fem.RegionQuad(mesh)
field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

boundaries, loadcase = fem.dof.uniaxial(field, axis=1, sym=False, clamped=True)
move = fem.math.linsteps([0, -0.1], num=5)


# %%
# The background material is assumed to be transversly isotropic hyperelastic [1]_,
# splitted into isotropic and anisotropic parts. The isotropic part is used for rigid
# body movements and hence, its elastic stiffness must be magnitudes lower than the one
# from the solid bodies. The anisotropic part acts in the direction of the
# smallest principal (squared) stretch which is used as an assumption for the surface
# normal vector.
def third_medium_contact(C, mu=1, a1=1, a2=1, a3=1, a4=23, a5=100, tol=0.5):
    """Transversly isotropic hyperelastic strain energy density function for the
    third medium."""

    # isotropic invariants
    I1 = tm.trace(C)
    I2 = (I1**2 - tm.trace(C @ C)) / 2
    I3 = tm.linalg.det(C)

    # anisotropic invariant for minimum principal squared stretch
    w, M = tm.linalg.eigh(C)
    λ2min = w[0]
    J4 = tm.special.ddot(C, M[0])

    # isotropic material paramter
    d = 2 * a1 + 4 * a2 + 2 * a3

    # deformation dependent material parameter
    c = tm.if_else(λ2min >= tol, 0 * λ2min, a5 / λ2min**2)

    # strain energy density functions
    W_iso = a1 * I1 + a2 * I2 + a3 * I3 - d * tm.log(I3) / 2
    W_aniso = c / (a4 + 1) * (1 - J4) ** (a4 + 1)

    return mu * W_iso + W_aniso


# %%
# A :class:`Neo-Hookean <felupe.NeoHooke>` material model formulation is used for both
# solid bodies. The third medium uses a general
# :class:`hyperelastic <felupe.Hyperelastic>` material class with automatic
# differentiation.
solids = [
    fem.SolidBody(fem.NeoHooke(mu=2692.3, bulk=5833.3), fields[0]),
    fem.SolidBody(fem.Hyperelastic(third_medium_contact, mu=1, a5=1e-2), fields[1]),
]

step = fem.Step(solids, boundaries=boundaries, ramp={boundaries["move"]: move})
job = fem.Job([step]).evaluate(x0=field)

# %%
# The vertical component of the Cauchy stress :math:`\sigma_{22}` is shifted to and
# averaged inside the solid body at the mesh points.
plotter = solids[0].plot("Cauchy Stress", component=1, project=fem.topoints)
plotter.show_bounds(xtitle="", ytitle="")
plotter.show()

# %%
# The deformed wireframe mesh shows the deformation of the background material.
field.plot(style="wireframe", nonlinear_subdivision=2, line_width=2).show()

# %%
# Instead of using a transversly anisotropic material formulation for the third medium
# contact method, a hessian-regularization may be used. Therefore, we must reload the
# region of the background material and enable the flag to evaluate the Hessian.
regions[1].reload(hess=True)


# %%
# The so-called HuHu-regularization is created by two weak-:func:`form <felupe.Form>`
# expressions. This time, the Neo-Hookean isotropic hyperelastic material formulation
# from the solid bodies is also used for the isotropic part of the background material
# but with scaled down material parameters.
@fem.Form(v=fields[1], u=fields[1])
def bilinearform():
    return [lambda v, u: dddot(hess(v), hess(u))]


@fem.Form(v=fields[1])
def linearform():
    u = fields[1][0]
    return [lambda v: dddot(hess(v), hess(u)[:2, :2, :2])]


solids = [
    fem.SolidBody(fem.NeoHooke(mu=2692.3, bulk=5833.3), fields[0]),
    fem.SolidBody(fem.NeoHooke(mu=2692.3 * 1e-3, bulk=5833.3 * 1e-3), fields[1]),
    fem.FormItem(bilinearform, linearform),
]

# %%
# The fields are resetted in order to restart the simulation with the Hessian-based
# regularization. This time, the enforced vertical displacement on the top edge of the
# upper body is doubled.
field[0].fill(0.0)
[f[0].fill(0.0) for f in fields]

move = fem.math.linsteps([0, -0.2], num=15)
step = fem.Step(solids, boundaries=boundaries, ramp={boundaries["move"]: move})
job = fem.Job([step]).evaluate(x0=field)

# %%
# The vertical component of the Cauchy stress :math:`\sigma_{22}` is plotted for the
# maximum applied vertical displacement.
plotter = solids[0].plot("Cauchy Stress", component=1, project=fem.topoints)
plotter.show_bounds(xtitle="", ytitle="")
plotter.show()

# %%
# Now, the upper solid body is moved along the horizontal direction and the vertical
# Cauchy stress component is plotted again.
boundaries, loadcase = fem.dof.shear(field, axes=(0, 1), sym=False, moves=(0, 0, -0.2))
move = fem.math.linsteps([0, 0.5], num=10)

step = fem.Step(solids, boundaries=boundaries, ramp={boundaries["move"]: move})
job = fem.Job([step]).evaluate(x0=field)

plotter = solids[0].plot("Cauchy Stress", component=1, project=fem.topoints)
plotter.show_bounds(xtitle="", ytitle="")
plotter.show()

# %%
# The deformed wireframe mesh shows again the deformation of the background material.
field.plot(style="wireframe", nonlinear_subdivision=2, line_width=2).show()

# %%
# References
# ~~~~~~~~~~
# .. [1] P. Wriggers, J. Schröder, and A. Schwarz, "A finite element method for contact
#        using a third medium", Computational Mechanics, vol. 52, no. 4. Springer
#        Science and Business Media LLC, pp. 837–847, Mar. 30, 2013. |DOI-1|.
#
# .. [2] https://en.wikipedia.org/wiki/Third_medium_contact_method
#
# .. [3] G. L. Bluhm, O. Sigmund, and K. Poulios, "Internal contact modeling for finite
#        strain topology optimization", Computational Mechanics, vol. 67, no. 4.
#        Springer Science and Business Media LLC, pp. 1099–1114, Mar. 04, 2021. |DOI-3|.
#
# .. |DOI-1| image:: https://zenodo.org/badge/DOI/10.1007/s00466-013-0848-5.svg
#    :target: https://www.doi.org/10.1007/s00466-013-0848-5
#
# .. |DOI-3| image:: https://zenodo.org/badge/DOI/10.1007/s00466-013-0848-5.svg
#    :target: https://www.doi.org/10.1007/s00466-013-0848-5
