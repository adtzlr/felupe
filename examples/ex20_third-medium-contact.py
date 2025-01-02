r"""
Third Medium Contact
--------------------

.. topic:: Frictionless contact method simulated by a third medium [1]_.

   * create a mesh container with multiple meshes
   
   * define multiple solid bodies and create a top-level field
   
   * create an animation of the deformed solid

This contact method uses a third medium for two solid contact bodies with a a Hessian-
based regularization [2]_. First, let's create sub meshes with quad cells for the solid
body. All sub meshes are merged by stacking the meshes of the
:class:`mesh container <felupe.MeshContainer>` into a :class:`mesh <felupe.Mesh>`.
"""

import numpy as np

import felupe as fem

t = 0.1
L = 1
H = 0.5

nt, nL, nH = (2, 10, 4)

a = fem.Rectangle(a=(0, 0), b=(t, t), n=nt)
b = a.translate(move=H - t, axis=1)
c = fem.Rectangle(a=(t, 0), b=(L, t), n=(nL, nt))
d = c.translate(move=H - t, axis=1)
e = fem.Rectangle(a=(0, t), b=(t, H - t), n=(nt, nH))
body = fem.MeshContainer([a, b, c, d, e], merge=True).stack()

# %%
# This is repeated for the medium, where two meshes are added to a mesh container. Extra
# cells are added on the right edge to improve convergence.
f = fem.Rectangle(a=(t, t), b=(L, H - t), n=(nL, nH))
g = fem.Rectangle(a=(L, 0), b=(L + t / (nt - 1), H), n=(2, 2 * (nt - 1) + nH))
medium = fem.MeshContainer([f, g], merge=True).stack()

# %%
# Both stacked meshes are added to a top-level mesh container. For each mesh of the
# container, quad regions and plane-strain vector-fields for the displacements are
# created. The Neo-Hookean isotropic hyperelastic material formulation from the solid
# body is also used for the isotropic part of the background material but with scaled
# down material parameters.
G = 5 / 14
K = 5 / 3
kr = 1e-6
gamma0 = 1e-6

container = fem.MeshContainer([body, medium], merge=True, decimals=6)
regions = [fem.RegionQuad(m, hess=True) for m in container.meshes]
fields = [fem.FieldPlaneStrain(r, dim=2).as_container() for r in regions]
umats = [fem.NeoHooke(mu=G, bulk=K), fem.NeoHooke(mu=G * gamma0, bulk=K * gamma0)]
solids = [fem.SolidBody(umat, f) for umat, f in zip(umats, fields)]

# %%
# A top-level plane-strain field is created on a stacked mesh. This field includes all
# unknowns and is required for the selection of prescribed degrees of freedom as well
# as for Newton's method. The left end edge is fixed and the vertical displacement is
# prescribed on the outermost top-right point.
region = fem.RegionQuad(container.stack())
field = fem.FieldPlaneStrain(region, dim=2).as_container()

bounds = dict(
    fixed=fem.Boundary(field[0], fx=0),
    move=fem.Boundary(field[0], fx=L, fy=H, mode="and", skip=(1, 0)),
)

# %%
# The so-called HuHu-regularization is created by two weak-:func:`forms <felupe.Form>`,
# which are derived from the regularization potential, see Eq.
# :eq:`huhu-regularization` [3]_.
#
# .. math::
#    :label: huhu-regularization
#
#    \Pi = \frac{1}{2} \int_V
#        \mathbb{H}(\boldsymbol{u})~\vdots~\mathbb{H}(\boldsymbol{u})~dV
#
from felupe.math import dddot, hess


@fem.Form(v=fields[1], u=fields[1], kwargs={"kr": 1.0})
def bilinearform():
    return [lambda v, u, kr: kr * dddot(hess(u), hess(v))]


@fem.Form(v=fields[1], kwargs={"kr": 1.0})
def linearform():
    u = fields[1][0]
    return [lambda v, kr: kr * dddot(hess(v), hess(u)[:2, :2, :2])]


regularization = fem.FormItem(
    bilinearform=bilinearform,
    linearform=linearform,
    kwargs={"kr": kr * K * L**2},
)

# %%
# The prescribed displacement is ramped up to the maximum value and released until zero.
move = fem.math.linsteps([0, 1, 0], num=30)
step = fem.Step(
    items=[*solids, regularization],
    ramp={bounds["move"]: -0.4 * move * L},
    boundaries=bounds,
)

# %%
# The deformation is captured in a GIF-file. The top-level field has to be passed as the
# ``x0``-argument for :func:`Newton's method <felupe.newtonrhapson>`, which is called on
# evaluation. After all frames are recorded, it is important to ``close()`` the plotter.
plotter = container.plot(colors=["grey", "white"], line_width=3, off_screen=True)
plotter.open_gif("third-medium-contact.gif", fps=5)


def record(stepnumber, substepnumber, substep, plotter):
    "Update the mesh of the plotter and write a frame."
    if substepnumber in np.arange(len(move), step=5):
        for m in plotter.meshes:
            m.points[:, :2] = region.mesh.points + field[0].values
        plotter.write_frame()


job = fem.Job([step], callback=record, plotter=plotter)
job.evaluate(x0=field)

plotter.close()

# %%
# References
# ~~~~~~~~~~
#
# .. [1] https://en.wikipedia.org/wiki/Third_medium_contact_method
#
# .. [2] G. L. Bluhm, O. Sigmund, and K. Poulios, "Internal contact modeling for finite
#        strain topology optimization", Computational Mechanics, vol. 67, no. 4.
#        Springer Science and Business Media LLC, pp. 1099–1114, Mar. 04, 2021. |DOI-2|.
#
# .. [3] A. H. Frederiksen, O. Sigmund, and K. Poulios, "Topology optimization of self-
#        contacting structures", Computational Mechanics, vol. 73, no. 4. Springer
#        Science and Business Media LLC, pp. 967–981, Oct. 07, 2023. |DOI-3|.
#
# .. |DOI-2| image:: https://zenodo.org/badge/DOI/10.1007/s00466-021-01974-x.svg
#    :target: https://www.doi.org/10.1007/s00466-021-01974-x
#
# .. |DOI-3| image:: https://zenodo.org/badge/DOI/10.1007/s00466-023-02396-7.svg
#    :target: https://www.doi.org/10.1007/s00466-023-02396-7
#
