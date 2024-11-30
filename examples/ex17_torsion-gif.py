r"""
Animated torsional-shear loading
--------------------------------

.. topic:: Save an animation of a custom boundary condition.

   * assign non-homogeneous displacements to a boundary condition

   * create an animation
   
   * export a GIF-movie
   
   * evaluate the reaction moment on a boundary condition

This example demonstrates how to create a :class:`~felupe.Boundary` for torsional
loading. This is somewhat more complicated than a simple boundary condition with
identical mesh-point values because the values of the mesh-points are different.
However, this is not a problem. FElupe supports arrays to be passed as the ``value``-
argument of a :class:`~felupe.Boundary`. This is even possible in a
:class:`~felupe.Step` with the ``ramp``-argument.
"""

# sphinx_gallery_thumbnail_number = -2
import matplotlib.pyplot as plt
import numpy as np

import felupe as fem

mesh = fem.mesh.Triangle(n=3).expand(n=6)
field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])

boundaries = {
    "bottom": fem.Boundary(field[0], fz=0),
    "top": fem.Boundary(field[0], fz=1),
}
solid = fem.SolidBody(umat=fem.NeoHookeCompressible(mu=1, lmbda=2), field=field)

angles_deg = fem.math.linsteps([0, -30, 0], num=10)
move = []
for phi in angles_deg:
    top = mesh.points[boundaries["top"].points]
    top_rotated = fem.mesh.rotate(
        points=top,
        cells=None,
        cell_type=None,
        angle_deg=phi,
        axis=2,
        center=[0, 0, 1],
    )[0]
    move.append(top_rotated - top)

# %%
# The reaction moment on the centerpoint of the right end face is tracked by a
# ``callback()`` function when we :meth:`~felupe.Job.evaluate` the :class:`~felupe.Job`.
# During the callback, the animated deformed body is recorded in a GIF-file. After all
# frames are recorded, it is important to ``close()`` the plotter.
moment = []

plotter = field.plot(
    "Principal Values of Logarithmic Strain", clim=[0, 0.2], off_screen=True
)
plotter.open_gif("result.gif", fps=5)


def record(stepnumber, substepnumber, substep, plotter):
    "Update the mesh-points and the scalars of the plotter."
    if substepnumber in np.arange(len(move), step=2):
        name = plotter.mesh.active_scalars_info.name
        data = substep.x.evaluate.log_strain(tensor=False).mean(-2)[-1]

        plotter.mesh.points[:] = mesh.points + field[0].values
        plotter.mesh[name] = data

        plotter.write_frame()

    # evaluate the reaction moment at the centerpoint of the right end face
    forces = substep.fun
    M = fem.tools.moment(field, forces, boundaries["top"], centerpoint=[0, 0, 1])
    moment.append(M)


step = fem.Step(items=[solid], ramp={boundaries["top"]: move}, boundaries=boundaries)
job = fem.Job(steps=[step], callback=record, plotter=plotter).evaluate()

plotter.close()

# %%
# Finally, let's plot the reaction moment vs. torsion angle curve.
fig, ax = plt.subplots()
ax.plot(abs(angles_deg), abs(np.array(moment)[:, 2]), "o-")
ax.set_xlabel(r"Torsion Angle $|\phi_3|$ in deg $\rightarrow$")
ax.set_ylabel(r"Torsion Moment $|M_3|$ in Nmm $\rightarrow$")
