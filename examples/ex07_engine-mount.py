"""
Engine Mount
------------

.. topic:: A rubberlike-metal component used as an engine-mount.

   * read and combine mesh files

   * define an isotropic hyperelastic solid body

   * create consecutive steps and add them to a job

   * export and plot characteristic curves


An engine-mount is loaded by a combined vertical and horizontal displacement. What is
being looked for are the characteristic force-displacement curves in vertical and
horizontal directions as well as the logarithmic strain distribution inside the rubber.
The air inside the structure is meshed as a hyperelastic solid with no volumetric part
of the strain energy function for a simplified treatment of the rubber contact. The
metal parts are simplified as rigid bodies. Three mesh files are provided for this
example:

* a `mesh for the metal parts <../_static/ex07_engine-mount_mesh-metal.vtk>`_,

* a `mesh for the rubber blocks <../_static/ex07_engine-mount_mesh-rubber.vtk>`_
  as well as

* a `mesh for the air <../_static/ex07_engine-mount_mesh-air.vtk>`_ inside the
  engine mount.
"""

# sphinx_gallery_thumbnail_number = -1
import numpy as np

import felupe as fem

metal = fem.mesh.read("ex07_engine-mount_mesh-metal.vtk", dim=2)[0]
rubber = fem.mesh.read("ex07_engine-mount_mesh-rubber.vtk", dim=2)[0]
air = fem.mesh.read("ex07_engine-mount_mesh-air.vtk", dim=2)[0]


# %%
# Sub-regions and fields for all materials are generated. The sub-fields must be merged
# to generate both the displacement fields for metal / rubber / air and a top-level
# displacement field.
regions = [fem.RegionQuad(m) for m in [metal, rubber, air]]
fields, field = fem.FieldContainer(
    [fem.FieldsMixed(r, n=1, planestrain=True) for r in regions]
).merge()


# %%
# The boundary conditions are created on the global displacement field. First, a mask
# for all points related to the metal parts is created. Then, this mask is splitted into
# the inner and the outer metal part. The global field holds a mesh-container attribute
# which may be used for plotting.
mesh = field.region.mesh
x, y = mesh.points.T
radius = np.sqrt(x**2 + y**2)

only_cells_metal = np.isin(
    np.arange(mesh.npoints), np.unique(fields[0].region.mesh.cells)
)
inner = np.logical_and(only_cells_metal, radius <= 45)
outer = np.logical_and(only_cells_metal, radius > 45)

boundaries = fem.BoundaryDict(
    fixed=fem.Boundary(field[0], mask=outer),
    u_x=fem.Boundary(field[0], mask=inner, skip=(0, 1)),
    u_y=fem.Boundary(field[0], mask=inner, skip=(1, 0)),
)
plotter = field.mesh_container.plot(colors=["grey", "black", "white"])
boundaries.plot(plotter=plotter, scale=0.02).show()

# %%
# The material behaviour of the rubberlike solid is defined through a built-in hyperelastic
# isotropic compressible Neo-Hookean material formulation. A solid body applies the material formulation on the
# displacement field. The air is also simulated by a Neo-Hookean material formulation
# but with no volumetric contribution and hence, no special mixed-field treatment is
# necessary here. A crucial parameter is the shear modulus which is used for the
# simulation of the air. The air is meshed and simulated to capture the contacts of the
# rubber blocks inside the engine mount during the deformation. Hence, its overall
# stiffness contribution must be as low as possible. Here, ``1 / 25`` of the shear
# modulus of the rubber is used. The bulk modulus of the rubber is lowered to provide a
# more realistic deformation for the three-dimensional component simulated by a plane-
# strain analysis.
shear_modulus = 1
rubber = fem.SolidBodyNearlyIncompressible(
    umat=fem.NeoHooke(mu=shear_modulus), field=fields[1], bulk=100
)
air = fem.SolidBody(umat=fem.NeoHooke(mu=shear_modulus / 25), field=fields[2])

# %%
# After defining the consecutive load steps, the simulation model is ready to be solved.
# As we are not interested in the strains of the simulated air, a trimmed mesh is
# specified during the evaluation of the characteristic-curve job. The lateral force-
# displacement curves are plotted for the two different levels of vertical displacement.
thickness = 100
vertical = fem.Step(
    items=[rubber, air],
    ramp={boundaries["u_y"]: fem.math.linsteps([0, 3], num=3)},
    boundaries=boundaries,
)
job = fem.CharacteristicCurve(steps=[vertical], boundary=boundaries["u_y"]).evaluate(
    x0=field, tol=1e-1
)
figv, axv = job.plot(
    xlabel=r"Displacement $u_y$ in mm $\longrightarrow$",
    ylabel=r"Normal Force $F_y$ in kN $\longrightarrow$",
    xaxis=1,
    yaxis=1,
    yscale=1 / 1000 * thickness,
    ls="-",
    lw=3,
)

horizontal = fem.Step(
    items=[rubber, air],
    ramp={boundaries["u_x"]: 8 * fem.math.linsteps([0, 1, 0, -1, 0], num=8)},
    boundaries=boundaries,
)
job = fem.CharacteristicCurve(steps=[horizontal], boundary=boundaries["u_y"]).evaluate(
    x0=field, tol=1e-1
)
figh, axh = job.plot(
    xlabel=r"Displacement $u_x$ in mm $\longrightarrow$",
    ylabel=r"Normal Force $F_x$ in kN $\longrightarrow$",
    yscale=1 / 1000 * thickness,
    lw=3,
    color="C0",
    label=r"$u_y=+3$ mm",
)

vertical = fem.Step(
    items=[rubber, air],
    ramp={boundaries["u_y"]: fem.math.linsteps([3, 0, -6], num=[7, 6])},
    boundaries=boundaries,
)
job = fem.CharacteristicCurve(steps=[vertical], boundary=boundaries["u_y"]).evaluate(
    x0=field, tol=1e-1
)
figv, axv = job.plot(
    xaxis=1,
    yaxis=1,
    yscale=1 / 1000 * thickness,
    ls="-",
    lw=3,
    color="C0",
    ax=axv,
)
horizontal = fem.Step(
    items=[rubber, air],
    ramp={boundaries["u_x"]: 5 * fem.math.linsteps([0, 1, 0, -1], num=5)},
    boundaries=boundaries,
)
job = fem.CharacteristicCurve(steps=[horizontal], boundary=boundaries["u_y"]).evaluate(
    x0=field, tol=1e-1
)
figh, axh = job.plot(
    yscale=1 / 1000 * thickness,
    lw=3,
    color="C1",
    label=r"$u_y=-6$ mm",
    ax=axh,
)
axh.legend()

# %%
# The maximum principal values of the logarithmic strain tensors are plotted on the
# deformed configuration.
plotter = fields[0].plot(color="grey", show_edges=False)
plotter = fields[1].plot("Principal Values of Logarithmic Strain", plotter=plotter)
plotter.show()
