r"""
Non-homogeneous Deformations
----------------------------

..  topic:: Combine views on deformed meshes and force-displacement curves in a single
    figure.

    * load a hyperelastic material formulation
    
    * create a meshed cube with hexahedron cells
    
    * define a numeric region along with a displacement field

    * apply uniaxial, planar and biaxial loadcases
    
    * solve the problems
    
    * plot force-displacement curves

In this tutorial you'll learn how to plot multiple force-displacement curves along with
views on deformed meshes in one single matplotlib figure. We start with a
:class:`Third-Order-Deformation <felupe.third_order_deformation>` isotropic hyperelastic
material formulation. :class:`~felupe.Hyperelastic` provides a
:meth:`~felupe.Hyperelastic.plot`-method to preview force-stretch curves on
incompressible elementary deformations.
"""
import felupe as fem

strain_energy_function = fem.third_order_deformation
kwargs = dict(C10=0.5, C01=0.1, C11=0.0, C20=-0.1, C30=0.02)

umat = fem.Hyperelastic(strain_energy_function, **kwargs)
ax = umat.plot(incompressible=True)

# %%
# We'd like to generate force-displacement characteristic curves for the
# non-homogeneous deformations :func:`uniaxial <felupe.dof.uniaxial>` and
# :func:`planar <felupe.dof.biaxial>` obtained by simulations using the finite element
# method. Therefore, let's define a meshed :class:`~felupe.Cube` with 5
# :class:`hexahedron <felupe.Hexahedron>` cells and a
# :class:`region <felupe.RegionHexahedron>`.
mesh = fem.Cube(n=6)
region = fem.RegionHexahedron(mesh)

# %%
# We need to initiate a matplotlib :class:`~matplotlib.figure.Figure` with multiple
# subplots. The force-displacement curve is tracked and
# :meth:`plotted <felupe.CharacteristicCurve.plot>` during
# :meth:`evaluation <felupe.Job.evaluate>` of a :class:`~felupe.CharacteristicCurve`-job
# for a :func:`uniaxial <felupe.dof.uniaxial>` compression/tension load case.
# These force-displacement curves are also evaluated for
# :func:`planar <felupe.dof.biaxial>` (shear) tension  and equi-
# :func:`biaxial <felupe.dof.biaxial>` tension. When we plot the planar and biaxial
# force-displacement curves, the ``ax["right"]``-object already has x- and y-labels
# defined and we only need to set the line labels accordingly. Finally, let's add the
# name and the parameters of the
# :class:`Third-Order-Deformation <felupe.third_order_deformation>` material formulation
# to the title of the figure.
import matplotlib.pyplot as plt

fig, ax = plt.subplot_mosaic(
    [["upper left", "right"], ["lower left", "right"]],
    layout="constrained",
    figsize=(6, 4),
    gridspec_kw=dict(width_ratios=[1, 2]),
)

# uniaxial
field = fem.FieldContainer([fem.Field(region, dim=3)])
boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
move = fem.math.linsteps([-0.1, 0, 0.9], num=[1, 9])
step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"]).evaluate()

ax["upper left"] = field.imshow(
    "Principal Values of Logarithmic Strain", ax=ax["upper left"]
)
ax["upper left"].set_title("Uniaxial", fontdict=dict(fontsize="small"))
fig, ax["right"] = job.plot(
    xlabel="Stretch $l/L$ in mm/mm $\longrightarrow$",
    ylabel="Normal Force per Undeformed Area \n $N/A$ in N/mm$^2$ $\longrightarrow$",
    label="Uniaxial",
    ax=ax["right"],
)

# planar
field = fem.FieldContainer([fem.Field(region, dim=3)])
boundaries, loadcase = fem.dof.biaxial(field, moves=(0, 0), clampes=(True, False))
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
step = fem.Step(
    items=[solid], ramp={boundaries["move-right-0"]: move}, boundaries=boundaries
)
job = fem.CharacteristicCurve(
    steps=[step], boundary=boundaries["move-right-0"]
).evaluate()

ax["lower left"] = field.imshow(
    "Principal Values of Logarithmic Strain", ax=ax["lower left"]
)
ax["lower left"].set_title("Planar", fontdict=dict(fontsize="small"))
fig, ax["right"] = job.plot(ax=ax["right"], label="Planar")

title = " ".join([name.title() for name in umat.fun.__name__.split("_")])
fig.suptitle(title, weight="bold")
ax["right"].set_title(
    ", ".join([f"{key}={value}" for key, value in kwargs.items()]),
    fontdict=dict(fontsize="small"),
)
ax["right"].legend()
ax["right"].grid()

# %%
# If the data of the force-displacement curves is needed for the calibration of the
# material parameters on given experimentally determined force-displacement curves, the
# data is available in the axes object of the plot.
data = [(line.get_xdata(), line.get_ydata()) for line in ax["right"].lines]
