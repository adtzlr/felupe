r"""
Nonlinear Truss Analysis
------------------------
This example describes a three-dimensional system of trusses with 5 points and 6 cells
(in total 5 active degrees of freedom). Given to its geometry, strong geometric
nonlinearities are to be expected when the given reference load is applied. First,  we
create a mesh, where points are defined by their coordinates and cells by pairs of point
connectivities.
"""

# sphinx_gallery_thumbnail_number = -1

import contique
import matplotlib.pyplot as plt
import numpy as np

import felupe as fem

mesh = fem.Mesh(
    points=[
        [2.5, 0, 0],
        [-1.25, 1.25, 0],
        [1, 2, 0],
        [-0.5, 1.5, 1.5],
        [-2.5, 4.5, 2.5],
    ],
    cells=[[0, 3], [1, 3], [2, 3], [2, 4], [1, 4], [3, 4]],
    cell_type="line",
)
region = fem.RegionTruss(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

# %%
# Beside points and cells we have to define displacement boundary conditions, external
# forces and the constitutive material formulation for the trusses.
boundaries = fem.BoundaryDict(
    fixed_xyz=fem.Boundary(field[0], mask=[1, 1, 1, 0, 0]),
    fixed_y=fem.Boundary(field[0], mask=[0, 0, 0, 0, 1], skip=(1, 0, 1)),
)
dof0, dof1 = fem.dof.partition(field, boundaries)

solid = fem.TrussBody(
    umat=fem.LinearElastic1D(E=1),
    field=field,
    area=[0.75, 1, 0.5, 0.75, 1, 1],
)

force_3 = np.array([1, 1, -1])
force_4 = np.array([-2, 0, -2])

load_3 = fem.PointLoad(field, [3], force_3)
load_4 = fem.PointLoad(field, [4], force_4)

# %%
# The undeformed configuration is plotted in a 3d-view.
plotter = mesh.plot(
    line_width=10,
    render_lines_as_tubes=True,
    show_edges=False,
)
plotter.add_points(
    mesh.points,
    color="black",
    point_size=20,
    render_points_as_spheres=True,
)
plotter = boundaries.plot(plotter=plotter)
plotter = load_3.plot(plotter=plotter, color="green", deformed=False)
plotter = load_4.plot(plotter=plotter, color="green", deformed=False)

plotter.show()


# %%
# For the numeric continuation, the equilibrium function ``fun`` and its derivatives
# w.r.t. the displacement field ``dfun_du`` and the load-proportionality-factor
# ``dfun_dlpf`` have to be defined. Here, we're only interested in the active degrees of
# freedom.
def fun(x, lpf, *args):
    field[0].values.ravel()[dof1] = x
    load_3.update(force_3 * lpf)
    load_4.update(force_4 * lpf)
    return fem.tools.fun([solid, load_3, load_4], field)[dof1]


def dfun_du(x, lpf, *args):
    field[0].values.ravel()[dof1] = x
    K = fem.tools.jac([solid, load_3, load_4], field)
    return fem.solve.partition(field, K, dof1, dof0)[2]


def dfun_dlpf(x, lpf, *args):
    load_3.update(force_3)
    load_4.update(force_4)
    return fem.tools.fun([load_3, load_4], field)[dof1]


# %%
# Now that the model is finished, some additional settings have to be chosen. Initial
# allowed incremental system vector components for both the displacement vector and the
# load-proportionality-factor (LPF) have to be specified. We use ``dlpf = 0.005`` and
# ``du = 0.05`` (figured out after some trial and error). Both parameters can't be
# specified automatically, as they depend on the model configuration. The job will be
# limited to a total amount of 163 increments (again, the total number has been figured
# out after some job runs to get good looking plots).
res = contique.solve(
    fun=fun,
    jac=[dfun_du, dfun_dlpf],
    x0=field[0][dof1],
    lpf0=0,
    dxmax=0.05,
    dlpfmax=0.005,
    maxsteps=163,
    rebalance=True,
    overshoot=1.25,
    tol=1e-8,
    low=1e-2,
    high=4,
    maxiter=8,
)
X = np.array([r.x for r in res])

# %%
# To visualize the deformed state of the model for increment 40 the deformed model plot
# is generated.
field[0].values.ravel()[dof1] = X[40, :-1]
force = solid.evaluate.gradient(field) * solid.area
plotter = field.view(cell_data={"Force": force}).plot(
    "Force",
    line_width=10,
    show_undeformed=False,
    view="xy",
    cmap="coolwarm",
    clim=[-abs(force).max(), abs(force).max()],
    render_lines_as_tubes=True,
    show_edges=False,
)
plotter.add_points(
    mesh.points + field[0].values,
    color="black",
    point_size=20,
    render_points_as_spheres=True,
)
plotter.show()

# %%
# Path-tracing of the displacement-LPF curves
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The path-tracing of the deformation process is shown as a History Plot of
# Displacement-LPF curves for all active DOF. Strong geometrical nonlinearities are
# observed for all active DOF.
fig, ax = plt.subplots()
ax.plot(*X[:, [0, -1]].T, ".-", label="Point 3")
ax.plot(*X[:, [3, -1]].T, ".-", label="Point 4")
ax.set_xlabel("Displacement X")
ax.set_ylabel("LPF")
ax.legend()

# %%
fig, ax = plt.subplots()
ax.plot(*X[:, [1, -1]].T, ".-", label="Point 3")
ax.set_xlabel("Displacement Y")
ax.set_ylabel("LPF")
ax.legend()

# %%
fig, ax = plt.subplots()
ax.plot(*X[:, [2, -1]].T, ".-", label="Point 3")
ax.plot(*X[:, [4, -1]].T, ".-", label="Point 4")
ax.set_xlabel("Displacement Z")
ax.set_ylabel("LPF")
ax.legend()
