"""
1/4-model analysis of the two-cylinder problem (reference figure),
with force control.

Geometry (per the analysis-conditions figure):
  upper lobe: quarter disk, center B(0, 1), radius 1  (arc A -> O)
  lower lobe: quarter disk, center D(0, -1), radius 1 (arc O -> C)
  the lobes share the single node O = (0, 0): the cylinder-cylinder
  contact plane is the symmetry plane, and because both arcs curve
  AWAY from the tangent plane, geometric interpenetration is
  impossible -- no contact algorithm is required (the trick of the
  original Code_Aster model).

Boundary conditions (figure):
  OB, OD (x = 0): u_x = 0                (symmetry wall)
  AB (y = 1):     rigid plate tie (uniform u_y and u_x via a multi-point
                  constraint to node B) with the nodal force P/2 at B
                  (force control; the plate sink u_y(B) is measured)
  CD (y = -1):    u_y = 0                (rollers; reaction = -P/2)

  No boundary condition on the arcs: the lobes interact only through
  the shared node O, and both arcs curve away from the tangent plane,
  so geometric interpenetration is impossible (no contact algorithm).
"""

import numpy as np

import felupe as fem


def quarter_model(n=6, E=100.0, nu=0.3):
    # quarter disks: sections [270] = 270-360 deg quarter (arc O -> A for the
    # upper disk), sections [0] = 0-90 deg quarter (arc C -> O for the lower)
    upper = fem.Circle(radius=1.0, centerpoint=[0, 1.0], sections=[270], n=n)
    lower = fem.Circle(radius=1.0, centerpoint=[0, -1.0], sections=[0], n=n)
    container = fem.MeshContainer([upper, lower], merge=True)
    mesh = container.stack()
    region = fem.RegionQuad(mesh)
    field = fem.Field(region, dim=2).as_container()
    solid = fem.SolidBody(
        # fem.NeoHooke returns a non-zero first Piola stress at the undeformed
        # state in 2D (P(F=I) = 0.333 I): use the linear-elastic plane-strain
        # formulation for 2D verification benchmarks
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu),
        field=field,
    )
    return mesh, field, solid


def run(force=0.2, n=6, E=100.0, nu=0.3, verbose=False):
    """Force control: the nodal force P/2 = `force` applied at node B on
    the rigid-plate tie; the result is the plate sink u_y(B) (the support
    reaction equals -force by equilibrium)."""
    mesh, field, solid = quarter_model(n=n, E=E, nu=nu)

    # edge AB: the flat top edge of the upper lobe (y = 1, x > 0); node B
    # = (0, 1) is the center point of the rigid-plate tie
    ids_ab = np.where((mesh.points[:, 1] == 1) & (mesh.points[:, 0] > 0))[0]
    node_B = int(np.where((mesh.points[:, 1] == 1) & (mesh.points[:, 0] == 0))[0][0])

    # supported edge (C .. D): u_y = 0. Note: no boundary condition on the
    # arcs -- the lobes interact only through the shared node O; the arcs
    # curve away from the tangent plane, so no contact algorithm is needed.
    lower_edge = mesh.points[:, 1] == -1

    boundaries = dict(
        wall=fem.Boundary(field[0], mask=mesh.points[:, 0] == 0, skip=(False, True)),
        support=fem.Boundary(field[0], mask=lower_edge, skip=(True, False)),
    )

    # rigid plate: tie the AB edge to node B (uniform u_y and u_x)
    plate = fem.MultiPointConstraint(field, ids_ab, node_B)

    # force control: P/2 at node B
    load = fem.PointLoad(field, [node_B], np.array([[0.0, -force]]))

    # ramp the force linearly: (nsub, 1, 2)
    forces = np.array([np.array([[0.0, -force]]) * f for f in np.linspace(0.0, 1.0, 6)])
    step = fem.Step(
        items=[solid, plate, load], ramp={load: forces}, boundaries=boundaries
    )
    fem.Job(steps=[step]).evaluate(verbose=verbose)

    u = field[0].values.copy()
    sink = float(u[node_B, 1])

    # support reaction (must equal -force by equilibrium)
    solid_fresh = fem.SolidBody(
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu),
        field=fem.Field(fem.RegionQuad(mesh), dim=2).as_container(),
    )
    solid_fresh.field[0].values[:] = u
    fs = solid_fresh.assemble.vector(field=solid_fresh.field).toarray().ravel()
    support_dofs = 2 * np.where(lower_edge)[0] + 1  # u_y dofs on CD
    R = float(fs[support_dofs].sum())

    # save the deformed (solid) vs undeformed (dashed) configuration
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 6))
    cells = mesh.cells
    for c in cells:
        x, y = mesh.points[c, 0], mesh.points[c, 1]
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), "k--", lw=0.3, alpha=0.4)
    xd, yd = mesh.points[:, 0] + u[:, 0], mesh.points[:, 1] + u[:, 1]
    for c in cells:
        ax.plot(np.append(xd[c], xd[c][0]), np.append(yd[c], yd[c][0]), "b-", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"1/4 model, P/2 = {force}: sink = {sink:.4f}")
    fig.savefig("benchmarks/solid-body-contact/quarter-model-deformed.png", dpi=150)
    plt.close(fig)

    return {
        "mesh_points": mesh.npoints,
        "force": force,
        "sink": sink,
        "support_reaction": R,
    }


if __name__ == "__main__":
    res = run()
    print(f"points: {res['mesh_points']}, P/2 = {res['force']}")
    print(f"plate sink u_y(B): {res['sink']:.5f}")
    print(f"support reaction (=-P/2): {res['support_reaction']:.5f}")
