"""
Half-model analysis of the two-cylinder problem (reference figure),
with force control.

The slide labels the two-lobe shape a "1/4 model", but the load P/2
(half of the total P) shows that it is a 1/2 model: the right half of
the two-cylinder stack, cut by the symmetry plane x = 0.

Geometry (undeformed):
  upper lobe: half disk, center (0, +0.5), radius 0.5 (arc B -> A -> O)
  lower lobe: half disk, center (0, -0.5), radius 0.5 (arc O -> C -> D)
  the lobes share the single node O = (0, 0): the cylinder-cylinder
  contact plane is the symmetry plane y = 0, and because both arcs
  curve AWAY from the tangent plane, geometric interpenetration is
  impossible -- no contact algorithm is required (the trick of the
  original Code_Aster model). The flat edge AB in the figure is the
  DEFORMED arc pressed onto the rigid plate (uniform-u_y tie).

Boundary conditions (figure):
  OB, OD (x = 0): u_x = 0                     (symmetry wall)
  arc B-A (upper): total force P/2 applied    (rigid plate, force control;
                                               the plate sink u_y is measured)
  arc C-D (lower): u_y = 0                    (rollers; reaction = -P/2)
"""

import numpy as np

import felupe as fem


def half_model(n=6, E=100.0, nu=0.3):
    # right half-disks: sections [270] gives the 270-360 deg quarter of the
    # circle, [0] the 0-90 deg quarter; two quarters per disk = half disk
    upper = fem.Circle(radius=0.5, centerpoint=[0, 0.5], sections=[270, 0], n=n)
    lower = fem.Circle(radius=0.5, centerpoint=[0, -0.5], sections=[270, 0], n=n)
    container = fem.MeshContainer([upper, lower], merge=True)
    mesh = container.stack()
    region = fem.RegionQuad(mesh)
    field = fem.Field(region, dim=2).as_container()
    solid = fem.SolidBody(
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu), field=field
    )
    return mesh, field, solid


def run(force=0.2, n=6, E=100.0, nu=0.3, verbose=False):
    """Force control: total P/2 = `force` (per unit thickness) applied on
    the plate zone; the result is the plate sink u_y (the support
    reaction equals -force by equilibrium)."""
    mesh, field, solid = half_model(n=n, E=E, nu=nu)

    # arcs of the two half-disks (x >= 0)
    upper_arc = (
        np.abs(np.sqrt(mesh.points[:, 0] ** 2 + (mesh.points[:, 1] - 0.5) ** 2) - 0.5)
        < 1e-6
    ) & (mesh.points[:, 0] >= 0)
    lower_arc = (
        np.abs(np.sqrt(mesh.points[:, 0] ** 2 + (mesh.points[:, 1] + 0.5) ** 2) - 0.5)
        < 1e-6
    ) & (mesh.points[:, 0] >= 0)

    # contact plane (A .. O and O .. C): u_y = 0 (symmetry: the plane stays
    # flat -- this replaces the cylinder-cylinder contact algorithm)
    # support (C .. D): u_y = 0 (rollers / floor)
    plate_zone = upper_arc & (mesh.points[:, 1] >= 0.5 - 1e-9)
    contact_zone = (upper_arc & ~plate_zone) | lower_arc

    boundaries = dict(
        wall=fem.Boundary(field[0], mask=mesh.points[:, 0] == 0, skip=(False, True)),
        contact=fem.Boundary(field[0], mask=contact_zone, skip=(True, False)),
    )

    # force control: P/2 applied as nodal forces on the plate zone
    plate_ids = np.where(plate_zone)[0]
    values_load = np.zeros((mesh.npoints, 2))
    values_load[plate_ids, 1] = -force / len(plate_ids)
    load = fem.PointLoad(field, plate_ids, values_load[plate_ids])

    # ramp the force linearly: (nsub, nnodes, 2)
    forces = np.array([values_load[plate_ids] * f for f in np.linspace(0.0, 1.0, 6)])
    step = fem.Step(items=[solid, load], ramp={load: forces}, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate(verbose=verbose)

    u = field[0].values.copy()

    # support reaction (must equal -force by equilibrium)
    solid_fresh = fem.SolidBody(
        # fem.NeoHooke returns a non-zero first Piola stress at the undeformed
        # state in 2D (P(F=I) = 0.333 I): use the linear-elastic plane-strain
        # formulation for 2D verification benchmarks
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu),
        field=fem.Field(fem.RegionQuad(mesh), dim=2).as_container(),
    )
    solid_fresh.field[0].values[:] = u
    fs = solid_fresh.assemble.vector(field=solid_fresh.field).toarray().ravel()
    support_dofs = 2 * np.where(contact_zone)[0] + 1
    R = float(fs[support_dofs].sum())

    # plate sink u_y
    sink = float(np.mean(u[plate_ids, 1]))

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
    ax.set_title(f"half model, P/2 = {force}: sink = {sink:.4f}")
    fig.savefig("benchmarks/solid-body-contact/half-model-deformed.png", dpi=150)
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
    print(f"plate sink u_y: {res['sink']:.5f}")
    print(f"support reaction (=-P/2): {res['support_reaction']:.5f}")
