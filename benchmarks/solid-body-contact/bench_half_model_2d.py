"""
Half-model analysis of the two-cylinder problem (reference figure),
with displacement control.

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
  arc B-A (upper): u_y uniform, prescribed    (rigid plate tie, displacement
                                               control; reaction at B = P/2)
  arc C-D (lower): u_y = 0                    (rollers)
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


def run(approach=0.05, n=6, E=100.0, nu=0.3, verbose=False):
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

    # plate zone: upper arc of the upper disk (B .. A): uniform u_y = ramp
    # contact plane (A .. O and O .. C): u_y = 0 (symmetry: the plane stays
    # flat -- this replaces the cylinder-cylinder contact algorithm)
    # support (C .. D): u_y = 0 (rollers / floor)
    plate_zone = upper_arc & (mesh.points[:, 1] >= 0.5 - 1e-9)
    contact_zone = (upper_arc & ~plate_zone) | lower_arc

    boundaries = dict(
        wall=fem.Boundary(field[0], mask=mesh.points[:, 0] == 0, skip=(False, True)),
        plate=fem.Boundary(field[0], mask=plate_zone, skip=(True, False)),
        contact=fem.Boundary(field[0], mask=contact_zone, skip=(True, False)),
    )
    move = fem.math.linsteps([0, -approach], num=5)

    step = fem.Step(
        items=[solid], ramp={boundaries["plate"]: move}, boundaries=boundaries
    )
    fem.Job(steps=[step]).evaluate(verbose=verbose)

    u = field[0].values.copy()

    # reaction at the plate edge (the P/2 of the force-controlled original)
    solid_fresh = fem.SolidBody(
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu),
        field=fem.Field(fem.RegionQuad(mesh), dim=2).as_container(),
    )
    solid_fresh.field[0].values[:] = u
    fs = solid_fresh.assemble.vector(field=solid_fresh.field).toarray().ravel()
    plate_dofs = 2 * np.where(mesh.points[:, 1] == 1)[0] + 1
    P = float(fs[plate_dofs].sum())  # vertical reaction (per unit thickness)

    return {"mesh_points": mesh.npoints, "u_max": np.abs(u).max(), "P": P}


if __name__ == "__main__":
    res = run()
    print(f"points: {res['mesh_points']}, max|u|: {res['u_max']:.5f}")
    print(f"plate reaction P (per unit thickness): {res['P']:.5f}")
