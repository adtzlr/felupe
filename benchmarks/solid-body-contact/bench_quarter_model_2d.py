"""
Quarter-model analysis of the two-cylinder problem (reference figure),
with displacement control.

Geometry (per the analysis-conditions figure):
  upper lobe: quarter disk, center B(0, 1), radius 1  (arc A -> O)
  lower lobe: quarter disk, center D(0, -1), radius 1 (arc O -> C)
  the lobes share the single node O = (0, 0): the cylinder-cylinder
  contact plane is a symmetry plane, and because both arcs curve AWAY
  from the tangent plane, geometric interpenetration is impossible --
  no contact algorithm is required (this is the trick of the original
  Code_Aster model).

Boundary conditions (figure):
  OB, OD (x = 0): u_x = 0                     (symmetry wall)
  AB (y = 1):     u_y uniform, prescribed     (rigid plate, displacement control;
                                               the reaction at B is the P/2)
  CD (y = -1):    u_y = 0                     (rollers)
"""

import numpy as np
import felupe as fem


def quarter_model(n=6, E=100.0, nu=0.3):
    upper = fem.Circle(radius=1.0, centerpoint=[0, 1.0], sections=[270], n=n)
    lower = fem.Circle(radius=1.0, centerpoint=[0, -1.0], sections=[0], n=n)
    container = fem.MeshContainer([upper, lower], merge=True)
    mesh = container.stack()
    region = fem.RegionQuad(mesh)
    field = fem.Field(region, dim=2).as_container()
    solid = fem.SolidBody(
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu), field=field
    )
    return mesh, field, solid


def run(approach=0.05, n=6, E=100.0, nu=0.3, verbose=False):
    mesh, field, solid = quarter_model(n=n, E=E, nu=nu)

    boundaries = dict(
        wall=fem.Boundary(field[0], mask=mesh.points[:, 0] == 0, skip=(False, True)),
        plate=fem.Boundary(
            field[0],
            mask=(mesh.points[:, 1] == 1),
            skip=(True, False),
        ),
        support=fem.Boundary(field[0], mask=mesh.points[:, 1] == -1, skip=(True, False)),
    )
    move = fem.math.linsteps([0, -approach], num=5)

    step = fem.Step(items=[solid], ramp={boundaries["plate"]: move}, boundaries=boundaries)
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
