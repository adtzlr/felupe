"""
Benchmark B: contact between two equal elastic cylinders (r1 = r2 = 0.5),
plane strain, frictionless, using the issue-#919 SolidBodyContact prototype
(benchmark A's verified form).

Checks
------
1. Contact half-width b and peak pressure p0 vs the Hertz line-contact
   solution with the numerically measured resultant force.
2. Penetration scale vs contact stiffness K.

E = 100, nu = 0.3 (both bodies) -> E* = E / (2 (1 - nu^2)),
1/R* = 1/r1 + 1/r2, b = sqrt(4 F' R* / (pi E*)), p0 = 2 F' / (pi b).
"""

import sys
from pathlib import Path

import numpy as np

import felupe as fem

sys.path.insert(0, str(Path(__file__).parent))
from bench_blocks_2d import SolidBodyContact  # verified in benchmark A


def make_cylinders(n1=6, n2=12, r1=0.5, r2=0.5):
    disk1 = fem.Circle(radius=r1, centerpoint=[0, r1 + r2], n=n1)
    disk2 = fem.Circle(radius=r2, centerpoint=[0, 0], n=n2)
    return disk1, disk2


def contact_edges(disk1, disk2, mesh, band=0.12, r1=0.5, r2=0.5):
    "Boundary regions on the cells adjacent to the respective contact pole."
    edge1 = fem.RegionQuadBoundary(disk1, mask=disk1.y < r2 + band)
    edge2 = fem.RegionQuadBoundary(disk2, mask=disk2.y > r2 - band)
    edge1 = edge1.copy(mesh=edge1.mesh.copy(points=mesh.points))
    edge2 = edge2.copy(
        mesh=edge2.mesh.copy(points=mesh.points, cells=edge2.mesh.cells + disk1.npoints)
    )
    return edge1, edge2


def run(
    K=100.0, approach=0.01, n1=6, n2=12, r1=0.5, r2=0.5, E=100.0, nu=0.3, verbose=False
):
    disk1, disk2 = make_cylinders(n1=n1, n2=n2, r1=r1, r2=r2)
    container = fem.MeshContainer([disk1, disk2], merge=False)
    mesh = container.stack()
    region = fem.RegionQuad(mesh)
    field = fem.Field(region, dim=2).as_container()
    edge1, edge2 = contact_edges(disk1, disk2, mesh, r1=r1, r2=r2)

    solid = fem.SolidBody(
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu), field=field
    )
    contact = SolidBodyContact(edge1, edge2, stiffness=K, flip=1.0)
    contact2 = SolidBodyContact(edge2, edge1, stiffness=K, flip=1.0)

    y_top = r1 + r2 + r1  # 1.5
    y_bot = -r2
    bounds = dict(
        fix_bot=fem.Boundary(field[0], mask=mesh.points[:, 1] < y_bot + 0.05 * r2),
        move=fem.Boundary(
            field[0],
            mask=(mesh.points[:, 1] > y_top - 0.05 * r1),
            skip=(True, False),
        ),
        fix_top_x=fem.Boundary(
            field[0], mask=(mesh.points[:, 1] > y_top - 0.05 * r1), skip=(False, True)
        ),
        fix_pole_x=fem.Boundary(
            field[0], mask=mesh.points[:, 1] < y_bot + 0.05 * r2, skip=(False, True)
        ),
    )
    move = fem.math.linsteps([0, -approach], num=5)
    step = fem.Step(
        items=[solid, contact, contact2], ramp={bounds["move"]: move}, boundaries=bounds
    )
    fem.Job(steps=[step]).evaluate(verbose=verbose)

    # active contact pairs: locate them, get patch half-width and peak pressure
    Fc, u, u_ext = contact.extract(field=field)
    N = contact.field.region.normals * contact.flip
    Fm = np.moveaxis(Fc, [0, 1], [-2, -1])
    t = np.einsum(
        "qpab,bqp->aqp", np.linalg.det(Fm)[..., None, None] * np.linalg.inv(Fm), N
    )
    n = t / np.linalg.norm(t, axis=0, keepdims=True)
    un = np.einsum("i...,i...->...", u - u_ext, n)
    da = np.linalg.norm(t, axis=0)
    F = float(np.sum(K * np.maximum(un, 0) * da * np.abs(n[0])))

    # Hertz line-contact reference with the measured resultant force
    Estar = E / (2 * (1 - nu**2))
    Rstar = 1 / (1 / r1 + 1 / r2)
    b_hertz = np.sqrt(4 * F * Rstar / (np.pi * Estar))
    p0_hertz = 2 * F / (np.pi * b_hertz)

    pressure = K * np.maximum(un, 0)
    nc1 = edge1.mesh.ncells
    xc1 = mesh.points[edge1.mesh.cells].mean(axis=1)[:, 0]
    xc2 = mesh.points[edge2.mesh.cells].mean(axis=1)[:, 0]
    active = un > 0
    qs, ps = np.where(active)
    ii, jj = ps % nc1, ps // nc1
    b_num = np.abs(np.concatenate([xc1[ii], xc2[jj]])).max() if len(ps) else 0.0

    return {
        "K": K,
        "F": F,
        "b_hertz": b_hertz,
        "b_num": b_num,
        "p0_hertz": p0_hertz,
        "p0_num": pressure[active].max() if active.any() else 0.0,
        "max_penetration": contact.penetration(field=field),
    }


if __name__ == "__main__":
    print(
        f"{'K':>6} {'F':>10} {'b_hertz':>9} {'b_num':>9} {'p0_hertz':>9} {'p0_num':>9} {'penetr':>9}"
    )
    for K in [100.0, 1000.0]:
        r = run(K=K)
        print(
            f"{r['K']:>6.0f} {r['F']:>10.4f} {r['b_hertz']:>9.4f} {r['b_num']:>9.4f} "
            f"{r['p0_hertz']:>9.3f} {r['p0_num']:>9.3f} {r['max_penetration']:>9.2e}"
        )
