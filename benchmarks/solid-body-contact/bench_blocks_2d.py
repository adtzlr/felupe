"""
Verification of the perturbed-Lagrangian / penalty contact prototype
from felupe issue #919 (comment thread, Dec 2024) against an exact
constraint-based reference solution.

Model: two unit-square plane-strain blocks stacked in compression.
The frictionless contact solution equals a bonded (shared-node) reference
here, because the flat interface, identical blocks and symmetric BCs produce
zero interface sliding by construction.

Checks
------
1. Displacement error of the contact model w.r.t. the bonded reference
   (expect ~ 1/K convergence as contact stiffness K grows).
2. Local penetration (gap) at the active contact Gauss points.
3. Force equilibrium: sum of both contact residual vectors ~ 0.
"""

import numpy as np
import tensortrax as tr
import tensortrax.math as tm
from scipy.spatial import cKDTree

import felupe as fem
from felupe.mechanics._helpers import Assemble, Results


class SolidBodyContact:
    """Frictionless contact between two boundary regions.

    Active-set penalty form of the issue #919 prototype: the contact
    traction +K * <g>_+ * n * da is assembled for every pair of Gauss
    points of the two boundary regions (pairwise tiling).
    """

    def __init__(self, region, other_region, stiffness=1.0, flip=1.0):
        """flip: sign to obtain the OUTWARD normal of the first body's face
        from the region normals (RegionQuadBoundary normals point +y for
        horizontal faces, independent of face orientation)."""
        self.flip = flip

        # RegionBoundary.normals are NOT reliably outward: they follow the
        # face parametrization and flip between faces, including on curved
        # surfaces where a constant sign flip cannot repair them. Recompute
        # them geometrically: face tangent rotated by 90 degrees, sign fixed
        # against the cell interior.
        def outward_normals(reg):
            "Outward normals per GP: face tangent rotated 90 deg, sign fixed"
            "to point away from the cell interior."
            X = reg.mesh.points
            fc = reg.mesh.cells_faces  # (ncells, 2) line cells of the faces
            tang = X[fc[:, 1]] - X[fc[:, 0]]  # (ncells, 2)
            nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            xgp = np.einsum("aqc,cak->qkc", reg.h, X[reg.mesh.cells])  # (nq, 2, nc)
            center = X[reg.mesh.cells].mean(axis=1)  # (nc, 2)
            sgn = np.sign(np.einsum("qkc,ck->qc", xgp - center.T[None], nrm))
            return nrm[:, None, :] * sgn.T[:, :, None]  # (ncells, nq, 2)

        nrm1, nrm2 = outward_normals(region), outward_normals(other_region)

        # RegionQuadBoundary keeps whole quads as cells (faces only for
        # quadrature): pair every Gauss point of one edge with every Gauss
        # point of the other by tiling the cells; tile the face normals
        # consistently with the cell tiling.
        cells = np.tile(region.mesh.cells, (other_region.mesh.ncells, 1))
        other_cells = np.tile(other_region.mesh.cells, (1, region.mesh.ncells)).reshape(
            cells.shape
        )

        self.region = region.copy(mesh=region.mesh.copy(cells=cells))
        self.other_region = region.copy(mesh=region.mesh.copy(cells=other_cells))
        for reg in (self.region, self.other_region):
            faces = reg._init_faces()
            reg.dA, reg.dV, reg.normals = faces[0], faces[1], faces[2]
        # tile the outward normals to the pairwise cell order
        self.region.normals = np.transpose(
            np.tile(nrm1, (other_region.mesh.ncells, 1, 1)), (2, 1, 0)
        )  # (2, nq, ncells)
        self.other_region.normals = np.transpose(
            np.tile(nrm2, (region.mesh.ncells, 1, 1)), (2, 1, 0)
        )

        self.field = fem.Field(self.region, dim=2).as_container()
        self.other_field = fem.Field(self.other_region, dim=2).as_container()

        self.stiffness = stiffness
        self.assemble = Assemble(vector=self.vector, matrix=self.matrix)
        self.results = Results()

    def fun(self, F, u, u_ext, K, N):
        # The weak form must be repulsive: with outward normal n1 of body 1
        # and gap gn = (u1 - u2) . n1, the residual contribution is
        # +K * <gn>_+ * n1 * da. The issue #919 convention -K * gn * t is
        # only correct if the region normals point inward; audit per
        # element/face orientation.
        x = (
            F
            if isinstance(F, tr.Tensor)
            else (u if isinstance(u, tr.Tensor) else u_ext)
        )
        N = tm.array(self.flip * N, shape=(2,), like=x)  # outward normal of body 1
        t = tm.linalg.det(F) * tm.transpose(tm.linalg.inv(F)) @ N  # n da
        n = t / tm.sqrt(tm.sum(t**2, axis=0))
        un = (u - u_ext) @ n  # positive = approach / penetration
        gn = tm.if_else(un.x > 0, un, 0 * un)
        return K * gn * t  # repulsive: force on body 1 = -f_contact = -K gn n_out

    def extract(self, field=None):
        if field is not None:
            self.field.link(field)
            self.other_field.link(field)
        F = self.field.extract()[0]
        u = self.field.extract(grad=False)[0]
        u_ext = self.other_field.extract(grad=False)[0]
        return F, u, u_ext

    def vector(self, field=None, **kwargs):
        F, u, u_ext = self.extract(field=field)
        K, N = self.stiffness, self.field.region.normals
        return fem.IntegralForm(
            fun=[tr.function(self.fun, ntrax=2)(F, u, u_ext, K, N)],
            v=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
        ).assemble()

    def matrix(self, field=None, **kwargs):
        F, u, u_ext = self.extract(field=field)
        K, N = self.stiffness, self.field.region.normals
        return (
            fem.IntegralForm(
                fun=[tr.jacobian(self.fun, ntrax=2, wrt=1)(F, u, u_ext, K, N)],
                v=self.field,
                dV=self.field.region.dV,
                u=self.field,
                grad_v=[False],
                grad_u=[False],
            ).assemble()
            + fem.IntegralForm(
                fun=[tr.jacobian(self.fun, ntrax=2, wrt=2)(F, u, u_ext, K, N)],
                v=self.field,
                dV=self.field.region.dV,
                u=self.other_field,
                grad_v=[False],
                grad_u=[False],
            ).assemble()
            + fem.IntegralForm(
                fun=[tr.jacobian(self.fun, ntrax=2, wrt=0)(F, u, u_ext, K, N)],
                v=self.field,
                dV=self.field.region.dV,
                u=self.field,
                grad_v=[False],
                grad_u=[True],
            ).assemble()
        )

    def penetration(self, field=None):
        "Maximum local gap violation gn over all Gauss-point pairs."
        F, u, u_ext = self.extract(field=field)
        N = self.flip * self.field.region.normals
        # n da = det(F) F^-T N (2d), evaluated per Gauss point / pair
        Fm = np.moveaxis(F, [0, 1], [-2, -1])  # (2, npairs, 2, 2)
        detF = np.linalg.det(Fm)  # (2, npairs)
        invF = np.linalg.inv(Fm)  # (2, npairs, 2, 2)
        t = np.einsum(
            "qpab,bqp->aqp", detF[..., None, None] * invF, N
        )  # (2, nq, npairs)
        n = t / np.linalg.norm(t, axis=0, keepdims=True)
        un = np.einsum("i...,i...->...", u - u_ext, n)
        return un[un > 0].max() if np.any(un > 0) else 0.0


def bottom_block():
    return fem.Rectangle(a=(0, 0), b=(1, 1), n=6)


def top_block():
    return fem.Rectangle(a=(0, 1), b=(1, 2), n=6)


def run_contact(K):
    "Two blocks with the issue-#919 contact between them."
    block1, block2 = bottom_block(), top_block()
    container = fem.MeshContainer([block1, block2], merge=False)
    mesh = container.stack()

    region = fem.RegionQuad(mesh)
    field = fem.Field(region, dim=2).as_container()

    # contact faces: bottom edge of block 1, bottom edge of block 2.
    # MeshContainer meshes share one points array: point masks such as
    # mesh.y == 1 would select points of BOTH bodies -- build the boundary
    # regions on the separate body meshes and remap to global ids after.
    edge1 = fem.RegionQuadBoundary(block1, mask=block1.y == 1)
    edge2 = fem.RegionQuadBoundary(block2, mask=block2.y == 1)
    n1 = block1.npoints
    edge1 = edge1.copy(mesh=edge1.mesh.copy(points=mesh.points))
    edge2 = edge2.copy(
        mesh=edge2.mesh.copy(points=mesh.points, cells=edge2.mesh.cells + n1)
    )
    assert edge1.mesh.ncells == 5 and edge2.mesh.ncells == 5

    umat = fem.constitution.LinearElasticPlaneStrain(E=100.0, nu=0.3)
    solid = fem.SolidBody(umat=umat, field=field)
    contact = SolidBodyContact(edge1, edge2, stiffness=K, flip=1.0)
    contact2 = SolidBodyContact(edge2, edge1, stiffness=K, flip=1.0)

    bounds = dict(
        fixed=fem.Boundary(field[0], fy=0, skip=(True, False)),
        fixed_x=fem.Boundary(field[0], fx=0, skip=(False, True)),
        move=fem.Boundary(field[0], fy=2, skip=(True, False)),
        move_x=fem.Boundary(field[0], fx=2, skip=(False, True)),
    )
    move = fem.math.linsteps([0, -0.1], num=5)
    step = fem.Step(
        items=[solid, contact, contact2], ramp={bounds["move"]: move}, boundaries=bounds
    )
    job = fem.Job(steps=[step])
    job.evaluate()

    err = rel_err(field[0].values, mesh.points)
    return {
        "K": K,
        "rel_err_u": err,
        "max_penetration": contact.penetration(field=field),
        "residual_sum": np.abs(
            contact.assemble.vector(field=field).toarray()
            + contact2.assemble.vector(field=field).toarray()
        ).max(),
    }


def run_reference():
    "Bonded reference: coincident interface nodes merged, no contact."
    container = fem.MeshContainer([bottom_block(), top_block()], merge=True)
    mesh = container.stack()
    region = fem.RegionQuad(mesh)
    field = fem.Field(region, dim=2).as_container()

    solid = fem.SolidBody(
        # fem.NeoHooke returns a non-zero first Piola stress at the undeformed
        # state in 2D (P(F=I) = 0.333 I): use the linear-elastic plane-strain
        # formulation for 2D verification benchmarks
        umat=fem.constitution.LinearElasticPlaneStrain(E=100.0, nu=0.3),
        field=field,
    )
    bounds = dict(
        fixed=fem.Boundary(field[0], fy=0, skip=(True, False)),
        fixed_x=fem.Boundary(field[0], fx=0, skip=(False, True)),
        move=fem.Boundary(field[0], fy=2, skip=(True, False)),
        move_x=fem.Boundary(field[0], fx=2, skip=(False, True)),
    )
    move = fem.math.linsteps([0, -0.1], num=5)
    step = fem.Step(items=[solid], ramp={bounds["move"]: move}, boundaries=bounds)
    fem.Job(steps=[step]).evaluate()
    return field[0].values.copy(), container.stack().points.copy()


if __name__ == "__main__":
    u_ref, xy_ref = run_reference()
    tree = cKDTree(xy_ref)

    def rel_err(u, xy):
        "Relative L2 error of u mapped onto the reference mesh by nearest node."
        _, idx = tree.query(xy)
        return np.linalg.norm(u - u_ref[idx]) / np.linalg.norm(u_ref)

    print(f"reference: max |u| = {np.abs(u_ref).max():.6f}")

    print(f"{'K':>6} {'rel_err_u':>12} {'max_penetration':>16} {'residual_sum':>14}")
    for K in [10.0, 100.0, 1000.0]:
        r = run_contact(K)
        print(
            f"{r['K']:>6.0f} {r['rel_err_u']:>12.3e} {r['max_penetration']:>16.3e} "
            f"{r['residual_sum']:>14.3e}"
        )
