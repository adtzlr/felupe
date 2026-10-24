"""
Benchmark C: two equal elastic cylinders (r1 = r2 = 0.5) with a proper
closest-point (Gauss-point-to-segment) contact search instead of the
all-Gauss-points pairing of the issue #919 prototype.

For each slave Gauss point on body 1's boundary:
  - find the closest point on body 2's boundary segments (deformed config),
  - gap g = (x_slave - x_proj) . n_master  (n_master = outward, g < 0 = penetration),
  - traction p = K * max(-g, 0), repulsive,
  - force transferred to the master segment nodes by the segment shape
    functions (standard node-to-segment transfer).

The contact residual is assembled explicitly; the tangent is a numeric
Jacobian over the involved dofs (cheap: only the interface nodes matter).

Expected vs. Hertz line contact: b = sqrt(4 F R* / (pi E*)), p0 = 2 F/(pi b),
R* = r/2 for equal radii, E* = E / (2 (1 - nu^2)) in plane strain.
"""

import numpy as np
import scipy.sparse as sp

import felupe as fem


def make_cylinders(n1=16, n2=32, r1=0.5, r2=0.5):
    disk1 = fem.Circle(radius=r1, centerpoint=[0, r1 + r2], n=n1)
    disk2 = fem.Circle(radius=r2, centerpoint=[0, 0.0], n=n2)
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


class ContactClosest:
    "Closest-point (Gauss-point-to-segment) frictionless penalty contact."

    def __init__(self, slave_region, master_region, stiffness=1.0, center=(0.0, 0.0)):
        self.field = fem.Field(slave_region, dim=2).as_container()
        self.other_field = fem.Field(master_region, dim=2).as_container()
        self.stiffness = stiffness
        self.assemble = AssembleLike(vector=self.vector, matrix=self.matrix)
        self.results = ResultsLike()

        # slave Gauss-point data
        reg = slave_region
        X = reg.mesh.points
        cells = reg.mesh.cells
        self.slave_nodes = np.unique(cells)
        self.X_gp = np.einsum("aqc,cak->qkc", reg.h, X[cells])  # (nq, 2, nc)
        # true area element per Gauss point: face length / quadrature points
        fc1 = reg.mesh.cells_faces
        L = np.linalg.norm(X[fc1[:, 1]] - X[fc1[:, 0]], axis=1)  # (nc,)
        nq_ = reg.quadrature.points.shape[0]
        self.da = np.tile(L / nq_, (nq_, 1))  # (nq, nc)
        self.h = reg.h  # (na, nq, nc) or (na, nq, 1)
        if self.h.ndim == 3 and self.h.shape[2] == 1:
            self.h = np.broadcast_to(
                self.h, (self.h.shape[0], self.h.shape[1], reg.mesh.ncells)
            )
        self.cells = cells

        # master segment data (line cells of the boundary faces)
        X2 = master_region.mesh.points
        seg = master_region.mesh.cells_faces  # (nseg, 2) global node ids
        self.seg = seg
        self.master_nodes = np.unique(seg)
        mid = X2[seg].mean(axis=1)
        nrm = mid - np.array(center)
        self.n_seg = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        self.X_seg_nodes = X2

    def extract(self, field=None):
        if field is not None:
            self.field.link(field)
            self.other_field.link(field)
        return self.field[0].values  # shared global values array (npoints, 2)

    def _forces(self, values):
        "Return (slave nodal forces, master nodal forces, diagnostics)."
        # slave GP positions and displacements
        u_gp = np.einsum("aqc,cak->qkc", self.h, values[self.cells])  # (nq,2,nc)
        x_gp = self.X_gp + u_gp

        # master segment node positions (deformed)
        xs = self.X_seg_nodes + values  # (npoints, 2)
        a = xs[self.seg[:, 0]]  # (nseg, 2)
        b = xs[self.seg[:, 1]]

        # closest point per slave gp (brute force over segments)
        nq, _, nc = u_gp.shape
        f_gp = np.zeros((nq, 2, nc))
        gap = np.zeros((nq, nc))
        # preallocate loop arrays
        for q in range(nq):
            for c in range(nc):
                x1 = x_gp[q, :, c]
                d = x1[None, :] - a  # to segment start
                ab = b - a
                L2 = (ab**2).sum(axis=1)
                L2[L2 == 0] = 1e-30
                t = np.clip((d * ab).sum(axis=1) / L2, 0.0, 1.0)
                proj = a + t[:, None] * ab
                diff = x1[None, :] - proj
                dist = np.linalg.norm(diff, axis=1)
                k = np.argmin(dist)
                # segment outward normal (fixed, from reference config)
                nvec = self.n_seg[k]
                g = diff[k] @ nvec
                gap[q, c] = g
                if g < 0:  # penetration
                    p = self.stiffness * (-g) * self.da[q, c]
                    f_gp[q, :, c] = p * nvec  # pushes slave out (+n_master)

        # distribute slave gp forces to slave nodes (shape functions)
        f_slave = np.zeros((len(self.slave_nodes), 2))
        idx = {n: i for i, n in enumerate(self.slave_nodes)}
        for a_ in range(self.h.shape[0]):
            for q in range(nq):
                for c in range(nc):
                    n_glob = self.cells[c, a_]
                    f_slave[idx[n_glob]] += self.h[a_, q, c] * f_gp[q, :, c]

        # transfer to master segment nodes (linear shape functions, t = t*)
        # recompute transfer weights cheaply: proportional to (1 - t), t
        f_master = np.zeros((len(self.master_nodes), 2))
        midx = {n: i for i, n in enumerate(self.master_nodes)}
        for q in range(nq):
            for c in range(nc):
                if gap[q, c] >= 0:
                    continue
                x1 = x_gp[q, :, c]
                d = x1[None, :] - a
                ab = b - a
                L2 = (ab**2).sum(axis=1)
                L2[L2 == 0] = 1e-30
                t = np.clip((d * ab).sum(axis=1) / L2, 0.0, 1.0)
                dist = np.linalg.norm(d - t[:, None] * ab, axis=1)
                k = np.argmin(dist)
                p = self.stiffness * (-gap[q, c]) * self.da[q, c]
                f = p * self.n_seg[k]
                na, nb = self.seg[k]
                f_master[midx[na]] -= (1 - t[k]) * f  # reaction: opposite side
                f_master[midx[nb]] -= t[k] * f

        diag = {
            "gap_min": gap.min(),
            "max_pressure": self.stiffness * max(-gap.min(), 0.0),
            "force": np.abs(f_gp.sum(axis=(0, 2))[0]),  # x-comp ~ 0 check
            "force_y": float(np.sum(f_gp[:, 1, :])),
        }
        return -f_slave, -f_master, diag  # residual = -applied contact force

    def vector(self, field=None, **kwargs):
        values = self.extract(field=field)
        f_slave, f_master, diag = self._forces(values)
        self.diag = diag
        n = values.shape[0]
        r = np.zeros((n, 2))
        r[self.slave_nodes] += f_slave
        r[self.master_nodes] += f_master
        return sp.csr_matrix(r.ravel()[:, None])

    def matrix(self, field=None, **kwargs):
        values = self.extract(field=field).copy()
        dofs = np.concatenate(
            [
                2 * self.slave_nodes,
                2 * self.slave_nodes + 1,
                2 * self.master_nodes,
                2 * self.master_nodes + 1,
            ]
        )
        rows, cols, vals = [], [], []
        h = 1e-6
        ntot = values.size
        for dof in dofs:
            up = values.copy()
            up.reshape(-1)[dof] += h
            um = values.copy()
            um.reshape(-1)[dof] -= h
            col = (self._vector_from(up) - self._vector_from(um)) / (2 * h)
            nz = np.nonzero(np.abs(col) > 1e-12)[0]
            for i in nz:
                rows.append(int(i))
                cols.append(int(dof))
                vals.append(col[i])
        return sp.csr_matrix((vals, (rows, cols)), shape=(ntot, ntot))

    def _vector_from(self, values):
        f_slave, f_master, _ = self._forces(values)
        n = values.shape[0]
        r = np.zeros((n, 2))
        r[self.slave_nodes] += f_slave
        r[self.master_nodes] += f_master
        return r.ravel()


class ResultsLike:
    def update_statevars(self):
        pass


class AssembleLike:
    def __init__(self, vector, matrix):
        self.vector = vector
        self.matrix = matrix
        self.multiplier = None


def run(K=100.0, approach=0.01, n1=16, n2=32, r=0.5, E=100.0, nu=0.3, verbose=False):
    disk1, disk2 = make_cylinders(n1=n1, n2=n2, r1=r, r2=r)
    container = fem.MeshContainer([disk1, disk2], merge=False)
    mesh = container.stack()
    region = fem.RegionQuad(mesh)
    field = fem.Field(region, dim=2).as_container()
    # generous search bands: full lower half arc of disk1, full upper of disk2
    edge1 = fem.RegionQuadBoundary(disk1, mask=disk1.y < r + 0.35)
    edge2 = fem.RegionQuadBoundary(disk2, mask=disk2.y > r - 0.35)
    edge1 = edge1.copy(mesh=edge1.mesh.copy(points=mesh.points))
    edge2 = edge2.copy(
        mesh=edge2.mesh.copy(points=mesh.points, cells=edge2.mesh.cells + disk1.npoints)
    )

    solid = fem.SolidBody(
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu), field=field
    )
    contact = ContactClosest(edge1, edge2, stiffness=K, center=(0.0, 0.0))
    # cells_faces kept the LOCAL point ids of disk2 after the remap above:
    # rebuild master segments in global ids and recompute the normals
    contact.seg = edge2.mesh.cells_faces + disk1.npoints
    contact.master_nodes = np.unique(contact.seg)
    mid = mesh.points[contact.seg].mean(axis=1)
    nrm = mid - np.array([0.0, 0.0])
    contact.n_seg = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)

    y_top = r + r + r
    y_bot = -r
    bounds = dict(
        fix_bot=fem.Boundary(field[0], mask=mesh.points[:, 1] < y_bot + 0.05 * r),
        fix_bot_x=fem.Boundary(
            field[0], mask=mesh.points[:, 1] < y_bot + 0.05 * r, skip=(False, True)
        ),
        move=fem.Boundary(
            field[0], mask=mesh.points[:, 1] > y_top - 0.05 * r, skip=(True, False)
        ),
        move_x=fem.Boundary(
            field[0], mask=mesh.points[:, 1] > y_top - 0.05 * r, skip=(False, True)
        ),
    )
    move = fem.math.linsteps([0, -approach], num=10)
    step = fem.Step(
        items=[solid, contact], ramp={bounds["move"]: move}, boundaries=bounds
    )
    fem.Job(steps=[step]).evaluate(verbose=verbose)

    # snapshot of the converged state (do not reuse item caches: they may
    # lag one Newton iteration behind the reported solution)
    u_snap = field[0].values.copy()

    # exact applied force: fresh assembly of the solid on the snapshot
    solid_fresh = fem.SolidBody(
        umat=fem.constitution.LinearElasticPlaneStrain(E=E, nu=nu),
        field=fem.Field(fem.RegionQuad(mesh), dim=2).as_container(),
    )
    solid_fresh.field[0].values[:] = u_snap
    fs = solid_fresh.assemble.vector(field=solid_fresh.field).toarray().ravel()
    top_dofs = 2 * bounds["move"].points + 1  # .points are already ids
    F = float(-fs[top_dofs].sum())

    Estar = E / (2 * (1 - nu**2))
    Rstar = 1 / (1 / r + 1 / r)
    b_hertz = np.sqrt(4 * F * Rstar / (np.pi * Estar))
    p0_hertz = 2 * F / (np.pi * b_hertz)

    return {
        "K": K,
        "F": F,
        "b_hertz": b_hertz,
        "p0_hertz": p0_hertz,
        "p0_num": contact.diag["max_pressure"],
        "gap_min": contact.diag["gap_min"],
    }


if __name__ == "__main__":
    print(
        f"{'K':>6} {'F':>10} {'b_hertz':>9} {'p0_hertz':>9} {'p0_num':>9} {'gap_min':>10}"
    )
    for K in [1000.0, 10000.0, 100000.0]:
        r = run(K=K)
        print(
            f"{r['K']:>6.0f} {r['F']:>10.4f} {r['b_hertz']:>9.4f} {r['p0_hertz']:>9.3f} "
            f"{r['p0_num']:>9.3f} {r['gap_min']:>10.2e}"
        )
