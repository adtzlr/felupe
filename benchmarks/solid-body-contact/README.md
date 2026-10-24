# Verification of the `SolidBodyContact` prototype from issue #919

Verified against FElupe `main` (af64eeb5). The scripts handle both the old
(3-value) and the new (4-value) `RegionBoundary._init_faces()` return
signature, and import `LinearElasticPlaneStrain` from
`felupe.constitution` (no longer exported at the top level).

Two self-contained 2D benchmarks that verify the perturbed-Lagrangian /
active-set penalty contact prototype sketched in
[issue #919](https://github.com/adtzlr/felupe/issues/919) (Dec 2024 comment
thread), adapted from the 3D hexahedron code in the issue to 2D quads.

Run with the repository venv:

```sh
python benchmarks/solid-body-contact/bench_blocks_2d.py
python benchmarks/solid-body-contact/bench_cylinders_2d.py
```

## Benchmark A: two blocks in compression (`bench_blocks_2d.py`)

Two unit-square plane-strain blocks pressed together by a prescribed
displacement. The frictionless contact solution equals a bonded
(shared-node) reference here, because the flat interface, identical blocks
and symmetric BCs produce zero interface sliding by construction.

Results (contact stiffness `K`, relative L2 error of the displacement field
w.r.t. the bonded reference, maximum local gap violation):

| K | rel_err_u | max_penetration |
|---:|---:|---:|
| 10 | 2.96e-1 | 5.2e-2 |
| 100 | 5.5e-2 | 9.7e-3 |
| 1000 | 6.0e-3 | 1.1e-3 |

Conclusions:

- With corrected signs (see below) the prototype converges to the exact
  constraint solution with the expected penalty behaviour
  (`penetration ~ 1/K`, displacement error ~ 1/K).
- Newton fails to converge at K = 10000 (active-set chatter, no
  stabilization / line search in the prototype).

## Benchmark B: two cylinders, Hertz line contact (`bench_cylinders_2d.py`)

Disks with r1 = r2 = 0.5 (equal diameters, as in the reference problem), E = 100, nu = 0.3, plane strain, prescribed
approach on the top arc. Compares the contact patch against the Hertz
line-contact solution `b = sqrt(4 F R* / (pi E*))`, `p0 = 2 F / (pi b)`
with the numerically measured resultant force F.

Results:

| K | F | b_Hertz | b_num | p0_Hertz | p0_num |
|---:|---:|---:|---:|---:|---:|
| 100 | 5.46 | 0.178 | 0.280 | 19.5 | 0.14 |
| 1000 | 5.82 | 0.184 | 0.280 | 20.2 | 0.46 |

The prototype reproduces the total contact force (converging with K), but
the patch metrics are non-physical for curved surfaces:

1. Every Gauss point of edge 1 is paired with every Gauss point of edge 2
   (pairwise tiling, as in the issue prototype). This multiplies the
   effective contact stiffness by the number of partners, so the local
   penetration and per-point pressure are far below the physical values
   (p0_num is ~40x below the Hertz value), while the total force is right.
2. The active set extends over the whole candidate band (b_num = band
   width) instead of the Hertz patch, because geometric overlap of the
   undeformed arcs activates pairs far outside the true contact zone.
3. Cost is O(n1 * n2) per assembly. The issue author noted this overhead
   himself in the thread.

Conclusion: for the two-cylinder problem the prototype as sketched is not
usable without (a) a proper closest-point / active-set search instead of
all-pairs pairing, and (b) normalization of the traction by the partner
count. For flat interfaces with matching discretizations (benchmark A) the
formulation is correct.

## Sign and orientation findings (important for a 2D port)

1. The weak form must be repulsive: with outward normal `n1` of body 1 and
   gap `gn = (u1 - u2) . n1`, the residual contribution is
   `+K * <gn>_+ * n1 * da`. The sign convention `-K * gn * t` in the issue
   prototype is only correct if the region normals point inward; it must be
   audited for each element/face orientation.
2. `RegionQuadBoundary.normals` (and the 3D equivalent) are NOT reliably
   outward: they follow the face parametrization and flip orientation
   between faces, including on curved surfaces where a constant sign flip
   cannot repair them. Outward normals must be recomputed geometrically:
   rotate the face tangent by 90 degrees and fix the sign against the cell
   interior (see `outward_normals()` in `bench_blocks_2d.py`).
3. `RegionQuadBoundary` keeps whole quads as cells (faces only for
   quadrature). Pairing Gauss points of two boundary regions therefore
   needs the tiled-cells trick from the issue; face normals must be tiled
   consistently with the cell tiling.
4. `MeshContainer` meshes share one points array: point masks such as
   `mesh.y == 1` select points of both bodies. Build boundary regions on
   the separate body meshes and remap to global ids afterwards.
5. `fem.NeoHooke` in this version returns a non-zero first Piola stress at
   the undeformed state in 2D (`P(F=I) = 0.333 I`) — use
   `fem.LinearElasticPlaneStrain` for 2D verification benchmarks.

## Benchmark C: closest-point (Gauss-point-to-segment) contact
(`bench_cylinders_2d_closest.py`)

Replacement for the all-pairs pairing: each slave Gauss point is projected
onto the closest master boundary segment (deformed configuration), the gap
is measured along the master outward normal, and the force is transferred
to the master segment nodes by linear shape functions. The tangent is a
numeric Jacobian over the interface dofs.

Results (equal radii, same load as benchmark B):

| K | F | b_Hertz | p0_Hertz | p0_num | min gap |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.045 | 0.016 | 1.77 | 0.57 | -5.7e-3 |
| 1000 | 0.124 | 0.027 | 2.95 | 1.58 | -1.6e-3 |
| 10000 | 0.150 | 0.030 | 3.24 | 1.92 | -1.9e-4 |

Findings:

1. The resultant force converges to the bonded-reference value
   (F = 0.150 vs 0.152 at K = 10000, 1 % error) and the penetration
   scales as ~ 1/K — the penalty contact is consistent.
2. p0_num approaches the Hertz value but the comparison is mesh-limited:
   the Hertz patch (b ~ 0.03) is smaller than the element size (~0.035).
3. Newton fails at K = 100000 (active-set chatter), as in benchmark A.

Implementation notes / pitfalls hit during development:

- `RegionBoundary.mesh.cells_faces` keeps the LOCAL point ids of the body
  mesh; after remapping `mesh.cells` to global ids, the segments must be
  remapped (and the node list recomputed) separately.
- `region.dV` on a boundary region is not the physical face area per Gauss
  point; use face length / number of quadrature points for area-consistent
  penalty forces, otherwise the contact stiffness depends on the mesh.
- After `Job.evaluate`, item-level cached assemblies may lag the reported
  solution; rebuild a fresh `SolidBody` on a copy of the converged values
  for post-processing (reactions etc.).
