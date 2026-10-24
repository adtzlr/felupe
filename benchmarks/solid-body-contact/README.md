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
python benchmarks/solid-body-contact/bench_half_model_2d.py
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

## Benchmark B: half-model analysis with displacement control
(`bench_half_model_2d.py`)

The reference problem itself (see `setup-b.png`). The slide labels the
two-lobe shape a "1/4 model", but the load P/2 (half of the total P)
shows that it is a 1/2 model: the right half of the two-cylinder stack,
cut by the symmetry plane x = 0. The flat edge AB in the figure is the
DEFORMED arc pressed onto the rigid plate (uniform-u_y tie).

Constraints (no contact algorithm required — the trick of the original
Code_Aster model): the cylinder-cylinder contact plane stays flat by
symmetry, so the arcs adjacent to O get prescribed `u_y = 0`; likewise
the lower arc pressed onto the floor gets `u_y = 0`; the wall has
`u_x = 0`; the plate zone of the upper arc gets the uniform prescribed
`u_y` ramp (displacement control instead of the force `P/2`).

Result (approach 0.05, E = 100, plane strain): plate reaction
P = -0.200 per unit thickness; the lobes squeeze and bulge sideways as
sketched in the reference figure.

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
