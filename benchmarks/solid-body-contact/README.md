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

## Benchmark B: 1/4-model analysis with force control
(`bench_quarter_model_2d.py`)

The reference problem itself (see `setup-b.png`): the 1/4 model — two
quarter-disk lobes (radius 1, centers (0, +/-1)) joined at the single
shared node O = (0, 0). The lobes interact only through O; both arcs
curve AWAY from the tangent plane, so geometric interpenetration is
impossible and no contact algorithm is required (the trick of the
original Code_Aster model).

Force control: the nodal force P/2 is applied at node B on the rigid
plate tie (a multi-point constraint linking the AB edge to B); the
wall OB/OD has u_x = 0; the supported edge CD has u_y = 0.

Result (P/2 = 0.2, E = 100, plane strain): plate sink u_y(B) = -0.0171;
support reaction = +0.2 = -P/2 (exact equilibrium). The deformed arcs
separate from the tangent plane -- no interpenetration (see
`quarter-model-deformed.png`).
