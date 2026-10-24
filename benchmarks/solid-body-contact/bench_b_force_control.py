"""
Benchmark B: force control on the SAME model, mesh and contact
algorithm as benchmark C (bench_cylinders_2d_closest.py).

Raw force-control Newton (force as the independent quantity) is unstable
for contact: the active-set kink at g = 0 makes the iteration oscillate.
Instead, the force is imposed through a secant (pseudo-arclength-style)
outer loop on the stable displacement-control solver of benchmark C:
the prescribed approach is iterated until the measured reaction equals
the target force P/2. The result is the plate sink u_y; the support
reaction equals -P/2 by equilibrium.
"""

import bench_cylinders_2d_closest as bench_c


def run(force=0.165, K=1000.0, tol=1e-4, verbose=False):
    "Force control: iterate the approach until the reaction equals `force`."
    d0, F0 = 0.5 * force, bench_c.run(K=K, approach=0.5 * force)["F"]
    d1, F1 = force, bench_c.run(K=K, approach=force)["F"]
    dn, Fn = d1, F1
    for it in range(20):
        if abs(Fn - force) < tol * force:
            break
        if F1 == F0:
            break
        dn = dn - (Fn - force) * (dn - d0) / (Fn - F0)
        Fn = bench_c.run(K=K, approach=dn)["F"]
        d0, F0 = d1, F1
        d1, F1 = dn, Fn
        if verbose:
            print(f"  secant {it}: approach = {dn:.6f}, F = {Fn:.6f}")
    return {"force": force, "K": K, "sink": dn, "reaction": Fn}


if __name__ == "__main__":
    # same load level as benchmark C (its measured reaction at K = 1e5);
    # the sink must approach the prescribed approach of benchmark C
    # (0.01) as the active-set solution matches
    for K in [1000.0, 10000.0, 100000.0]:
        res = run(force=0.165, K=K)
        print(
            f"K = {res['K']:>7.0f}: sink = {res['sink']:.5f}, "
            f"reaction = {res['reaction']:.4f} (target 0.165)"
        )
