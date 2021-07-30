# Benchmark 1 - Poisson problem
The 2d poisson problem from [here](https://github.com/adtzlr/felupe/blob/main/scripts/script_performance_poisson.py) is solved with 4-noded quadrilaterals (linear basis) on a AMD Ryzen 5 2400G / 16GB RAM. The table below contains time spent on both assembly and linear solve (PyPardiso) as a function of the degrees of freedom. Assembly times do not contain numba JIT (just-in-time) compilation times. Note that felupe's default solver taken from SciPy (SuperLU) may be slower.

Performance without Numba (`b.assemble()`, `A.assemble()`)

|   DOF   | Assembly | Linear solve |
| ------- | -------- | ------------ |
|    5041 |   0.1 s  |     0.0 s    |
|   10000 |   0.3 s  |     0.0 s    |
|   19881 |   0.5 s  |     0.1 s    |
|   50176 |   1.3 s  |     0.2 s    |
|   99856 |   2.6 s  |     0.5 s    |
|  199809 |   5.3 s  |     1.0 s    |
|  499849 |  13.5 s  |     3.1 s    |
| 1000000 |  26.9 s  |     7.1 s    |


and performance with Numba (`b.assemble(parallel=True)`, `A.assemble(parallel=True)`):

|   DOF   | Assembly | Linear solve |
| ------- | -------- | ------------ |
|    5041 |   0.1 s  |     0.1 s    |
|   10000 |   0.3 s  |     0.0 s    |
|   19881 |   0.5 s  |     0.1 s    |
|   50176 |   1.2 s  |     0.2 s    |
|   99856 |   2.4 s  |     0.5 s    |
|  199809 |   5.0 s  |     1.0 s    |
|  499849 |  12.3 s  |     3.1 s    |
| 1000000 |  25.9 s  |     7.3 s    |

# Benchmark 2 - Hyperelasticity

Another [more practical performance benchmark](https://github.com/adtzlr/felupe/blob/main/scripts/script_performance_neohooke.py) is shown for a quarter model of a rubber block under uniaxial compression. The material is described with an isotropic hyperelastic Neo-Hooke material in its so-called nearly-incompressible formulation. Eight-noded hexahedrons are used in combination with a displacement field only. The time for linear solve (PyPardiso) represents the time spent for the first Newton-Rhapson iteration. While the assembly process is still acceptable up to 200000 DOF, linear solve time rapidly increases for >100000 DOF. However, this could also be due to the nearly 100% RAM load on the local machine for 500000 DOF.

|   DOF   | Assembly | Linear solve |
| ------- | -------- | ------------ |
|    5184 |   0.2 s  |     0.5 s    |
|   10125 |   0.5 s  |     0.2 s    |
|   20577 |   0.9 s  |     0.6 s    |
|   52728 |   2.3 s  |     3.2 s    |
|   98304 |   4.8 s  |    10.0 s    |
|  206763 |  10.4 s  |    42.5 s    |
|  499125 |  28.1 s  |   313.1 s    |

# Benchmark 3 - Mixed-Field Hyperelasticity

And now the big news. Given a hyperelastic problem with a Neo-Hookean solid in combination with a $(\bm{u},p,J)$ mixed-field-formulation applied on a meshed unit cube out of eight-noded hexahedrons with 9x9x9=729 nodes (=2187 DOF) for nearly-incompressibility takes about 0.3s in felupe. Again, please note that felupe timings are without Numba JIT (just-in-time) compilation times and the whole Newton-Rhapson iteration contains the time spent on linear solve (PyPardiso was used in these examples). For a detailed comparison see the table below:

|   Nodes per axis (total) | displacement - DOF | felupe |
| ------------------------ | ------------------ | ------ |
|         3    (27)        |         81         |  0.02s |
|         5   (125)        |        375         |  0.05s |
|         9   (729)        |       2187         |   0.3s |
|        17  (4913)        |      14739         |     2s |
|        33 (35937)        |     107811         |    37s |