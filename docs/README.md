# felupe

> Finite Element Analysis

[![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://madewithlove.now.sh/at?heart=true&colorB=%231f744f&text=Graz+%28Austria%29) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black)

![FElupe](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/felupe_logo.svg)

FElupe is an open-source finite element package focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. Its name is a combination of FE (finite element) and the german word *Lupe* (magnifying glass) as a synonym for getting a little insight how a finite element analysis code looks like under the hood. The user code for defining the integral form of equilibrium equations as well as their linearizations over a region are kept as close as possible to the analytical expressions. FElupe is both written in Python and fast in execution times thanks to NumPy (and optional Numba). No complicated installation, just pure Python. Another key feature is the easy and straightforward definition of mixed field formulations for nearly incompressible material behavior. Several useful utilities are available, i.e. an incremental approach for the application of boundary conditions and subsequent solution of the nonlinear equilibrium equations or the calculation of forces and moments on boundaries. Finally, results are ready to be exported to VTK or XDMF files using meshio.

**Question**: *Okay, why not use existing projects like Fenics or scikit-fem?*
Fenics is great but way too complicated to install on Windows. Scikit-fem is close to perfection but was quite slow for my [problem of interest](https://github.com/kinnala/scikit-fem/issues/616) in combination with hyperelastic material formulations. In fact the utilities of FElupe could also be wrapped around the core code of scikit-fem but in the end I decided to create a code base which works the way I think it should. FElupe is still in its alpha stage and serves as my own finite elements code playground - be warned: module usage may still change in future versions. If one is interested in a comparison of the assembly - times of felupe and [scikit-fem](https://github.com/kinnala/scikit-fem/#Benchmark) both are nearly equal for the poisson problem.

## Installation
Install Python, fire up a terminal and run `pip install felupe`; import FElupe as follows in your script.

```python
import felupe as fe
```

## Performance
The 2d poisson problem from [here](https://github.com/adtzlr/felupe/blob/main/scripts/script_performance_poisson.py) is solved with 4-noded quadrilaterals (linear basis) on a AMD Ryzen 5 2400G / 16GB RAM. The table below contains time spent on both assembly and linear solve as a function of the degrees of freedom. Assembly times do not contain numba JIT (just-in-time) compilation times.

|   DOF   | Assembly | Linear solve |
| ------- | -------- | ------------ |
|    5000 |   0.1 s  |     0.2 s    |
|   10082 |   0.2 s  |     0.1 s    |
|   20000 |   0.4 s  |     0.1 s    |
|   49928 |   0.9 s  |     0.3 s    |
|  100352 |   1.9 s  |     0.6 s    |
|  199712 |   3.9 s  |     1.3 s    |
|  500000 |   9.0 s  |     3.5 s    |
|  999698 |  18.3 s  |     8.6 s    |

Another [more practical performance benchmark](https://github.com/adtzlr/felupe/blob/main/scripts/script_performance_neohooke.py) is shown for a quarter model of a rubber block under uniaxial compression. The material is described with an isotropic hyperelastic Neo-Hooke material in its so-called nearly-incompressible formulation. Eight-noded hexahedrons are used in combination with a displacement field only. The time for linear solve represents the time spent for the first Newton-Rhapson iteration. While the assembly process is still acceptable up to 200000 DOF, linear solve time rapidly increases for >100000 DOF. However, this could also be due to the nearly 100% RAM load on the local machine for 500000 DOF.

|   DOF   | Assembly | Linear solve |
| ------- | -------- | ------------ |
|    5184 |   0.2 s  |     0.5 s    |
|   10125 |   0.5 s  |     0.2 s    |
|   20577 |   0.9 s  |     0.6 s    |
|   52728 |   2.3 s  |     3.2 s    |
|   98304 |   4.8 s  |    10.0 s    |
|  206763 |  10.4 s  |    42.5 s    |
|  499125 |  28.1 s  |   313.1 s    |

## License
FElupe - finite element analysis (C) 2021 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.