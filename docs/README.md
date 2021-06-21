# felupe

> Finite Element Analysis

[![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://madewithlove.now.sh/at?heart=true&colorB=%231f744f&text=Graz+%28Austria%29) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black)

![FElupe](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/felupe_logo.svg)

FElupe is an open-source finite element package focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. Its name is a combination of FE (finite element) and the german word *Lupe* (magnifying glass) as a synonym for getting a little insight how a finite element analysis code looks like under the hood. The user code for defining the integral form of equilibrium equations as well as their linearizations over a region are kept as close as possible to the analytical expressions. FElupe is both written in Python and fast in execution times thanks to NumPy (and optional Numba). No complicated installation, just pure Python. Another key feature is the easy and straightforward definition of mixed field formulations for nearly incompressible material behavior. Several useful utilities are available, i.e. an incremental approach for the application of boundary conditions and subsequent solution of the nonlinear equilibrium equations or the calculation of forces and moments on boundaries. Finally, results are ready to be exported to VTK or XDMF files using meshio.

**Question**: *Okay, why not use existing projects like Fenics or scikit-fem?*
Fenics is great but way too complicated to install on Windows. Scikit-fem is close to perfection but slow (too slow) for my problems of interest in combination with hyperelastic material formulations. In fact the utilities of FElupe could also be wrapped around the core code of scikit-fem but in the end I decided to create a code base which works the way I think it should.

## Installation
Install Python, fire up a terminal and run `pip install felupe`; import FElupe as follows in your script.

```python
import felupe as fe
```

## License
FElupe - finite element analysis (C) 2021 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.