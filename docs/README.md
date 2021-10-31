# felupe

> Finite Element Analysis

[![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://madewithlove.now.sh/at?heart=true&colorB=%231f744f&text=Graz+%28Austria%29) [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black)

![FElupe](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/logo.svg)

FElupe is a Python 3.6+ finite element analysis package focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. Its name is a combination of FE (finite element) and the german word *Lupe* (magnifying glass) as a synonym for getting a little insight how a finite element analysis code looks like under the hood. The user code for defining the integral form of equilibrium equations as well as their linearizations over a region are kept as close as possible to the analytical expressions. FElupe is both written in Python and fast in execution times thanks to NumPy and (optional) Numba. No complicated installation, just pure Python. Another key feature is the easy and straightforward definition of mixed field formulations for the treatment of nearly incompressible material behavior. In combination with [matadi](https://github.com/adtzlr/matadi) isotropic hyperelastic material formulations are defined in terms of their strain energy density function - gradient (stress) and hessian (elasticity tensor) are evaluated with automatic differentiation. Several useful utilities are available, i.e. the calculation of reaction forces and moments on given boundaries. Finally, results are ready to be exported to VTK or XDMF files using [meshio](https://github.com/nschloe/meshio).

## Installation
Install Python, fire up a terminal and run `pip install felupe[all]`, where `[all]` installs all optional dependencies. By default, FElupe does not require `numba` and `sparse`. In order to make use of all features of FElupe, it is suggested to install all optional dependencies. For constitutive material definitions using Automatic Differentation please also install [matadi](https://github.com/adtzlr/matadi).

```shell
pip install felupe[all]
```

*optional:*
```shell
pip install matadi
```

## Get Started
Explore the power of FElupe with a [simple hyperelastic example](quickstart.md). Other examples are located in the [examples](examples.md) section.

## License
FElupe - finite element analysis (C) 2021 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.