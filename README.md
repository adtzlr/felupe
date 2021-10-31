# FElupe - Finite Element Analysis

[![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/gh/adtzlr/felupe/branch/main/graph/badge.svg?token=J2QP6Y6LVH)](https://codecov.io/gh/adtzlr/felupe) [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![GitHub Repo stars](https://img.shields.io/github/stars/adtzlr/felupe?logo=github) ![PyPI - Downloads](https://img.shields.io/pypi/dm/felupe)

<img src="https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/logo.svg" width="220px"/>

FElupe is a Python 3.6+ finite element analysis package focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. Its name is a combination of FE (finite element) and the german word *Lupe* (magnifying glass) as a synonym for getting a little insight how a finite element analysis code looks like under the hood.

# Installation
Install Python, fire up a terminal and run

```shell
pip install felupe[all]
```

where `[all]` installs all optional dependencies. By default, FElupe does not require `numba` and `sparse`. In order to make use of all features of FElupe, it is suggested to install all optional dependencies.

# Hello, FElupe!
A quarter model of a solid cube with hyperelastic material behavior is subjected to a uniaxial elongation applied at a clamped end-face. This involves the creation of a mesh, a region and a displacement field. Furthermore, the boundary conditions are created by a template for a uniaxial loadcase. The material behavior is defined through a Neo-Hookean material formulation. By assembling both linear and bilinear forms, the internal force vector and the tangent stiffness matrix are generated. Finally, the linear solution of the incremental displacements is calculated (the iterative Newton-Rhapson procedure is not shown here). For more details, have a look at the documentation.

```python
import felupe as fe

# create a hexahedron-region on a cube
region = fe.RegionHexahedron(fe.Cube(n=11))

# add a displacement field and apply a uniaxial elongation on the cube
displacement = fe.Field(region, dim=3)
boundaries, dof0, dof1, ext0 = fe.dof.uniaxial(displacement, move=0.2, clamped=True)

# deformation gradient
F = displacement.extract(grad=True, sym=False, add_identity=True)

# define the constitutive material behavior
umat = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)
P  = umat.gradient(F)
A4 = umat.hessian(F)
    
# force residuals from assembly of equilibrium (weak form)
r = fe.IntegralForm(
    P, v=displacement, dV=region.dV, grad_v=True
).assemble().toarray()[:,0]
    
# tangent stiffness matrix from (parallel) assembly of linearized equilibrium
K = fe.IntegralForm(
    A4, v=displacement, dV=region.dV, u=displacement, grad_v=True, grad_u=True,
).assemble(parallel=True)

# solution: partition, solve linear system and update field values
system = fe.solve.partition(displacement, K, dof1, dof0, r)
displacement += fe.solve.solve(*system, ext0)

# export results
fe.tools.save(region, displacement, filename="result.vtk")
```

<img src="https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/readme.png" width="600px"/>

# Documentation
The documentation is located [here](https://adtzlr.github.io/felupe).

# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2021-10-31

### Added
- Add template regions, i.e. a region with a `Hexahedron()` element and a quadrature scheme `GaussLegendre(order=1, dim=3)` as `RegionHexahedron`, etc.
- Add biaxial and planar loadcases (like uniaxial).
- Add a minimal README-example (Hello FElupe!).

### Changed
- Deactivate clamped boundary (`clamped=False`) as default option for uniaxial loading `dof.uniaxial`

## [1.1.0] - 2021-10-30

### Added
- Add inverse quadrature method `quadrature.inv()` for Gauss-Legendre schemes.
- Add discontinous representation of a mesh as mesh method `mesh.as_discontinous()`.
- Add `tools.project()` to project (and average) values at quadrature points to mesh points.

### Changed
- Removed `quadpy` dependency and use built-in polynomials of `numpy` for Gauss-Legendre calculation.

### Fixed
- Fix typo in first shear component of `math.tovoigt()` function.
- Fix wrong stress projection in `tools.topoints()` due to different quadrature and cell ordering.

## [1.0.1] - 2021-10-19

### Fixed
- Fix import of dof-module if `sparse` is not installed.

## [1.0.0] - 2021-10-19

### Added
- Start using a Changelog.
- Added docstrings for essential classes, methods and functions.
- Add array with point locations for all elements.

### Changed
- Rename element methods (from `basis` to `function` and from `basisprime` to `gradient`).
- Make constitutive materials more flexible (allow material parameters to be passed at stress and elasticity evaluation `umat.gradient(F, mu=1.0)`).
- Rename `ndim` to `dim`.
- Simplify element base classes.
- Speed-up calculation of indices (rows, cols) for Fields and Forms (about 10x faster now).
- Update `test_element.py` according to changes in element methods.

### Removed
- Automatic check if the gradient of a region can be calculated based on the dimensions. The `grad` argument in `region(grad=False)` has to be enforced by the user.


# License
FElupe - finite element analysis (C) 2021 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
