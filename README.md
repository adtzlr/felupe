# FElupe - Finite Element Analysis

[![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![Documentation Status](https://readthedocs.org/projects/felupe/badge/?version=latest)](https://felupe.readthedocs.io/en/latest/?badge=latest) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/gh/adtzlr/felupe/branch/main/graph/badge.svg?token=J2QP6Y6LVH)](https://codecov.io/gh/adtzlr/felupe) [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![GitHub Repo stars](https://img.shields.io/github/stars/adtzlr/felupe?logo=github) ![PyPI - Downloads](https://img.shields.io/pypi/dm/felupe)

<img src="https://raw.githubusercontent.com/adtzlr/felupe/main/docs/_static/logo_light.svg" width="220px"/>

FElupe is a Python 3.6+ finite element analysis package focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. Its name is a combination of FE (finite element) and the german word *Lupe* (magnifying glass) as a synonym for getting an insight how a finite element analysis code looks like under the hood.

# Installation
Install Python, fire up a terminal and run

```shell
pip install felupe[all]
```

where `[all]` installs all optional dependencies. By default, FElupe only depends on `numpy` and `scipy`. In order to make use of all features of FElupe, it is suggested to install all optional dependencies.

# Hello, FElupe!
A quarter model of a solid cube with hyperelastic material behaviour is subjected to a uniaxial elongation applied at a clamped end-face. This involves the creation of a mesh, a region as well as a displacement field (encapsulated in a field container). Furthermore, the boundary conditions are created by a template for a uniaxial loadcase. A Neo-Hookean material formulation is applied on a solid body. A step generates the consecutive substep-movements of a given boundary condition. The step is added to a job. During evaluation, each substep of each step is solved by an iterative Newton-Rhapson procedure. Finally, the solution of the last step is exported as a VTK file. For more details beside this high-level code snippet, please have a look at the [documentation](https://felupe.readthedocs.io/en/latest/?badge=latest).

```python
import felupe as fem

# create a hexahedron-region on a cube
region = fem.RegionHexahedron(fem.Cube(n=11))

# add a displacement field and apply a uniaxial elongation on the cube
displacement = fem.Field(region, dim=3)
field = fem.FieldContainer([displacement])
boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

# define the constitutive material behaviour and create a solid body
umat = fem.NeoHooke(mu=1.0, bulk=2.0)
solid = fem.SolidBody(umat, field)

# prepare a step with substeps
move = fem.math.linsteps([0, 0.2], num=10)
step = fem.Step(
    items=[solid], 
    ramp={boundaries["move"]: move}, 
    boundaries=boundaries
)

# add the step to a job and evaluate all substeps
job = fem.Job(steps=[step])
job.evaluate()

# save result of last substep
fem.save(region, field, filename="result.vtk")
```

<img src="https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/readme.png" width="600px"/>

# Documentation
The documentation is located [here](https://felupe.readthedocs.io/en/latest/?badge=latest).

# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [5.1.0] - 2022-09-09

### Changed
- Enhance `Boundary`: Select Points by value in addition to a callable (`fx=lambda x: x == 0` is equivalent to `fx=0`), also add `mode="and"` and `mode="or"` argument.
- Support line elements within the revolution function `mesh.revolve()`.
- Import previously hidden functions `fun_items()` and `jac_items()` as `tools.fun()` and `tools.jac()`, respectively (useful for numeric continuation).
- Add step- and substep-numbers as arguments to the `callback(stepnumber, substepnumber, substep)`-function of a `Job`.

## [5.0.0] - 2022-08-21

### Added
- Add `SolidBodyGravity` for body forces acting on a solid body.
- Support list of linked fields in Newton-Rhapson solver `newtonrhapson(fields=[field_1, field_2])`.
- Automatic init of state variables in `SolidBodyTensor`.
- Add `mesh.runouts()` for the creation of runouts of rubber-blocks of rubber-metal structures.
- Add `FieldPlaneStrain` which is a 2d-field and returns gradients of shape `(3, 3)` (for plane strain problems with 3d user materials).
- Add `PointLoad` for the creation of external force vectors.
- Add `Step` with a generator for substeps, `Job` and `CharacteristicCurve`.

### Changed
- Move `MultiPointConstraint` to mechanics module and unify handling with `SolidBody`.
- Rename `bodies` argument of Newton-Rhapson solver to `items` (now supports MPC).
- Return partitioned system as dict from loadcases `loadcase=dict(dof0=dof0, dof1=dof1, ext0=ext0)`.
- Check function residuals norm in `newtonrhapson()` instead of incremental field-values norm.

### Fixed
- Fix assembled vectors and results of `SolidBodyPressure` for initially defined pressure values.
- Fix `verbose=0` option of `newtonrhapson()`.
- Fix wrong assembly of axisymmetric mixed-fields due to introduced plane strain field-trimming.

## [4.0.0] - 2022-08-07

### Added
- Add `SolidBody.evaluate.kirchhoff_stress()` method. Contrary to the Cauchy stress method, this gives correct results in incompressible plane stress.
- Add `SolidBodyTensor` for tensor-based material definitions with state variables.
- Add `bodies` argument to `newtonrhapson()`.
- Add a container class for fields, `FieldContainer` (renamed from `FieldMixed`).
- Add `len(field)` method for `FieldContainer` (length = number of fields).

### Changed
- Unify handling of `Field` and `FieldMixed`.
- Constitutive models use lists as in- and output (consistency between single- and mixed-formulations).
- Allow field updates directly from 1d sparse-solved vector without splitted by field-offsets.

### Fixed
- Fix `tovoigt()` helper for data with more or less than two trailing axes and 2D tensors.
- Fix errors for `force()` and `moment()` helpers if the residuals are sparse.

### Removed
- Remove wrapper for matADi-materials (not necessary with field containers).
- Remove `IntegralFormMixed` and `IntegralFormAxisymmetric` from global namespace.

## [3.1.0] - 2022-05-02

### Added
- Add optional parallel (threaded) basis evaluation and add `Form(v, u, parallel=True)`.
- Add `mechanics` submodule with `SolidBody` and `SolidBodyPressure`.

### Fixed
- Fix matADi materials for (mixed) axisymmetric analyses.
- Fix missing radius in axisymmetric integral forms.

## [3.0.0] - 2022-04-28

### Added
- Add `sym` argument to `Bilinearform.integrate()` and `Bilinearform.assemble()`.
- Add `FieldsMixed` which creates a `FieldMixed` of length `n` based on a template region.
- Add function to mirror a Mesh `mesh.mirror()`.
- Add a new `parallel` assembly that uses a threaded version of `np.einsum` instead ([einsumt](https://pypi.org/project/einsumt/)).
- Add parallel versions of math helpers (`dya`, `cdya`, `dot`, `ddot`) using [einsumt](https://pypi.org/project/einsumt/).
- Add `parallel` keyword to constitutive models (`NeoHooke`, `LinearElasticTensorNotation` and `ThreeFieldVariation`).
- Add `RegionBoundary` along with template regions for `Quad` and `Hexahedron` and `GaussLegendreBoundary`.
- Add optional normal vector argument for function and gradient methods of `AreaChange`.
- Add a new Mesh-tool `triangulate()`, applicable on Quad and Hexahedron meshes.
- Add a new Mesh-method `Mesh.as_meshio()`.
- Add a function decorator `@Form(...)` for linear and bilinear form objects.

### Changed
- Enforce consistent arguments for functions inside `mesh` (`points, cells, cell_data` or `Mesh`).
- Rename Numba-`parallel` assembly to `jit`.
- Move single element shape functions and their derivatives from `region.h` to `region.element.h` and `region.dhdr` to `region.element.dhdr`.
- [Repeat](https://numpy.org/doc/stable/reference/generated/numpy.tile.html) element shape functions and their derivatives for each cell (as preparation for an upcoming `RegionBoundary`).
- Improve `mesh.convert()` by using the function decorator `@mesh_or_data`.
- Allow an array to be passed as the expansion arguments of `mesh.expand()` and `mesh.revolve()`.
- Allow optional keyword args to be passed to `Mesh.save(**kwargs)`, acts as a wrapper for `Mesh.as_meshio(**kwargs).write()`.

### Fixed
- Fix area normal vectors of `RegionBoundary`.
- Fix integration and subsequent assembly of `BilinearForm` if field and mesh dimensions are not equal.

## [2.0.1] - 2022-01-11

### Fixed
- Fixed wrong result of assembly generated by a parallel loop with `prange`.

## [2.0.0] - 2022-01-10

### Added
- Add a new method to deepcopy a `Mesh` with `Mesh.copy()`
- Add [*broadcasting*](https://numpy.org/doc/stable/user/basics.broadcasting.html) capability for trailing axes inside the parallel form integrators.
- Add `Basis` on top of a field for virtual fields used in linear and bilinear forms.
- Add `LinearForm` and `BilinearForm` (including mixed variants) for vector/matrix assembly out of weak form expressions.
- Add `parallel` keyword for threaded integration/assembly of `LinearForm` and `BilinearForm`.

### Changed
- Enhance `Boundary` for the application of prescribed values of any user-defined `Field` which is part of `FieldMixed`.
- The whole mixed-field has to be passed to `dof.apply()` along with the `offsets` returned from `dof.partition` for mixed-field formulations.
- Set default value `shape=(1, 1)` for `hessian()` methods of linear elastic materials.

### Fixed
- Fixed einstein summation of `math.dot()` for two vectors with trailing axes.

### Removed
- Remove `dof.extend` because `dof.partition` does not need it anymore.

## [1.6.0] - 2021-12-02

### Added
- Add `LinearElasticPlaneStress` and `LinearElasticPlaneStrain` material formulations.
- Add `region` argument for `LinearElastic.hessian()`.

### Changed
- Re-formulate `LinearElastic` materials in terms of the deformation gradient.
- Re-formulate `LinearElastic` material in matrix notation (Speed-up of ~10 for elasticity matrix compared to previous implementation.) 
- Move previous `LinearElastic` to `constitution.LinearElasticTensorNotation`.

## [1.5.0] - 2021-11-29

### Added
- Add kwargs of `field.extract()` to `fun` and `jac` of `newtonrhapson`.

### Changed
- Set default number of `threads` in `MatadiMaterial` to `multiprocessing.cpu_count()`.
- Moved documentation to Read the Docs (Sphinx).

### Fixed
- Fix `dim` in calculation of reaction forces (`tools.force`) for `FieldMixed`.
- Fix calculation of reaction moments (`tools.moment`) for `FieldMixed`.

## [1.4.0] - 2021-11-15

### Added
- Add `mask` argument to `Boundary` for the selection of user-defined points.
- Add `shear` loadcase.
- Add a wrapper for `matadi` materials as `MatadiMaterial`.
- Add `verbose` and `timing` arguments to `newtonrhapson`.

### Fixed
- Obtain internal `dim` from Field in calculation of reaction force `tools.force`.
- Fix `math.dot` for combinations of rank 1 (vectors), rank 2 (matrices) and rank 4 tensors.

## [1.3.0] - 2021-11-02

### Changed
- Rename `mesh.as_discontinous()` to `mesh.disconnect()`.
- Rename `constitution.Mixed` to `constitution.ThreeFieldVariation`.
- Rename `unstack` to `offsets` as return of dof-partition and all subsequent references.
- Import tools (`newtonrhapson`, `project`, `save`) and constitution (`NeoHooke`, `LinearElastic` and `ThreeFieldVariation`) to FElupe's namespace.
- Change minimal README-example to a high-level code snippet and refer to docs for details.

## [1.2.0] - 2021-10-31

### Added
- Add template regions, i.e. a region with a `Hexahedron()` element and a quadrature scheme `GaussLegendre(order=1, dim=3)` as `RegionHexahedron`, etc.
- Add biaxial and planar loadcases (like uniaxial).
- Add a minimal README-example (Hello FElupe!).

### Changed
- Deactivate clamped boundary (`clamped=False`) as default option for uniaxial loading `dof.uniaxial`.

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
FElupe - finite element analysis (C) 2022 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
