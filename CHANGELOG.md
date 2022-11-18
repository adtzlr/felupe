# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `project(mean=True)` to project cell mean-values to mesh-points. Now `project()` supports Triangles and Tetrahedrons.
- Add `RegionBoundary.mesh_faces()` for a mesh with face-cells on the selected boundary of a region.
- Add pseudo-elastic material `OgdenRoxburgh()` which may be used with the new `SolidBodyTensorNearlyIncompressible()`.
- Add `umat = UserMaterial(stress, elasticity, nstatevars=0, **kwargs)` with user-defined functions for the (first Piola-Kirchhoff) stress tensor `P, statevars_new = umat.gradient([F, statevars], **kwargs)` and the according fourth-order elasticity tensor `A = umat.hessian([F, statevars], **kwargs)`.
- Add `UserMaterialStrain()` for small-strain based user-defined material formulations with an umat-interface suitable for elastic-plastic frameworks.
- Add `LinearElasticPlasticIsotropicHardening()` which is based on `UserMaterialStrain()` and `constitution.linear_elastic_plastic_isotropic_hardening()`.
- Add new math helpers `math.ravel()` and `math.reshape()`.
- Add optional axis argument on which the norm is evaluated `math.norm(axis=None)`.

### Changed
- Unify material definition with methods for the stress `P, statevars_new = umat.gradient([F, statevars])` and the elasticity tensor `A = umat.hessian([F, statevars])`. This breaks support for materials defined by matadi<=0.1.10.
- Do not broadcast the (constant) elasticity tensor for linear-elastic materials as `einsumt>=0.9.3` supports broadcasting along the parallel-executed dimension.
- Change not-updating attribute of `FieldContainer(fields).values` to a method `FieldContainer(fields).values()` which returns the current field values.

### Removed
- Remove unused `SolidBodyTensor()` and `SolidBodyTensorNearlyIncompressible()`.
- Remove unused `region` argument of `LinearElastic().hessian()`.

## [5.3.1] - 2022-11-03

### Fixed
- Fix volume evaluation of (nearly) incompressible solids for axisymmetric fields.

## [5.3.0] - 2022-11-03

### Added
- Add optional pre-compression to shear-loadcase `dof.shear(compression=0.0)`.
- Add `MeshContainer` and string-representation for `Mesh` objects.
- Add a mesh-reader using meshio `mesh.read(filename, ...)`.
- Add `SolidBodyNearlyIncompressible(umat, field, bulk)` for (nearly) incompressible solids and a given (distortional-part only) constitutive material formulation. This is a pure displacement-based alternative to the three-field-formulation technique.

### Changed
- Support an optional user-defined meshio-object in `Job().evaluate(mesh=None, filename="result.xdmf")`.
- Support a distortional-part only Neo-Hookean material formulation with no bulk modulus defined `NeoHooke(mu=1.0)`.

### Fixed
- Fix missing `ArbitraryOrderLagrangeElement.points` attribute.
- Fix ignored mask `only_surface=True` for `RegionBoundary().mesh.cells_faces`.
- Set default pressure to zero in `SolidBodyPressure()`.
- Take the mesh from the global `x0`-field if `x0` is passed to `job.evaluate(x0=...)`.
- Fix missing update of global field `x0` in `job.evaluate(x0)` after each completed substep.

## [5.2.0] - 2022-10-08

### Added
- Add `xscale` and `yscale` arguments to `CharacteristicCurve.plot()`.
- Add `mesh.Grid(*xi)` as generalized line, rectangle or cube with custom linspaces.
- Add `mesh.concatenate(meshes)` to join a sequence of meshes with identical cell types.
- Add `x0` argument to `Job.evaluate(x0=field)`.
- Add `mask` argument to `mesh.runouts(mask=slice(None))`.
- Add `callback(stepnumber, substepnumber, substep)` argument to `CharacteristicCurve()` (like in `Job()`).
- Add an on-the-fly XDMF writer for a job (via meshio) `Job.evaluate(filename="result.xdmf")` with the possibility to add optional `point_data` and `cell_data` dicts.

### Changed
- Remove Warning if `einsumt` requirement is not found (switch to numpy without any warnings).
- Requires Python 3.7+.

### Fixed
- Fix ignored axis argument of `mesh.revolve(axis=1)`.
- Fix missing `ArbitraryOrderLagrangeElement.points` attribute.

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
