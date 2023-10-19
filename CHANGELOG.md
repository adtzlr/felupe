# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add cell-type argument to `Mesh.add_midpoints_volumes(cell_type=None)` and its variants for edges and faces.

### Changed
- Change function signature and enhance `dof.biaxial(field, lefts=(None, None), rights=(None, None), moves=(0.2, 0.2), axes=(0, 1), clampes=(False, False), sym=True)`. Now with a full-featured docstring including an example.
- Change function signature and enhance `dof.shear(field, bottom=None, top=None, moves=(0.2, 0.0, 0.0), axes=(0, 1), sym=True)`. Now with a full-featured docstring including an example. This is not backward compatible! However, due to the fact, that this was previously a non-documented function this won't enforce a new major version.
- Merge keyword-arguments for the dual-regions with hard-coded arguments in `FieldsMixed(region, **kwargs)`.
- Replace `np.product()` (will be removed in NumPy 2.0) with the equivalent `np.prod()`.

### Fixed
- Fix `FieldsMixed()` for regions with MINI-element formulations: Disable the disconnection of the dual mesh.
- Fix `dof.shear(sym=True)` which was previously ignored due to a wrong setup of the symmetry boundaries.
- Fix the install command on Python 3.12 by adding an extra-index-url for VTK wheels (provided by Kitware).

### Removed
- Remove `dof.planar()` because this is a special case of the biaxial load case `dof.biaxial(field, clampes=(True, False), moves=(0.2, 0), sym=False, axes=(0, 1))`.

## [7.10.0] - 2023-09-14

### Added
- Add `ViewField` as method to a field container `FieldContainer.view()`. Now a field container provides the view sub-methods directly, i.e. add `FieldContainer.plot()` and `FieldContainer.screenshot(filename="mesh.png")`.

### Changed
- Hide the undeformed mesh in `Mesh.plot()` by default (also affects `Mesh.screenshot()`).

### Fixed
- Fix taking screenshots of a mesh.

## [7.9.0] - 2023-09-13

### Added
- Add `ViewMesh` as method to a mesh `Mesh.view()`. Now a mesh provides the view sub-methods directly, i.e. add `Mesh.plot()` and `Mesh.screenshot(filename="mesh.png")`.

## [7.8.0] - 2023-09-05

### Added
- Add `mesh.Triangle(a, b, c, n=2)` for the creation of a quad-meshed triangle.

### Changed
- Change `mesh.Circle(n=6)` to the minimum `mesh.Circle(n=2)`.
- Enhance `dof.uniaxial(axis=0, sym=(False, True, False))` by a user-defined axis and selective symmetries.
- Enhance `dof.shear(axis_compression=1, axis_shear=0, sym=True)` by user-defined axes of compression and shear.

### Fixed
- Fix `mesh.concatenate([mesh])` for a list of length one by enforcing the dtype of the offset as integer.

## [7.7.0] - 2023-08-31

### Added
- Add optional normalization of mesh runouts (which are then indents) by `mesh.runouts(normalize=False)`.
- Add `LinearElasticLargeStrain(E=None, nu=None, parallel=False)`, suitable for large-rotation analyses. This is based on `NeoHooke()` with converted Lamé-constants.
- Add a simple boundary-based quad- or hex mesher: A mesh tool for filling the face or volume between two line- or quad-meshes `mesh.fill_between(mesh, other_mesh, n=11)`.
- Add `Circle(radius, centerpoint, n)` for the creation of a quad-meshed circle.

### Changed
- Update the mesh also with a new points array: this changes the Mesh-update function `mesh.update(cells, cell_type=None)` to `mesh.update(points=None, cells=None, cell_type=None)`. Note that this could break old scripts which use `mesh.update(new_cells)` instead of `mesh.update(cells=new_cells)`.
- Move the copy-method of a Mesh `Mesh.copy()` to its base-class and extend it to optionally update the attributes `DiscreteGeometry.copy(points=None, cells=None, cell_type=None)`.

### Removed
- Remove tests on Python 3.7 (end of life).

## [7.6.1] - 2023-08-07

### Changed
- Start a virtual framebuffer in Jupyter notebook plotting of `View.plot(notebook=True)`. Note: Requires `sudo apt-get install xvfb`.

## [7.6.0] - 2023-08-06

### Added
- Add Jupyter-Notebook argument to `View.plot(notebook=False)`, which passes the notebook-argument to the pyvista-plotter.

## [7.5.1] - 2023-08-05

### Changed
- Only set a theme for pyvista if the theme-argument is given, i.e. don't call `pv.set_plot_theme(theme)` if the theme-argument is None `View(field).plot(theme=None)`.

## [7.5.0] - 2023-07-20

### Added
- Add `ViewSolid`, which enables `view = ViewSolid(field, solid=None)` the view of cauchy stresses, e.g. `view.plot("Principal Values of Cauchy Stress").show()`.
- Add constitutive models to top-level namespace, e.g. `yeoh()` from `constitution.yeoh()`. This makes typing hyperelastic material formulations shorter: `Hyperelastic(yeoh, C10=0.5, C20=-0.1, C30=0.02)`.
- Add `CharacteristicCurve.plot(swapaxes=False)`.
- Add `MaterialAD`: A user-defined material definition with Automatic Differentiation. Only the first Piola-Kirchhoff stress tensor must be provided.

### Changed
- Add optional point- and cell-data args for `ViewMesh(mesh, point_data=None, cell_data=None)` like already implemented in `ViewField`.
- Enforce contiguous arrays in `UserMaterialHyperelastic` (enhance performance).
- `View`: Switch from `ViewField` to `ViewSolid`.
- `View`: Always plot the undeformed mesh with `opacity=0.2` and `show_edges=False`.
- Rename `UserMaterial` to `Material`, `UserMaterialStrain` to `MaterialStrain`, `UserMaterialHyperelastic` to `Hyperelastic` (keep old alias names until next major release).
- Use consistent indices in `einsum()` for (elementwise operating) trailing axes: `q` for quadrature point and `c` for cell.
- Rename internal `IntegralFormMixed` to `IntegralForm`, which is now consistent internally and in the top-level namespace. The previous internal base-class for a single-field `IntegralForm` is renamed to `WeakForm`.
- Don't plot x- and y-labels in `CharacteristicCurve.plot(xlabel=None, ylabel=None)` if they are not specified.

### Fixed
- Don't warp the mesh in `ViewMesh.plot()`.
- Warp the mesh in case no name is passed in `View.plot(name=None)`.
- Don't modify a given label in `Scene.plot(label=None)`.
- Fix the second invariant of the distortional part of the right Cauchy-Green deformation tensor in hyperelastic material formulations using tensortrax, i.e. fix the implementations of `mooney_rivlin()`, `third_order_deformation()` and `van_der_waals()`.

### Removed
- Remove internal (unused) imports of the assembly submodule.

## [7.4.1] - 2023-05-02

### Changed
- Change the logo.

## [7.4.0] - 2023-04-29

### Added
- Add `ViewMesh(mesh)` and rebase `View` (internally renamed to `ViewField`) on `ViewMesh` with additional point- and cell-data.

### Changed
- Change `math.linsteps(axis=None, axes=None)` to create optional multi-column arrays, where the steps are inserted at the given `axis`.
- Make last `gravity` and `density` arguments of `SolidBodyGravity(field, gravity=None, density=1.0)` optional.

## [7.3.0] - 2023-04-28

### Changed
- Show a progress bar during `Job.evaluate(verbose=True)` (new optional dependency `tqdm`). The legacy detailed output is available with `Job.evaluate(verbose=2)`.

### Removed
- Remove config files for MyBinder. They are now located in a different repository [adtzlr/felupe-web](https://github.com/adtzlr/felupe-web).

## [7.2.0] - 2023-04-26

### Added
- Add `environment.yml` config file for [MyBinder](https://mybinder.org/).
- Add a timetrack-list as `Job.timetrack` which is updated incrementally on `Job.evaluate()`.
- Add `View(field, point_data=None, cell_data=None)`, a result plotter powered by [`pyvista`](https://github.com/pyvista/pyvista).
- Add `ViewXdmf(filename, time=0)`, a result plotter powered by [`pyvista`](https://github.com/pyvista/pyvista).

### Changed
- Make everything in `/src` compliant with [flake8](https://flake8.pycqa.org/).
- Generalize the math-module so that all functions handle an arbitrary number of elementwise-operating trailing axes.
- The special contraction modes of `math.dot(mode=(2,2))` and `math.ddot(mode=(2,2))` have to be specified by the `mode`-argument and are not detected by the shapes of the operands.
- Enhance the overall performance by enforcing the identity matrix to a C-contiguous array.
- Change point- and cell-data functions used in `Job.evaluate(point_data=None, cell_data=None)` from `fun(substep)` to `fun(field, substep)`.

### Fixed
- Fix timings shown in `newtonrhapson(verbose=True)`: The solve time was only related to one call of the solver while the assembly time referred to the whole runtime subtracted by the single-call solve time.

## [7.1.0] - 2023-04-15

### Added
- Add string representations for `Region` and `FieldContainer.`
- Add `Job.evaluate(parallel=True)` in addition to `Job.evaluate(kwargs={"parallel": True})`. If both are given, the key in the dict is overwritten by the user-defined value.
- Add `mesh.stack(meshes)` for joining meshes with identical points-arrays and cell-types. Contrary to `mesh.concatenate(meshes)`, the points are not stacked and no offsets are inserted into the cells-arrays.
- Add `mesh.translate(move, axis)` for the translation of mesh-points.

### Changed
- Pass optional keyword-arguments in `CharacteristicCurve.plot(**kwargs)` to the figure.
- Don't invoke `CharacteristicCurve.evaluate()` from `CharacteristicCurve.plot()`, raise an error if the current job is not evaluated instead.
- Make the endpoint of `math.linsteps(endpoint=True)` optional.
- Don't modify the mesh for the dual regions `RegionConstantQuad()` and `RegionConstantHexahedron()`. Instead, it is required to pass a dual (disconnected) mesh with one point per cell `RegionConstantQuad(mesh.dual(points_per_cell=1))`.
- Make requirement `einsumt` optional again due to issues with JupyterLite.
- Add `matplotlib` to optional requirements.

### Fixed
- Catch `ModuleNotFoundError` if `from einsumt import einsumt` fails (in JupyterLite) and fall back to `from numpy import einsum as einsumt`.

## [7.0.0] - 2023-04-07

### Added
- Add boundary regions `RegionQuadraticQuadBoundary` and `RegionBiQuadraticQuadBoundary` for quadratic quads.
- Add boundary regions `RegionQuadraticHexahedronBoundary` and `RegionTriQuadraticHexahedronBoundary` for quadratic hexahedrons.
- Add `mesh.flip(mask=None)` to flip a mirrored or wrong indexed cells array, applied on a given boolean `mask` of cells.

### Changed
- Change `einsumt` from an optional to a required dependency.
- Vectorize implementations of `MultiPointConstraint` and `MultiPointContact` and re-implement both as `scipy.sparse.lil_matrix()`.
- Rename old `Mesh` to `DiscreteGeometry` and rebase new `Mesh` on `DiscreteGeometry`. 
- Simplify the usage of explicit mesh-related tools by adding them as methods to `Mesh`, i.e. `mesh.tools.expand(Rectangle())` is equivalent to `Rectangle().expand()`.
- Print runtimes for time spent on Assembly and Solve in `newtonrhapson(verbose=True)`.
- Check for negative cell-volumes at quadrature points and print a warning along with a possible fix.

### Fixed
- Fix `tools.project()` for higher-order quad- and hexahedron elements.
- Fix transposed output of `tools.project()`.
- Fix failed scalar-value checks by using `np.isscalar()` in `mesh.expand(z=1)` and `mesh.revolve(phi=180)` where `z` or `phi` are of type `np.int32`.
- Fix read a Mesh with no cells in `mesh.read()`.

### Removed
- Remove `jit`-compilation of forms (`parallel` is the preferred method).
- Remove unused `tools.check()`.
- Remove optional dependency `sparse`.

## [6.4.0] - 2023-04-01

### Added
- Add a new argument to pass a mesh for the dual regions in `FieldsMixed(mesh=None)`.
- Add quadrature and grad arguments to `RegionLagrange`.
- Add order attribute to `RegionLagrange`.
- Add items-argument for custom slicing of characteristic curve plots in `CharacteristicCurve.plot(items=None)`.

### Changed
- Enhance Domain integration in `IntegralForm`: Ensure C-contiguous arrays as `fun`-argument.
- Enhance performance of hyperelastic material model formulations using automatic differentiation.
- Perform reshape and broadcasting if user-defined `Field`-values with correct size are given.
- Make symmetry-related boundary conditions in uniaxial loadcase optional `dof.uniaxial(sym=True)`.
- Merge custom point- and cell-data dicts with default dicts in `Job` instead of replacing them, also add `Job(point_data_default=True, cell_data_default=True)`.
- Allow to change cell-type in `Mesh.update(cells, cell_type=None)`.
- Enhance the creation of a disconnected mesh for mixed-field formulations by `Mesh.disconnect(points_per_cell=None, calc_points=True)`.
- Change required to optional arguments in `Step(items, ramp=None, boundaries=None)`.

### Fixed
- Fix broadcast arrays for the geometric stiffness contribution of hyperelastic material model formulations using automatic differentiation.

## [6.3.0] - 2023-02-06

### Added
- Add more hyperelastic material formulations: Saint Venant-Kirchhoff, Mooney-Rivlin, Arruda-Boyce, Extended Tube and Van der Waals.
- Add `BiQuadraticQuad` element (as a permuted version of `ArbitraryOrderLagrangeElement`).
- Add `Quadratic` element.
- Add new arguments for more flexible region templates, e.g. `RegionQuad(mesh, quadrature=GaussLegendre(order=1, dim=2), grad=True)`.
- Add support for triangle/tetra in `FieldsMixed()`.
- Add optional state variables to `UserMaterialHyperelastic(model, nstatevars=0)`.
- Add finite-strain viscoelastic material formulation.

### Changed
- Switch to a `src/`-layout.
- Import base-element class as `element.Element()`.
- Auto-detect min/max-coordinates of mesh-points for loadcases if `dof.uniaxial(right=None)`.
- Ensure compatibility with `tensortrax>0.6.0` (Van-der-Waals and Viscoelastic models).

### Fixed
- Fix rotation matrix for a rotation around the y-axis.

## [6.2.5] - 2023-01-02

### Fixed
- Once again fix init if `tensortrax` is not installed.

## [6.2.4] - 2023-01-01

### Changed
- Update actions used in CI/CD workflows.

## [6.2.3] - 2023-01-01

### Changed
- Remove `setup.cfg`, change `pyproject.toml` and store the version tag only once within the source code (`__about__.py`).

## [6.2.2] - 2022-12-20

### Fixed
- Fix init if `tensortrax` is not installed.

## [6.2.1] - 2022-12-19

### Fixed
- Fix version string.

## [6.2.0] - 2022-12-16

### Added
- Add Total-Lagrange `UserMaterialHyperelastic(fun, parallel=False, **kwargs)` based on optional `tensortrax`. Only available if `tensortrax` is installed.
- Add constitutive isotropic hyperelastic model formulations to be used with `UserMaterialHyperelastic()` (`constitution.ogden(C, mu, alpha)`, etc.).

## [6.1.0] - 2022-12-10

### Changed
- Enhance plotting with custom `x`- and `y`-data in `CharacteristicCurve.plot(x, y)` and allow a list of items for force evaluation in `CharacteristicCurve(items=[...])` to be passed.
- Enhance `math.linsteps(points=[0, 5, 0], num=[5, 10])` by supporting a list of substeps.
- Enhance compression of shear loadcase: Apply the compression on the bottom and the top `dof.shear(compression=(0, 0))`.

## [6.0.0] - 2022-11-20

### Added
- Add `project(mean=True)` to project cell mean-values to mesh-points. Now `project()` supports Triangles and Tetrahedrons.
- Add `RegionBoundary.mesh_faces()` for a mesh with face-cells on the selected boundary of a region.
- Add pseudo-elastic material `OgdenRoxburgh()` which may be used with `SolidBodyNearlyIncompressible()`.
- Add `umat = UserMaterial(stress, elasticity, nstatevars=0, **kwargs)` with user-defined functions for the (first Piola-Kirchhoff) stress tensor `P, statevars_new = umat.gradient([F, statevars], **kwargs)` and the according fourth-order elasticity tensor `A = umat.hessian([F, statevars], **kwargs)` based on the deformation gradient.
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
