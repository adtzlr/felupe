## Example 1 - (u,p,J) - Mixed-field formulation for nearly-incompressible hyperelasticity
FElupe supports mixed-field formulations in a similar way it can handle (default) single-field variations. The definition of a mixed-field variation is shown for the hydrostatic-volumetric selective three-field-variation with independend fields for displacements $\bm{u}$, pressure $p$ and volume ratio $J$. The total potential energy for nearly-incompressible hyperelasticity is formulated with a determinant-modified deformation gradient. We take the [Getting Started](quickstart.md) example and modify it accordingly. We take the built-in Neo-Hookean material model and use it for FElupe's `ThreeFieldVariation` constitutive material class as described [here](guide.md).

```python
import felupe as fe

neohooke = fe.constitution.NeoHooke(mu=1.0, bulk=5000.0)
umat = fe.constitution.ThreeFieldVariation(neohooke)
```

Next, let's create a meshed cube. Two regions, one for the displacements and another one for the pressure and the volume ratio are created.


```python
mesh  = fe.Cube(n=6)

region  = fe.RegionHexahedron(mesh)
region0 = fe.RegionConstantHexahedron(mesh)

dV = region.dV

displacement = fe.Field(region,  dim=3)
pressure     = fe.Field(region0, dim=1)
volumeratio  = fe.Field(region0, dim=1, values=1)

fields = fe.FieldMixed((displacement, pressure, volumeratio))
```

Boundary conditions are enforced in the same way as in [Getting Started](quickstart.md).

```python
import numpy as np

f1 = lambda x: np.isclose(x, 1)

boundaries = fe.doftools.symmetry(displacement)
boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
boundaries["move" ] = fe.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=-0.4)

dof0, dof1, offsets = fe.dof.partition(fields, boundaries)
u0ext = fe.dof.apply(displacement, boundaries, dof0)
```

The Newton-Rhapson iterations are coded quite similar to the one used in [Getting Started](quickstart.md). FElupe provides a Mixed-field version of it's `IntegralForm`, called `IntegralFormMixed`. It assumes that the first field operates on the gradient and all the others don't. The resulting system vector with incremental values of the fields has to be splitted at the field-offsets and updated.

```python
for iteration in range(8):

    F, p, J = fields.extract()
    
    linearform   = fe.IntegralFormMixed(umat.gradient(F, p, J), fields, dV)
    bilinearform = fe.IntegralFormMixed(umat.hessian(F, p, J), fields, dV, fields)

    r = linearform.assemble().toarray()[:, 0]
    K = bilinearform.assemble()
    
    system = fe.solve.partition(fields, K, dof1, dof0, r)
    dfields = np.split(fe.solve.solve(*system, u0ext), offsets)
    
    fields += dfields

    norm = np.linalg.norm(dfields[0])
    print(iteration, norm)

    if norm < 1e-12:
        break

fe.tools.save(region, fields, offsets=offsets, filename="result.vtk")
```

The deformed cube is visualized by the VTK output with the help of Paraview.

![deformed cube](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/threefield_cube.png)