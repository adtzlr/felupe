# Examples

## Example 1 - 2d Poisson equation
The poisson problem $-\nabla u = f$ is solved with fixed boundaries and a unit load. In a first step, we create a rectangular mesh, initialize an instance of a linear quad element and the appropriate quadrature. A numeric region is created with all three objects.

```python
import numpy as np
import felupe as fe

n = 51
mesh = fe.mesh.Rectangle(n=n)
element = fe.element.Quad()
quadrature = fe.quadrature.GaussLegendre(order=1, dim=2)

region = fe.Region(mesh, element, quadrature)
```

![poisson mesh](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson_mesh.png)

In the next step we create a one-dimensional (scalar) field for our 2d-region. 

```python
field = fe.Field(region)
```

Both the laplace equation and the right-hand-side of the poisson problem are transformed from the local differential into the (weak) integral form as follows. We start with the local differential form.

$$-\nabla^2 u = f$$

Multiplying the above equation with $\delta u$ and integrate it in the region $\Omega$.

$$-\int_\Omega \nabla^2 u \cdot \delta u \ d\Omega = \int_\Omega f \cdot \delta u \ d\Omega$$

Next we do integration by parts to obtain the (weak) integral form. Note: We don't care about the boundary terms as they are all zero in this example. We now have the integral form of the laplace equation and and the right-hand-side **at hand**.

$$\int_\Omega \nabla \delta u \cdot \nabla u\ \ d\Omega = \int_\Omega f \cdot \delta u \ d\Omega$$

```python
unit_load = np.ones_like(field.interpolate())

bilinearform = fe.IntegralForm(
    fun=fe.math.laplace(field), v=field, dV=region.dV, u=field, grad_v=True, grad_u=True
)

linearform = fe.IntegralForm(fun=unit_load, v=field, dV=region.dV)
```

The assembly of the two forms give the "stiffness" matrix $\bm{A}$ and the right-hand-side $\bm{b}$.

```python
A = bilinearform.assemble()
b = linearform.assemble()
```

As already mentioned all boundary points are fixed.

```python
f0, f1 = lambda x: np.isclose(x, 0), lambda x: np.isclose(x, 1)

boundaries = {}
boundaries["left"  ] = fe.Boundary(field, fx=f0)
boundaries["right" ] = fe.Boundary(field, fx=f1)
boundaries["bottom"] = fe.Boundary(field, fy=f0)
boundaries["top"   ] = fe.Boundary(field, fy=f1)

dof0, dof1 = fe.doftools.partition(field, boundaries)
```

```python
system = fe.solve.partition(field, A, dof1, dof0, -b)
dfield = fe.solve.solve(*system)
field += dfield

fe.utils.save(region, field, filename="poisson.vtk")
```

The solution of the unknowns is first visualized by the VTK output with the help of Paraview.

![poisson solution](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson_solution.png)

Another possibility would be a plot with matplotlib.

```python

import matplotlib.pyplot as plt
cf = plt.contourf(
    mesh.points[:,0].reshape(n, n), 
    mesh.points[:,1].reshape(n, n), 
    field.values[:,0].reshape(n, n),
)
plt.gca().set_aspect("equal")
plt.gcf().colorbar(cf, ax=plt.gca(), shrink=0.9)

plt.savefig("poisson.svg")
```

![poisson plot](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson.svg)

## Example 2 - Mixed-field variation for nearly-incompressible hyperelasticity
Felupe supports mixed-field formulations in a similar way it can handle (default) single-field variations. The definition of a mixed-field variation is shown for the hydrostatic-volumetric selective three-field-variation with independend fields for displacements $\bm{u}$, pressure $p$ and volume ratio $J$. The total potential energy for nearly-incompressible hyperelasticity is formulated with a determinant-modified deformation gradient.

### FElupe implementation
We take the [Getting Started](quickstart.md) example and modify it accordingly. We take the built-in Neo-Hookean material model and use it for FElupe's `ThreeFieldVariation` constitutive material class as described [here](guide.md).

```python
neohooke = fe.constitution.NeoHooke(mu=1.0, bulk=5000.0)
umat = fe.constitution.variation.upJ(neohooke.P, neohooke.A)
```

We create a meshed cube and a converted version for the piecewise constant fields per cell. Two element definitions are necessary: one for the displacements and one for the pressure and volume ratio. Both elements use the same quadrature rule. Two regions are created, which will be further used by the creation of the fields.


```python
mesh  = fe.mesh.Cube(n=6)
mesh0 = fe.mesh.convert(mesh, order=0)

element0 = fe.element.ConstantHexahedron()
element1 = fe.element.Hexahedron()
quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

region0 = fe.Region(mesh0, element0, quadrature)
region  = fe.Region(mesh,  element1, quadrature)

dV = region.dV

displacement = fe.Field(region,  dim=3)
pressure     = fe.Field(region0, dim=1)
volumeratio  = fe.Field(region0, dim=1, values=1)
fields = fe.FieldMixed((displacement, pressure, volumeratio))
```

Boundary conditions are enforced in the same way as in [Getting Started](quickstart.md).

```python
f1 = lambda x: np.isclose(x, 1)

boundaries = fe.doftools.symmetry(displacement)
boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
boundaries["move" ] = fe.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=-0.4)

dof0, dof1, unstack = fe.doftools.partition(fields, boundaries)
u0ext = fe.doftools.apply(displacement, boundaries, dof0)
```

The Newton-Rhapson iterations are coded quite similar to the one used in [Getting Started](quickstart.md). FElupe provides a Mixed-field version of it's `IntegralForm`, called `IntegralFormMixed`. It assumes that the first field operates on the gradient and all the others don't. Of course, the incremental solutions of the fields have to be splitted and updated seperately.

```python
for iteration in range(8):
    
    dudX = displacement.grad()
    
    F = fm.identity(dudX) + dudX
    p = pressure.interpolate()
    J = volumeratio.interpolate()
    
    linearform   = fe.IntegralFormMixed(umat.f(F, p, J), fields, dV)
    bilinearform = fe.IntegralFormMixed(umat.A(F, p, J), fields, dV)

    r = linearform.assemble().toarray()[:, 0]
    K = bilinearform.assemble()
    
    system = fe.solve.partition(fields, K, dof1, dof0, r)
    dfields = np.split(fe.solve.solve(*system, u0ext), unstack)
    
    for field, dfield in zip(fields.fields, dfields):
        field += dfield

    norm = np.linalg.norm(dfields[0])
    print(iteration, norm)

    if norm < 1e-12:
        break

fe.utils.save(region, fields, unstack=unstack, filename="result.vtk")
```

The deformed cube is visualized by the VTK output with the help of Paraview.

![deformed cube](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/threefield_cube.png)