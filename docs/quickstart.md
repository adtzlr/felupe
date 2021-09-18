## Problem Definition

As with every module: first install FElupe with `pip install felupe`, then import FElupe in your script as shown below. It is recommended to shorten the imported name for better readability.

```python
import felupe as fe
```

Start setting up a problem in FElupe by the creation of a numeric **Region** with a geometry (**Mesh**), a finite **Element** and a **Quadrature** rule, e.g. for hexahedrons or tetrahedrons.

![FElupe](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/numeric_region.png)

```python
mesh = fe.mesh.Cube(n=9)
element = fe.element.Hexahedron()
quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)
```

## Region
A region essentially pre-calculates basis functions and derivatives evaluated at every quadrature point of every cell. An array containing products of quadrature weights multiplied by the geometrical jacobians is stored as the differential volume.

```python
region = fe.Region(mesh, element, quadrature)

dV = region.dV
V = dV.sum()
```

![FElupe](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/undeformed_mesh.png)

## Field
In a second step fields may be added to the Region. These may be either scalar or vector-valued fields. The nodal values are obtained with the attribute `values`. Interpolated field values at quadrature points are calculated with the `interpolate()` method. Additionally, the displacement gradient w.r.t. the undeformed coordinates is calculated for every quadrature point of every cell in the region with the field method `grad()`. 

```python
displacement = fe.Field(region, dim=3)

u    = displacement.values
ui   = displacement.interpolate()
dudX = displacement.grad()
```

The deformation gradient is obtained by a sum of the identity and the displacement gradient.

```python
from felupe.math import identity

F = identity(dudX) + dudX
```

## Constitution
The material behavior has to be provided by the first Piola-Kirchhoff stress tensor as a function of the deformation gradient. FElupe provides a constitutive library which contains a definition for the Neo-Hookean material model and the associated fourth-order elasticity tensor. Alternatively, an isotropic material formulation is defined by a strain energy density function - both variation (stress) and linearization (elasticity) are carried out by automatic differentiation. The latter one is demonstrated here with a nearly-incompressible version of the Neo-Hookean material model.

$$\psi = \frac{\mu}{2} \left( J^{-2/3} \text{tr}\boldsymbol{C} - 3 \right) + \frac{K}{2} \left( J - 1 \right)^2$$

```python
from casadi import det, transpose, trace

def W(F, mu, bulk):
    "Neo-Hooke"

    J = det(F)
    C = transpose(F) @ F

    return mu/2 * (J**(-2/3)*trace(C) - 3) + bulk/2 * (J - 1)**2

umat = fe.constitution.StrainEnergyDensity(W, mu=1.0, bulk=2.0)

P = umat.P
A = umat.A
```

## Boundary Conditions
Next we introduce boundary conditions on the displacement field. Boundary conditions are stored inside a dictionary of multiple boundary instances. First, we fix the left end of the cube. Displacements on the right end are fixed in directions y and z whereas displacements in direction x are prescribed with a value. A boundary instance hold useful attributes like `points` or `dof`.

```python
import numpy as np

f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

boundaries = {}
boundaries["left"]  = fe.Boundary(displacement, fx=f0)
boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1,0,0))
boundaries["move"]  = fe.Boundary(displacement, fx=f1, skip=(0,1,1), value=0.5)
```

## Partition of deegrees of freedom
The separation of active and inactive degrees of freedom is performed by a so-called partition. External values of prescribed displacement degrees of freedom are obtained by the application of the boundary values to the displacement field.

```python
dof0, dof1 = fe.doftools.partition(displacement, boundaries)
u0ext = fe.doftools.apply(displacement, boundaries, dof0)
```

## Integral forms of equilibrium equations
The integral (or weak) forms of equilibrium equations are defined by the `IntegralForm` class. The function of interest has to be passed as the `fun` argument whereas the virtual field as the `v` argument. Setting `grad_v=True` FElupe passes the gradient of the virtual field to the integral form. FElupe assumes a linear form if `u=None` (default) or creates a bilinear form if a field is passed to the argument `u`.

$\int_V P_i^{\ J} : \frac{\partial \delta u^i}{\partial X^J} \ dV \qquad$ and $\qquad \int_V \frac{\partial \delta u^i}{\partial X^J} : \mathbb{A}_{i\ k\ }^{\ J\ L} : \frac{\partial u^k}{\partial X^L} \ dV$

```python
linearform   = fe.IntegralForm(P(F), displacement, dV, grad_v=True)
bilinearform = fe.IntegralForm(A(F), displacement, dV, u=displacement, grad_v=True, grad_u=True)
```

Assembly of both forms lead to the (point-based) internal forces and the (sparse) stiffness matrix.

```python
r = linearform.assemble().toarray()[:,0]
K = bilinearform.assemble()
```

## Prepare (partition) and solve the linearized equation system
In order to solve the linearized equation system a partition into active and inactive degrees of freedom has to be performed. This system may then be passed to the (sparse direct) solver. Given a set of nonlinear equilibrium equations $\boldsymbol{g}$ the unknowns $\boldsymbol{u}$ are found by linearization at a valid initial state of equilibrium and an iterative Newton-Rhapson solution prodecure. The incremental values of inactive degrees of freedom are given as the difference of external prescribed and current values of unknowns. The (linear) solution is equal to the first result of a Newton-Rhapson iterative solution procedure. The resulting point values `du` are finally added to the displacement field. 

$\boldsymbol{g}_1(\boldsymbol{u}) = -\boldsymbol{r}_1(\boldsymbol{u}) + \boldsymbol{f}_1$

$\boldsymbol{g}_1(\boldsymbol{u} + d\boldsymbol{u}) \approx -\boldsymbol{r}_1 + \boldsymbol{f}_1 - \frac{\partial \boldsymbol{r}_1}{\partial \boldsymbol{u}_1} \ d\boldsymbol{u}_1 - \frac{\partial \boldsymbol{r}_1}{\partial \boldsymbol{u}_0} \ d\boldsymbol{u}_0 = \boldsymbol{0}$

$d\boldsymbol{u}_0 = \boldsymbol{u}_0^{(ext)} - \boldsymbol{u}_0$

solve $\boldsymbol{K}_{11}\ d\boldsymbol{u}_1 = \boldsymbol{g}_1 - \boldsymbol{K}_{10}\ d\boldsymbol{u}_{0}$

$\boldsymbol{u}_0 += d\boldsymbol{u}_0$

$\boldsymbol{u}_1 += d\boldsymbol{u}_1$

```python
system = fe.solve.partition(displacement, K, dof1, dof0, r)
du = fe.solve.solve(*system, u0ext).reshape(*u.shape)
# displacement += du
```

A very simple newton-rhapson code looks like this:

```python
for iteration in range(8):
    dudX = displacement.grad()
    F = identity(dudX) + dudX

    linearform = fe.IntegralForm(P(F), displacement, dV, grad_v=True)
    bilinearform = fe.IntegralForm(
        A(F), displacement, dV, displacement, grad_v=True, grad_u=True
    )

    r = linearform.assemble().toarray()[:, 0]
    K = bilinearform.assemble()

    system = fe.solve.partition(displacement, K, dof1, dof0, r)
    du = fe.solve.solve(*system, u0ext).reshape(*u.shape)

    norm = np.linalg.norm(du)
    print(iteration, norm)
    displacement += du

    if norm < 1e-12:
        break
```

```shell
0 8.174171563674951
1 0.29407692748950903
2 0.02082985120923449
3 0.00010287527746122543
4 6.014241696552864e-09
5 5.345117354172557e-16
```

All 3x3 components of the deformation gradient of integration point 1 of cell 1 are obtained with

```python
F[:,:,0,0]
```

```shell
array([[ 1.21739987e+00, -2.08931457e-02, -2.08944754e-02],
       [ 2.45510875e-01,  8.26331927e-01,  4.79381040e-04],
       [ 2.45526598e-01,  4.79183377e-04,  8.26337960e-01]])
```

## Export of results
Results can be exported as VTK or XDMF files using meshio.

```python
fe.utils.save(region, displacement, filename="result.vtk")
```

![FElupe](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/deformed_mesh.png)