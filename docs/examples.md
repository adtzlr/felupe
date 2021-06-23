# Examples

## Example 1 - 2d Poisson equation
The poisson problem $\Delta u = - f$ is solved with fixed nodes at the boundaries and a unit load. In a first step, we create a rectangular mesh, initialize an instance of a linear quad element and the appropriate quadrature. A numeric region is created with all three objects.

```python
import numpy as np
import felupe as fe

n = 51
mesh = fe.mesh.Rectangle(n=(n, n))
element = fe.element.Quad1()
quadrature = fe.quadrature.Linear(2)

region = fe.Region(mesh, element, quadrature)
```

![poisson mesh](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson_mesh.png)

In the next step we create a 2d-field for our 2d-region and calculate the gradient of the unknowns w.r.t. to the undeformed coordinates. Both the laplace equation and the right-hand-side of the poisson problem are transformed from the local differential into the (weak) integral form as follows. We start with the local differential form.

$$\Delta \bm{u} + \bm{f} = 0$$

Multiplying the above equation with $\delta \bm{u}$ and integrate it in the region $\Omega$.

$$\int_\Omega \Delta \bm{u} \cdot \delta \bm{u} \ d\Omega + \int_\Omega \bm{f} \cdot \delta \bm{u} \ d\Omega = 0$$

Next we do integration by parts to obtain the (weak) integral form. Note: We don't care about the boundary terms as they are all zero in this example.

$$-\int_\Omega \frac{\partial \delta \bm{u}}{\partial \bm{X}} : \frac{\partial \bm{u}}{\partial \bm{X}} \ d\Omega + \int_\Omega \bm{f} \cdot \delta \bm{u} \ d\Omega = 0$$

We now have the integral form of the laplace equation

$$\int_\Omega \frac{\partial \delta \bm{u}}{\partial \bm{X}} : \frac{\partial \bm{u}}{\partial \bm{X}} \ d\Omega = \int_\Omega \frac{\partial \delta \bm{u}}{\partial \bm{X}} : \bm{I} \overset{ik}{\odot} \bm{I} : \frac{\partial \bm{u}}{\partial \bm{X}}  \ d\Omega$$

and the right-hand-side at hand.

$$-\int_\Omega \bm{f} \cdot \delta \bm{u} \ d\Omega$$

```python
field = fe.Field(region, dim=2, values=0)
dudX = field.grad()

eye = fe.math.identity(dudX)
poisson = fe.math.cdya_ik(eye, eye)
rhs = -np.ones_like(field.interpolate())

bilinearform = fe.IntegralForm(poisson, field, region.dV, field, 
                               grad_v=True, grad_u=True)

linearform = fe.IntegralForm(rhs, field, region.dV)
```

The assembly of the two forms give the stiffness matrix $\bm{K}$ and the right-hand-side $\bm{r}$.

```python
K = bilinearform.assemble()
r = linearform.assemble()
```

As already mentioned all boundary nodes are fixed.

```python
f0, f1 = lambda x: np.isclose(x, 0), lambda x: np.isclose(x, 1)

boundaries = {}
boundaries["left"  ] = fe.Boundary(field, fx=f0)
boundaries["right" ] = fe.Boundary(field, fx=f1)
boundaries["bottom"] = fe.Boundary(field, fy=f0)
boundaries["top"   ] = fe.Boundary(field, fy=f1)

dof0, dof1 = fe.doftools.partition(field, boundaries)
unknowns0_ext = fe.doftools.apply(field, boundaries, dof0)
```

We are now able to solve our system. Recall from the [Getting Started](quickstart.md) section that FElupe needs the assembled stiffness matrix and nodal residuals from a linearization of nonlinear equilibrium equations $\bm{r}(\bm{u})$.

$$\bm{r}(\bm{u}) = \bm{0}$$

$$\bm{r} + \bm{K} d \bm{u} = \bm{0} \qquad \text{with} \qquad \bm{K} = \frac{\partial \bm{r}}{\partial \bm{u}}$$

$$\bm{K} d \bm{u} = -\bm{r}$$

```python
system = fe.solve.partition(field, K, dof1, dof0, r)
d_unknowns = fe.solve.solve(*system, unknowns0_ext)
field += d_unknowns

fe.utils.save(region, field, filename="poisson.vtk")
```

The solution of the first component of the nodal unknowns is first visualized by the VTK output with the help of Paraview.

![poisson solution](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson_solution.png)

Another possibility would be a plot with matplotlib.

```python

import matplotlib.pyplot as plt
cf = plt.contourf(
    mesh.nodes[:,0].reshape(n, n), 
    mesh.nodes[:,1].reshape(n, n), 
    field.values[:,0].reshape(n, n),
)
plt.gca().set_aspect("equal")
plt.gcf().colorbar(cf, ax=plt.gca(), shrink=0.9)

plt.savefig("poisson.svg")
```

![poisson plot](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson.svg)