# Examples

## Example 1 - Poisson equation
The poisson problem is solved with fixed nodes at the boundaries and a unit load.

![poisson mesh](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson_mesh.png)

![poisson solution](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson_solution.png)

![poisson plot](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/poisson.svg)

```python
import numpy as np
import felupe as fe

n = 51
mesh = fe.mesh.Rectangle(n=(n, n))
element = fe.element.Quad1()
quadrature = fe.quadrature.Linear(2)

region = fe.Region(mesh, element, quadrature)

field = fe.Field(region, dim=2, values=0)
dudX = field.grad()

eye = fe.math.identity(dudX)
poisson = fe.math.cdya_ik(eye, eye)
rhs = -np.ones_like(field.interpolate())

bilinearform = fe.IntegralForm(poisson, field, region.dV, field, 
                               grad_v=True, grad_u=True)

linearform = fe.IntegralForm(rhs, field, region.dV)

A = bilinearform.assemble()
b = linearform.assemble()

f0, f1 = lambda x: np.isclose(x, 0), lambda x: np.isclose(x, 1)

boundaries = {}
boundaries["left"  ] = fe.Boundary(field, fx=f0)
boundaries["right" ] = fe.Boundary(field, fx=f1)
boundaries["bottom"] = fe.Boundary(field, fy=f0)
boundaries["top"   ] = fe.Boundary(field, fy=f1)

dof0, dof1 = fe.doftools.partition(field, boundaries)
unknowns0_ext = fe.doftools.apply(field, boundaries, dof0)

system = fe.solve.partition(field, A, dof1, dof0, b)
d_unknowns = fe.solve.solve(*system, unknowns0_ext)
field += d_unknowns

fe.utils.save(region, field, filename="poisson.vtk")

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