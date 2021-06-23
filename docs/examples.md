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

In the next step we create a 2d-field for our 2d-region and calculate the gradient of the unknowns w.r.t. to the undeformed coordinates. 

```python
field = fe.Field(region, dim=2, values=0)
dudX = field.grad()
```

Both the laplace equation and the right-hand-side of the poisson problem are transformed from the local differential into the (weak) integral form as follows. We start with the local differential form.

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

## Example 2 - Mixed-field variation for nearly-incompressible hyperelasticity
Felupe supports mixed-field formulations in a similar way it can handle (default) single-field variations. The definition of a mixed-field variation is shown for the hydrostatic-volumetric selective three-field-variation with independend fields for displacements $\bm{u}$, pressure $p$ and volume ratio $J$. The total potential energy for nearly-incompressible hyperelasticity is formulated with a determinant-modified deformation gradient.

### Total potential energy: variation and linearization

$$\Pi = \Pi_{int} + \Pi_{ext}$$

$$\Pi_{int} = \int_V \psi(\bm{F}) \ dV \qquad \rightarrow \qquad \Pi_{int}(\bm{u},p,J) = \int_V \psi(\overline{\bm{F}}) \ dV + \int_V p (J-\overline{J}) \ dV$$

$$\overline{\bm{F}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \bm{F}$$

The variations of the total potential energy w.r.t. $(\bm{u},p,J)$ lead to the following expressions.

$$\delta_{\bm{u}} \Pi_{int} = \int_V \bm{f}_{\bm{u}} : \delta \bm{F} \ dV = \int_V \left( \frac{\partial \psi}{\partial \overline{\bm{F}}} : \frac{\partial \overline{\bm{F}}}{\partial \bm{F}} + p J \bm{F}^{-T} \right) : \delta \bm{F} \ dV$$

$$\delta_{p} \Pi_{int} = \int_V f_{p} \ \delta p \ dV = \int_V (J - \overline{J}) \ \delta p \ dV$$

$$\delta_{\overline{J}} \Pi_{int} = \int_V f_{\overline{J}} \ \delta \overline{J} \ dV = \int_V \left( \frac{\partial \psi}{\partial \overline{\bm{F}}} : \frac{\partial \overline{\bm{F}}}{\partial \overline{J}} - p \right) : \delta \overline{J} \ dV$$

The projection tensors from the variations lead the following results.

$$\frac{\partial \overline{\bm{F}}}{\partial \bm{F}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \left( \bm{I} \overset{ik}{\odot} \bm{I} - \frac{1}{3} \bm{F} \otimes \bm{F}^{-T} \right)$$

$$\frac{\partial \overline{\bm{F}}}{\partial \overline{J}} = \frac{1}{3 \overline{J}} \overline{\bm{F}}$$

The double-dot products from the variations are now evaluated.

$$\overline{\bm{P}} = \frac{\partial \psi}{\partial \overline{\bm{F}}} = \overline{\overline{\bm{P}}} - \frac{1}{3} \left(  \overline{\overline{\bm{P}}} : \bm{F} \right) \bm{F}^{-T} \qquad \text{with} \qquad \overline{\overline{\bm{P}}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \frac{\partial \psi}{\partial \overline{\bm{F}}}$$

$$\frac{\partial \psi}{\partial \overline{\bm{F}}} : \frac{1}{3 \overline{J}} \overline{\bm{F}} = \frac{1}{3 \overline{J}} \overline{\overline{\bm{P}}} : \bm{F}$$

We now have three formulas; one for the first Piola Kirchhoff stress and two additional control equations.

$$\bm{f}_{\bm{u}} (= \bm{P}) = \overline{\overline{\bm{P}}} - \frac{1}{3} \left(  \overline{\overline{\bm{P}}} : \bm{F} \right) \bm{F}^{-T} $$

$$f_p = J - \overline{J} $$

$$f_{\overline{J}} =  \frac{1}{3 \overline{J}} \left( \overline{\overline{\bm{P}}} : \bm{F} \right) - p$$

A linearization of the above formulas gives six equations (only results are given here).

$$\mathbb{A}_{\bm{u},\bm{u}} =  \mathbb{P} : \overline{\overline{\mathbb{A}}} : \mathbb{P}^T - \frac{1}{3} \left( \bm{F}^{-T} \otimes \overline{\overline{\bm{P}}} + \overline{\overline{\bm{P}}} \otimes \bm{F}^{-T} \right) + p J \left( \bm{F}^{-T} \otimes \bm{F}^{-T} - \bm{F}^{-T} \overset{il}{\odot} \bm{F}^{-T} \right) $$

$$\bm{A}_{\bm{u},p} = J \bm{F}^{-T} $$

$$\bm{A}_{\bm{u},\overline{J}} = \frac{1}{3 \overline{J}} \left( \bm{P}' + \bm{F} : \overline{\overline{\mathbb{A}}} - \frac{1}{3} \left( \bm{F} : \overline{\overline{\mathbb{A}}} : \bm{F} \right) \bm{F}^{-T} \right) $$

$$A_{p,\overline{J}} = -1 $$

$$A_{p,p} = 0 $$

$$A_{\overline{J},\overline{J}} = \frac{1}{9 \overline{J}^2} \left( \bm{F} : \overline{\overline{\mathbb{A}}} : \bm{F} \right) - 2 \left( \overline{\overline{\bm{P}}} : \bm{F} \right) $$

with $$ \overline{\overline{\mathbb{A}}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \frac{\partial^2 \psi}{\partial \overline{\bm{F}} \partial \overline{\bm{F}}} \left(\frac{\overline{J}}{J}\right)^{1/3}$$

as well as

$$\bm{P}' = \bm{P} - p J \bm{F}^{-T}$$

Finally... done with theory. :champagne: :clinking_glasses: :champagne: :clinking_glasses: :champagne:

### FElupe implementation
We take the [Getting Started](quickstart.md) example and modify it accordingly. First, we have to implement the above equations as `GeneralizedMixedField`.

```python
import numpy as np
import felupe as fe
import felupe.math as fm

class GeneralizedMixedField:
    def __init__(self, P, A):
        self.fun_P = P
        self.fun_A = A

    def f_u(self, F, p, J):
        """Variation of total potential w.r.t displacements
        (1st Piola Kirchhoff stress).

        δ_u(Π_int) = ∫_V (∂ψ/∂F + p cof(F)) : δF dV
        """

        return self.Pbb - self.PbbF / 3 * self.iFT + p * self.detF * self.iFT

    def f_p(self, F, p, J):
        """Variation of total potential energy w.r.t pressure.

        δ_p(Π_int) = ∫_V (det(F) - J) δp dV
        """

        return self.detF - J

    def f_J(self, F, p, J):
        """Variation of total potential energy w.r.t volume ratio.

        δ_J(Π_int) = ∫_V (∂U/∂J - p) δJ dV
        """

        return fm.ddot(self.Pbb, F) / (3 * J) - p

    def f(self, F, p, J):
        """List of variations of total potential energy w.r.t
        displacements, pressure and volume ratio."""
        self.detF = fm.det(F)
        self.iFT = fm.transpose(fm.inv(F))
        self.Fb = (J / self.detF) ** (1 / 3) * F
        self.Pb = self.fun_P(self.Fb)
        self.Pbb = (J / self.detF) ** (1 / 3) * self.Pb
        self.PbbF = fm.ddot(self.Pbb, F)

        return [self.f_u(F, p, J), self.f_p(F, p, J), self.f_J(F, p, J)]

    def A(self, F, p, J):
        """List of linearized variations of total potential energy w.r.t
        displacements, pressure and volume ratio (these expressions are
        symmetric; A_up = A_pu if derived from a total potential energy
        formulation). List entries have to be arranged as a flattened list
        from the upper triangle blocks:

        [[0 1 2],
         [  3 4],
         [    5]] --> [0 1 2 3 4 5]

        """
        self.detF = fm.det(F)
        self.iFT = fm.transpose(fm.inv(F))
        self.Fb = (J / self.detF) ** (1 / 3) * F
        self.Pbb = (J / self.detF) ** (1 / 3) * self.fun_P(self.Fb)

        self.eye = fm.identity(F)
        self.P4 = fm.cdya_ik(self.eye, self.eye) - 1 / 3 * fm.dya(F, self.iFT)
        self.A4b = self.fun_A(self.Fb)
        self.A4bb = (J / self.detF) ** (2 / 3) * self.A4b

        self.PbbF = fm.ddot(self.Pbb, F)
        self.FA4bbF = fm.ddot(fm.ddot(F, self.A4bb), F)

        return [
            self.A_uu(F, p, J),
            self.A_up(F, p, J),
            self.A_uJ(F, p, J),
            self.A_pp(F, p, J),
            self.A_pJ(F, p, J),
            self.A_JJ(F, p, J),
        ]

    def A_uu(self, F, p=None, J=None):
        """Linearization w.r.t. displacements of variation of
        total potential energy w.r.t displacements.

        Δ_u(δ_u(Π_int)) = ∫_V δF : (∂²ψ/(∂F∂F) + p ∂cof(F)/∂F) : ΔF dV

        """

        A4 = (
            self.A4bb
            + self.FA4bbF * fm.dya(self.iFT, self.iFT) / 9
            - (fm.dya(fm.ddot(self.A4bb, F), self.iFT) + fm.dya(self.iFT, fm.ddot(F, self.A4bb)))
            / 3
            - (fm.dya(self.Pbb, self.iFT) + fm.dya(self.iFT, self.Pbb)) / 3
            + self.PbbF
            / 3
            * (fm.cdya_il(self.iFT, self.iFT) + fm.dya(self.iFT, self.iFT) / 3)
            + p * self.detF * (fm.dya(self.iFT, self.iFT) - fm.cdya_il(self.iFT, self.iFT))
        )

        return A4

    def A_pp(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t pressure.

        Δ_p(δ_p(Π_int)) = ∫_V δp 0 Δp dV

        """
        return np.zeros_like(p)

    def A_JJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t volume ratio.

        Δ_J(δ_J(Π_int)) = ∫_V δJ ∂²ψ/(∂J∂J) ΔJ dV

        """

        return (self.FA4bbF - 2 * self.PbbF) / (9 * J ** 2)

    def A_up(self, F, p, J):
        """Linearization w.r.t. pressure of variation of
        total potential energy w.r.t displacements.

        Δ_p(δ_u(Π_int)) = ∫_V δF : J cof(F) Δp dV

        """

        return self.detF * self.iFT

    def A_uJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t displacements.

        Δ_J(δ_u(Π_int)) = ∫_V δF :  ∂²ψ/(∂F∂J) ΔJ dV

        """

        P = self.f_u(F, 0 * p, J)
        return (-self.FA4bbF / 3 * self.iFT + P + fm.ddot(F, self.A4bb)) / (3 * J)

    def A_pJ(self, F, p, J):
        """Linearization w.r.t. volume ratio of variation of
        total potential energy w.r.t pressure.

        Δ_J(δ_p(Π_int)) = ∫_V δp (-1) ΔJ dV

        """
        return -np.ones_like(J)
```

Next, we use the built in Neo-Hookean material model and use it for our `GeneralizedMixedField` class.

```python
neohooke = fe.constitution.NeoHooke(mu=1.0, bulk=5000.0)
umat = GeneralizedMixedField(neohooke.P, neohooke.A)
```

We create a meshed cube and a converted version for the elementwise constant fields. Two element definitions are necessary: one for the displacements and one for the pressure and volume ratio. Both elements use the same quadrature rule. Two regions are created, which will be further used by the creation of the fields.


```python
mesh  = fe.mesh.Cube(n=6)
mesh0 = fe.mesh.convert(mesh)

element0 = fe.element.Hex0()
element1 = fe.element.Hex1()
quadrature = fe.quadrature.Linear(dim=3)

region0 = fe.Region(mesh0, element0, quadrature)
region  = fe.Region(mesh,  element1, quadrature)

dV = region.dV

displacement = fe.Field(region,  dim=3)
pressure     = fe.Field(region0, dim=1)
volumeratio  = fe.Field(region0, dim=1, values=1)
fields = (displacement, pressure, volumeratio)
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

    r =   linearform.assemble().toarray()[:, 0]
    K = bilinearform.assemble()
    
    system = fe.solve.partition(fields, K, dof1, dof0, r)
    dfields = np.split(fe.solve.solve(*system, u0ext), unstack)
    
    for field, dfield in zip(fields, dfields):
        field += dfield

    norm = np.linalg.norm(dfields[0])
    print(iteration, norm)

    if norm < 1e-12:
        break

fe.utils.save(region, fields, unstack=unstack, filename="result.vtk")
```

The deformed cube is visualized by the VTK output with the help of Paraview.

![deformed cube](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/threefield_cube.png)