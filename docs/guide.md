# Theory Guide
This section gives an overview of some selected topics of the theory behing felupe.

## Constitution
Stresses are calculated by a constitutive material law which is a function of the stress - work-conjugate deformation quantity. Felupe provides several template classes which may be used for user-defined materials. E.g. `InvariantBased`- and `PrincipalStretchBased` classes are built on top of a user material function (`umat`) for isotropic hyperelastic material formulations. This `umat` - function takes a list of invariants as input and provides first and second partial derivatives of the strain energy density function w.r.t. the invariants as output for the case of `InvariantBased` materials. These materials may be optionally wrapped in an `AsIsochoric` class. An additional hydrostatic volumetric material behavior ($p = K (J-1)$) is provided by a `Hydrostatic` material class which takes the bulk modulus as its only argument. A list of materials may be combined by a `Composite` material, e.g. the isochoric and volumetric parts of a material or parallel contributions from a single formula architecture with different sets of material parameters. Finally the resulting material has to be converted to the work-conjugate pair of the first Piola-Kirchhoff stress tensor and the (virtual) deformation gradient by hand or with the template class `Material`. If one wishes to use a mixed-field formulation for nearly-incompressible materials (see section below), the resulting stress and elasticity functions have to be provided to the `fe.constitution.GeneralizedThreeField` class.

The described classes (*Total Lagrange*) are visualized in the following Figure.

![constitution tl](https://raw.githubusercontent.com/adtzlr/felupe/main/docs/images/constitution_tl.png)

For example we define a simple Neo-Hookean solid with an invariant-based `umat_invariants` function.

```python
import numpy as np

def umat_invariants(invariants):
    """Calculate first (W_a) and second (W_ab) partial derivatives 
    of the strain energy density function w.r.t. the invariants."""
    
    # header section (don't change)
    # --------------------------------
    I1, I2, I3 = invariants

    W_a  = np.zeros((3, *I1.shape))
    W_ab = np.zeros((3, 3, *I1.shape))
    # --------------------------------
    
    # user code
    # --------------------------------
    mu = 1.0
    W_a[0] = mu

    return W_a, W_ab
```

This umat is passed as described above to an instance of an `InvariantBased` (*Total Lagrange*) material.

```python
import felupe as fe

neohooke_iso = fe.constitution.InvariantBased(umat_invariants)
neohooke_dev = fe.constitution.AsIsochoric(neohooke_iso)
neohooke_vol = fe.constitution.Hydrostatic(bulk=20.0)
neohooke     = fe.constitution.Composite(neohooke_dev, neohooke_vol)

mat = fe.constitution.Material(neohooke)
# use `mat` as
# mat.P(F) and # mat.A(F)

# three-field-variation (u,p,J)
#mat_upJ = fe.constitution.GeneralizedThreeField(mat.P, mat.A)

# use `mat_upJ` as
# mat_upJ.f(F, p, J) and # mat_upJ.A(F, p, J)
```

A template function for principal stretch based materials is defined in a similar way.

```python
import numpy as np

def umat_stretches(stretches):
    """Calculate first (W_a) and second (W_ab) partial 
    derivatives of the strain energy density function 
    w.r.t. the principal stretches."""

    # header section (don't change)
    # -------------------------------------------
    # get shape
    ndim, ngauss, nelems = stretches.shape
    diag = np.arange(ndim), np.arange(ndim)

    W_a  = np.zeros((ndim, ngauss, nelems))
    W_ab = np.zeros((ndim, ndim, ngauss, nelems))
    # -------------------------------------------
    
    # user code
    # -------------------------------------------
    mu = 1.0
    k = 0.7
    
    W_a = mu * stretches ** (k - 1)
    W_ab[diag] = mu * (k - 1) * stretches ** (k - 2)

    return W_a, W_ab
```

```python
import felupe as fe

ogden_iso = fe.constitution.PrincipalStretchBased(umat_stretches)
ogden_dev = fe.constitution.AsIsochoric(ogden_iso)
ogden_vol = fe.constitution.Hydrostatic(bulk=20.0)
ogden     = fe.constitution.Composite(ogden_dev, ogden_vol)

mat = fe.constitution.Material(ogden)
```

### Mixed-field formulations
FElupe supports mixed-field formulations in a similar way it can handle (default) single-field variations. The definition of a mixed-field variation is shown for the hydrostatic-volumetric selective three-field-variation with independend fields for displacements $\bm{u}$, pressure $p$ and volume ratio $J$. The total potential energy for nearly-incompressible hyperelasticity is formulated with a determinant-modified deformation gradient. Pressure and Volume ratio fields should be kept one order lower than the interpolation order of the displacement field, i.e. linear displacement fields should be paired with cellwise-constant (mean) values of pressure and volume ratio.

### Total potential energy: variation and linearization
The total potential energy of internal forces is defined with a strain energy density function in terms of a determinant-modified deformation gradient and an additional control equation.

$$\Pi = \Pi_{int} + \Pi_{ext}$$

$$\Pi_{int} = \int_V \psi(\bm{F}) \ dV \qquad \rightarrow \qquad \Pi_{int}(\bm{u},p,J) = \int_V \psi(\overline{\bm{F}}) \ dV + \int_V p (J-\overline{J}) \ dV$$

$$\overline{\bm{F}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \bm{F}$$

The variations of the total potential energy w.r.t. $(\bm{u},p,J)$ lead to the following expressions. We denote first partial derivatives as $\bm{f}_{(\bullet)}$ and second partial derivatives as $\bm{A}_{(\bullet,\bullet)}$.

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

$$\mathbb{A}_{\bm{u},\bm{u}} =  \overline{\overline{\mathbb{A}}} + \frac{1}{9} \left(  \bm{F} : \overline{\overline{\mathbb{A}}} : \bm{F} \right) \bm{F}^{-T} \otimes \bm{F}^{-T} - \frac{1}{3} \left( \bm{F}^{-T} \otimes \left( \overline{\overline{\bm{P}}} + \bm{F} : \overline{\overline{\mathbb{A}}} \right) + \left( \overline{\overline{\bm{P}}} + \overline{\overline{\mathbb{A}}} : \bm{F} \right) \otimes \bm{F}^{-T} \right)$$

$$+\left( p J + \frac{1}{9} \overline{\overline{\bm{P}}} : \bm{F} \right) \bm{F}^{-T} \otimes \bm{F}^{-T} - \left( p J - \frac{1}{3} \overline{\overline{\bm{P}}} : \bm{F} \right) \bm{F}^{-T} \overset{il}{\odot} \bm{F}^{-T} $$

$$A_{p,p} = 0 $$

$$A_{\overline{J},\overline{J}} = \frac{1}{9 \overline{J}^2} \left( \bm{F} : \overline{\overline{\mathbb{A}}} : \bm{F} \right) - 2 \left( \overline{\overline{\bm{P}}} : \bm{F} \right) $$

$$\bm{A}_{\bm{u},p} = \bm{A}_{p, \bm{u}} = J \bm{F}^{-T} $$

$$\bm{A}_{\bm{u},\overline{J}} = \bm{A}_{\overline{J}, \bm{u}} = \frac{1}{3 \overline{J}} \left( \bm{P}' + \bm{F} : \overline{\overline{\mathbb{A}}} - \frac{1}{3} \left( \bm{F} : \overline{\overline{\mathbb{A}}} : \bm{F} \right) \bm{F}^{-T} \right) $$

$$A_{p,\overline{J}} = A_{\overline{J}, p} = -1 $$

with $$ \overline{\overline{\mathbb{A}}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \frac{\partial^2 \psi}{\partial \overline{\bm{F}} \partial \overline{\bm{F}}} \left(\frac{\overline{J}}{J}\right)^{1/3}$$

as well as

$$\bm{P}' = \bm{P} - p J \bm{F}^{-T}$$

## Axisymmetric Analysis
Axisymmetric scenarios are modeled with a 2D-mesh and a 2D element formulation. The rotation axis is chosen along the global X-axis $(X,Y,Z) \widehat{=} (Z,R,\varphi)$. The 3x3 deformation gradient consists of an in-plane 2x2 sub-matrix and one additional entry for the out-of-plane stretch which is equal to the ratio of deformed and undeformed radius.

$$\bm{F} = \begin{bmatrix} \bm{F}_{(2D)} & \bm{0} \\ \bm{0}^T & \frac{r}{R} \end{bmatrix}$$

The variation of the deformation gradient consists of both in- and out-of-plane contributions.

$$\delta \bm{F}_{(2D)} = \delta \frac{\partial \bm{u}}{\partial \bm{X}} \qquad \text{and} \qquad \delta \left(\frac{r}{R}\right) = \frac{\delta u_r}{R}$$

Again, the internal virtual work leads to two seperate terms.
$$-\delta W_{int} = \int_V \bm{P} : \delta \bm{F} \ dV = \int_V \bm{P}_{(2D)} : \delta \bm{F}_{(2D)} \ dV + \int_V \frac{P_{33}}{R} : \delta u_r \ dV$$

The differential volume is further expressed as a product of the differential in-plane area and the differential arc length. The arc length integral is finally pre-evaluated.

$$\int_V dV = \int_{\varphi=0}^{2\pi} \int_A R\ dA\ d\varphi = 2\pi \int_A R\ dA $$

Inserting the differential volume integral into the expression of internal virtual work, this leads to:

$$-\delta W_{int} = 2\pi \int_A \bm{P}_{(2D)} : \delta \bm{F}_{(2D)} \ R \ dA + 2\pi \int_A P_{33} : \delta u_r \ dA$$

A Linearization of the internal virtual work expression gives four terms.

$$-\Delta \delta W_{int} = \Delta_{(2D)} \delta_{(2D)} W_{int} + \Delta_{33} \delta_{(2D)} W_{int} + \Delta_{(2D)} \delta_{33} W_{int} + \Delta_{33} \delta_{33} W_{int}$$

$$-\Delta_{(2D)} \delta_{(2D)} W_{int} = 2\pi \int_A \delta \bm{F}_{(2D)} : \mathbb{A}_{(2D),(2D)} : \Delta \bm{F}_{(2D)} \ R \ dA$$

$$-\Delta_{33} \delta_{(2D)} W_{int} = 2\pi \int_A \delta \bm{F}_{(2D)} : \mathbb{A}_{(2D),33} : \Delta u_r \ dA$$

$$-\Delta_{(2D)} \delta_{33} W_{int} = 2\pi \int_A \delta u_r : \mathbb{A}_{33,(2D)} : \Delta \bm{F}_{(2D)} \ dA$$

$$-\Delta_{33} \delta_{33} W_{int} = 2\pi \int_A \delta u_r : \frac{\mathbb{A}_{33,33}}{R} : \Delta u_r \ dA$$

whith $\mathbb{A}_{(2D),(2D)} = \frac{\partial \psi}{\partial \bm{F}_{(2D)} \partial \bm{F}_{(2D)}}$, $\mathbb{A}_{(2D),33} = \mathbb{A}_{33,(2D)} = \frac{\partial \psi}{\partial \bm{F}_{(2D)} \partial F^3_{\hphantom{3}3}}$ and $\mathbb{A}_{33,33} = \frac{\partial \psi}{F^3_{\hphantom{3}3} \partial F^3_{\hphantom{3}3}}$ or in Python code notation:

```python
# python is zero-indexed `i = (0,1,2)`
# `:2` represents a slice of the first two components `(0,1)`
A_2d2d = A[:2,:2,:2,:2]
A_2d33 = A[:2,:2, 2, 2]
A_332d = A[ 2, 2,:2,:2]
A_3333 = A[ 2, 2, 2, 2]
```

### Felupe implementation
For axisymmetric analyses an axisymmetric vector-valued field has to be created for the in-plane displacements.

```python
import felupe as fe

mesh = fe.mesh.Rectangle(n=3)
element = fe.element.Quad()
quadrature = fe.quadrature.GaussLegendre(order=1, dim=2)

region  = fe.Region(mesh, element, quadrature)
dA = region.dV
```

```python
u  = fe.FieldAxisymmetric(region)
```

Now it gets important: The 3x3 deformation gradient for axisymmetric problems is obtained with the default `grad` method. For instances of `FieldAxisymmetric` this returns the full 3x3 gradient as described above.

```python
H = fe.math.grad(u)
F = fe.math.identity(H) + H
```

For simplicity, let's assume a (built-in) Neo-Hookean material.

```python
umat = fe.constitution.NeoHooke(mu=1, bulk=5)
```

Felupe provides an adopted IntegralForm class for the integration and the sparse matrix assemblage of axisymmetric problems. It uses the additional information (e.g. radial coordinates at integration points) stored in `FieldAxisymmetric` to provide a consistent interface in comparison to classic IntegralForms.

```python

r = fe.IntegralFormAxisymmetric(umat.P(F), u, dA).assemble()
K = fe.IntegralFormAxisymmetric(umat.A(F), u, dA).assemble()
```

To sum up, for axisymmetric problems use `FieldAxisymmetric` in conjunction with `IntegralFormAxisymmetric`.

## Supported Finite Elements
FElupe supports lagrangian line, quad and hexaeder elements with arbitrary order polynomial basis functions. However, FElupe's mesh generation module is designed for linear and constant order elements only. For simulations with arbitrary order elements a user-defined mesh has to be provided.

| Description | Order | Nodes | Mesh module compatible | save as VTK |
|---|---|---|---|---|
| Constant Hexahedron | 0 | 1 | yes | yes |
| Linear Hexahedron | 1 | 8 | yes | yes |
| Quadratic Serendipity Hexahedron | 2 | 20 | no | yes |
| Lagrange Hexahedron | $p$ | $(p+1)^3$ | no | yes |

## Arbitrary Order Lagrange Elements
A complete lagrange polynomial $h(r)$ of order $p$ results from a sum of products of curve parameters $a_i$ associated to the normalized i-th power of a variable $\frac{r^i}{i!}$. For the determination of the curve parameters $a_i$ a total number of $n=p+1$ evaluation points are necessary.

$$h(r) = \sum_{i=0}^p a_i \frac{r^i}{i!} = \bm{a}^T \bm{r}(r)$$

### Polynomial basis functions
The basis function vector is generated with row-stacking of the individual lagrange polynomials. Each polynomial defined in the interval $[-1,1]$ is a function of the parameter $r$. The curve parameters matrix $\bm{A}$ is of symmetric shape due to the fact that for each evaluation point $r_j$ exactly one basis function $h_j(r)$ is needed.

$$\bm{h}(r) = \bm{A}^T \bm{r}(r)$$

### Curve parameter matrix
The evaluation of the curve parameter matrix $\bm{A}$ is carried out by boundary conditions. Each basis function $h_i$ has to take the value of one at the associated nodal coordinate $r_i$ and zero at all other nodal coordinates.

$$\bm{A}^T \bm{R} = \bm{I} \qquad \text{with} \qquad \bm{R} = \begin{bmatrix}\bm{r}(r_1) & \bm{r}(r_2) & \dots & \bm{r}(r_p)\end{bmatrix}$$

$$\bm{A}^T = \bm{R}^{-1}$$

### Interpolation and partial derivatives
The approximation of nodal unknowns $\hat{\bm{u}}$ as a function of the parameter $r$ is evaluated as

$$\bm{u}(r) \approx \hat{\bm{u}}^T \bm{h}(r)$$

For the calculation of the partial derivative of the interpolation field w.r.t. the parameter $r$ a simple shift of the entries of the parameter vector is enough. This shifted parameter vector is denoted as $\bm{r}^-$. A minus superscript indices the negative shift of the vector entries by $-1$.

$$\frac{\partial \bm{u}(r)}{\partial r} \approx \hat{\bm{u}}^T \frac{\partial \bm{h}(r)}{\partial r}$$

$$\frac{\partial \bm{h}(r)}{\partial r} = \bm{A}^T \bm{r}^-(r) \qquad \text{with} \qquad r_0^- = 0 \qquad \text{and} \qquad r_i^- = \frac{r^{(i-1)}}{(i-1)!} \qquad \text{for} \qquad  i=(1 \dots p)$$

### n-dimensional basis functions
Multi-dimensional basis function matrices $\bm{H}_{2D}, \bm{H}_{3D}$ are simply evaluated as dyadic (outer) vector products of one-dimensional basis function vectors. The multi-dimensional basis function vector is a one-dimensional representation (flattened version) of the multi-dimensional basis function matrix.

$$ \bm{H}_{2D}(r,s) = \bm{h}(r) \otimes \bm{h}(s)$$

$$ \bm{H}_{3D}(r,s,t) = \bm{h}(r) \otimes \bm{h}(s) \otimes \bm{h}(t)$$

## Point numbering of linear hexahedrons
A linear hexahedron has points arranged in a way that the connectivity starts at `[-1,-1,-1]` where points are connected counterclockwise at`z=-1` in a first step. Finally the same applies for `z=1`. For arbitrary order elements the first axis goes from `-1` to `1`, then the second axis  from `-1` to `1` and finally the third axis  from `-1` to `1`. This is due to VTK compatibility.

```python
# Linear hexahedron with eight points

points = np.array([
    [-1,-1,-1], #0
    [ 1,-1,-1], #1
    [ 1, 1,-1], #2
    [-1, 1,-1], #3
    [-1,-1, 1], #4
    [ 1,-1, 1], #5
    [ 1, 1, 1], #6
    [-1, 1, 1], #7
])

cells = np.array([[0,1,2,3,4,5,6,7]])
```