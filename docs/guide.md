# Theory Guide

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

### Node numbering
A linear hexahedron has nodes arranged in a way that the connectivity starts at `[-1,-1,-1]` where nodes are connected counterclockwise at`z=-1` in a first step. Finally the same applies for `z=1`. For arbitrary order elements the first axis goes from `-1` to `1`, then the second axis  from `-1` to `1` and finally the third axis  from `-1` to `1`.

```python
# Linear hexahedron with eight nodes

nodes = np.array([
    [-1,-1,-1], #0
    [ 1,-1,-1], #1
    [ 1, 1,-1], #2
    [-1, 1,-1], #3
    [-1,-1, 1], #4
    [ 1,-1, 1], #5
    [ 1, 1, 1], #6
    [-1, 1, 1], #7
])

connectivity = np.array([[0,1,2,3,4,5,6,7]])
```