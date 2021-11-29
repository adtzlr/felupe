Polynomial shape functions
--------------------------

The basis function vector is generated with row-stacking of the individual lagrange polynomials. Each polynomial defined in the interval :math:`[-1,1]` is a function of the parameter :math:`r`. The curve parameters matrix :math:`\boldsymbol{A}` is of symmetric shape due to the fact that for each evaluation point :math:`r_j` exactly one basis function :math:`h_j(r)` is needed.

..  math::

    \boldsymbol{h}(r) = \boldsymbol{A}^T \boldsymbol{r}(r)


Curve parameter matrix
**********************

The evaluation of the curve parameter matrix :math:`\boldsymbol{A}` is carried out by boundary conditions. Each shape function :math:`h_i` has to take the value of one at the associated nodal coordinate :math:`r_i` and zero at all other nodal coordinates.

..  math::

    \boldsymbol{A}^T \boldsymbol{R} &= \boldsymbol{I} \qquad \text{with} \qquad \boldsymbol{R} = \begin{bmatrix}\boldsymbol{r}(r_1) & \boldsymbol{r}(r_2) & \dots & \boldsymbol{r}(r_p)\end{bmatrix}

    \boldsymbol{A}^T &= \boldsymbol{R}^{-1}


Interpolation and partial derivatives
*************************************

The approximation of nodal unknowns :math:`\hat{\boldsymbol{u}}` as a function of the parameter :math:`r` is evaluated as

..  math::

    \boldsymbol{u}(r) \approx \hat{\boldsymbol{u}}^T \boldsymbol{h}(r)

For the calculation of the partial derivative of the interpolation field w.r.t. the parameter :math:`r` a simple shift of the entries of the parameter vector is enough. This shifted parameter vector is denoted as :math:`\boldsymbol{r}^-`. A minus superscript indices the negative shift of the vector entries by :math:`-1`.

..  math::

    \frac{\partial \boldsymbol{u}(r)}{\partial r} &\approx \hat{\boldsymbol{u}}^T \frac{\partial \boldsymbol{h}(r)}{\partial r}

    \frac{\partial \boldsymbol{h}(r)}{\partial r} &= \boldsymbol{A}^T \boldsymbol{r}^-(r) \qquad \text{with} \qquad r_0^- = 0 \qquad \text{and} \qquad r_i^- = \frac{r^{(i-1)}}{(i-1)!} \qquad \text{for} \qquad  i=(1 \dots p)


n-dimensional shape functions
*****************************

Multi-dimensional shape function matrices :math:`\boldsymbol{H}_{2D}, \boldsymbol{H}_{3D}` are simply evaluated as dyadic (outer) vector products of one-dimensional shape function vectors. The multi-dimensional shape function vector is a one-dimensional representation (flattened version) of the multi-dimensional shape function matrix.

..  math::

    \boldsymbol{H}_{2D}(r,s) &= \boldsymbol{h}(r) \otimes \boldsymbol{h}(s)

    \boldsymbol{H}_{3D}(r,s,t) &= \boldsymbol{h}(r) \otimes \boldsymbol{h}(s) \otimes \boldsymbol{h}(t)