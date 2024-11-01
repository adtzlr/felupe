# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from tensortrax import Tensor, Δ, Δδ, f, δ
from tensortrax.math import exp, log, trace
from tensortrax.math.linalg import det


def alexander(C, C1, C2, C3, gamma, k):
    r"""Strain energy function of the isotropic hyperelastic
    `Alexander <https://doi.org/10.1016/0020-7225(68)90006-2>`_ material
    formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    C1 : float
        Scale factor for the first invariant term.
    C2 : float
        Scale factor for the second invariant term.
    C3 : float
        Scale factor for the logarithmic second invariant term.
    gamma : float
        Offset-normalization parameter for the logarithmic second invariant term.
    k : float
        Scale factor for the exponential first invariant term.

    Notes
    -----
    ..  warning::
        The strain energy function of the Alexander model formulation is not directly
        implemented. Only its gradient and hessian w.r.t. the right Cauchy-Green
        deformation tensor are defined. This is because the imaginary error-function
        :math:`\text{erfi}(x)` is not included in NumPy - this would require SciPy as a
        dependency.

    The strain energy function is given in Eq. :eq:`psi-alexander`

    ..  math::
        :label: psi-alexander

        \psi = C_1 \int_{\hat{I}_1} \exp \left(
                   k \left(\hat{I}_1 - 3 \right)^2
               \right) \ d\hat{I}_1
             + C_2 \ln \left(\frac{\hat{I}_2 - 3 + \gamma}{\gamma} \right)
             + C_3 \left(\hat{I}_2 - 3 \right)

    with the first and second main invariant of the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`invariants-alexander`.

    ..  math::
        :label: invariants-alexander

        \hat{I}_1 &= J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

        \hat{I}_2 &= J^{-4/3} \frac{1}{2} \left(
            \text{tr}\left(\boldsymbol{C}\right)^2 -
            \text{tr}\left(\boldsymbol{C}^2\right)
        \right)

    The initial shear modulus :math:`\mu` is given in Eq.
    :eq:`shear-modulus-alexander`.

    ..  math::
        :label: shear-modulus-alexander

        \mu = 2 \left( C_1 + \frac{C_2}{\gamma} + C_3 \right)

    Examples
    --------

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...    fem.alexander, C1=17, C2=19.85, C3=1, gamma=0.735, k=0.00015
        ... )
        >>> ux = fem.math.linsteps([0.6, 5], num=50)
        >>> ps = fem.math.linsteps([1, 5], num=50)
        >>> bx = fem.math.linsteps([1, 3], num=50)
        >>>
        >>> ax = umat.plot(ux=ux, ps=ps, bx=bx, incompressible=True)

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    References
    ----------
    .. [1] H. Alexander, "A constitutive relation for rubber-like materials",
       International Journal of Engineering Science, vol. 6, no. 9. Elsevier BV, pp.
       549–563, Sep. 1968. doi:
       `10.1016/0020-7225(68)90006-2 <https://doi.org/10.1016/0020-7225(68)90006-2>`_.
    """
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (I1**2 - J3**2 * trace(C @ C)) / 2

    def W(I1):
        "A non-evaluated function with defined first- and second derivatives."

        dWdI1 = exp(k * (I1 - 3) ** 2)
        return Tensor(
            x=np.full_like(f(I1), fill_value=np.nan),
            δx=f(dWdI1) * δ(I1),
            Δx=f(dWdI1) * Δ(I1),
            Δδx=δ(dWdI1) * Δ(I1) + f(dWdI1) * Δδ(I1),
            ntrax=I1.ntrax,
        )

    return C1 * W(I1) + C2 * log(((I2 - 3) + gamma) / gamma) + C3 * (I2 - 3)
