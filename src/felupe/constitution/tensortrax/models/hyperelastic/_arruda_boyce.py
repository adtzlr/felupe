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

from tensortrax.math import sum as tsum
from tensortrax.math import trace
from tensortrax.math.linalg import det


def arruda_boyce(C, C1, limit):
    r"""Strain energy function of the isotropic hyperelastic
    `Arruda-Boyce <https://en.wikipedia.org/wiki/Arruda-Boyce_model>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    C1 : float
        Initial shear modulus.
    limit : float
        Limiting stretch :math:`\lambda_m` at which the polymer chain network becomes
        locked.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-ab`
    
    ..  math::
        :label: psi-ab

        \psi = C_1 \sum_{i=1}^5 \alpha_i \beta^{i-1} \left( \hat{I}_1^i - 3^i \right)

    with the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor as given in Eq. :eq:`invariants-ab`
    
    ..  math::
        :label: invariants-ab
    
        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)
    
    and :math:`\alpha_i` and :math:`\beta` as denoted in Eq. :eq:`ab-param`.
    
    ..  math::
        :label: ab-param
        
        \boldsymbol{\alpha} &= \begin{bmatrix} 
            \frac{1}{2} \\ 
            \frac{1}{20} \\
            \frac{11}{1050} \\
            \frac{19}{7000} \\
            \frac{519}{673750}
        \end{bmatrix}
        
        \beta &= \frac{1}{\lambda_m^2}
    
    The initial shear modulus is a function of both material parameters, see Eq.
    :eq:`shear-modulus-ab`.
    
    ..  math::
        :label: shear-modulus-ab
        
        \mu = C_1 \left( 
            1 + \frac{3}{5 \lambda_m^2} + \frac{99}{175 \lambda_m^4} 
              + \frac{513}{875 \lambda_m^6} + \frac{42039}{67375 \lambda_m^8} 
        \right)

    Examples
    --------
    
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.arruda_boyce, C1=1.0, limit=3.2)
        >>> ax = umat.plot(incompressible=True)

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    """
    I1 = det(C) ** (-1 / 3) * trace(C)

    alphas = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    beta = 1 / limit**2

    out = []
    for j, alpha in enumerate(alphas):
        i = j + 1
        out.append(alpha * beta ** (i - 1) * (I1**i - 3**i))

    return C1 * tsum(out)
