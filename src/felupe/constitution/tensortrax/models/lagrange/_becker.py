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

from tensortrax.math import base, einsum, log, sqrt, trace
from tensortrax.math.linalg import eigh, inv
from tensortrax.math.special import dev

from ..._total_lagrange import total_lagrange


@total_lagrange
def becker(F, mu, lmbda):
    r"""Second Piola-Kirchhoff stress tensor of
    `Becker <https://en.wikipedia.org/wiki/George_Ferdinand_Becker>`_'s
    logarithmic material model formulation [1]_ [2]_.

    Parameters
    ----------
    F : tensortrax.Tensor or jax.Array
        Deformation gradient tensor.
    mu : float
        Shear modulus (second Lamé parameter).
    lmbda : float
        First Lamé parameter.

    Returns
    -------
    S : tensortrax.Tensor or jax.Array
        Second Piola-Kirchhoff stress tensor.

    Notes
    -----
    This logarithmic material model formulation utilizes a linear-elastic stress-strain
    formulation for the Biot stress tensor, based on the Lagrangian logarithmic strain
    tensor, see Eq. :eq:`becker-biot-stress`.

    ..  math::
        :label: becker-biot-stress

        \boldsymbol{T} = 2 \mu \ \ln \boldsymbol{U}
            + \lambda \ \operatorname{tr}(\ln \boldsymbol{U}) \boldsymbol{1}

    The second Piola-Kirchhoff stress tensor is then obtained from the Biot stress
    tensor, see Eq. :eq:`becker-pk2-stress`.

    ..  math::
        :label: becker-pk2-stress

        \boldsymbol{S} = \boldsymbol{U}^{-1} \boldsymbol{T}

    Examples
    --------
    First, choose the desired automatic differentiation backend

    ..  plot::
        :context: close-figs

        >>> import felupe as fem
        >>>
        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    and create the material.

    ..  plot::
        :context: close-figs

        >>> umat = mat.Material(mat.models.lagrange.becker, mu=1.0, bulk=2.0)
        >>> ax = umat.plot()

    References
    ----------
    ..  [1] P. Neff, I. Münch, and R. Martin, "Rediscovering GF Becker's early axiomatic
        deduction of a multiaxial nonlinear stress–strain relation based on logarithmic
        strain", Mathematics and Mechanics of Solids, vol. 21, no. 7, pp. 856–911, Aug.
        2014, doi:
        `10.1177/1081286514542296 <https://doi.org/10.1177/1081286514542296>`_.
    ..  [2] G.F. Becker, "The Finite Elastic Stress-Strain Function", The American
        Journal of Science, pp.337–356, 1893.

    """

    # right Cauchy-Green deformation tensor C
    C = F.T @ F

    # eigenvalues λC, principal stretches λ and eigenbases M
    λC, M = eigh(C)
    λ = sqrt(λC)

    # right stretch tensor U and Lagrangian logarithmic strain tensor E
    U = einsum("a...,aij...->ij...", λ, M)
    E = einsum("a...,aij...->ij...", log(λ), M)

    # Biot stress tensor T and second Piola-Kirchhoff stress tensor S
    T = 2 * mu * E + lmbda * trace(E) * base.eye(U)
    S = inv(U) @ T

    return S
