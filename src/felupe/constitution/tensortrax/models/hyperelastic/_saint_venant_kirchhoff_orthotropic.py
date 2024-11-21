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
from numpy import array
from tensortrax.math import base, einsum, log
from tensortrax.math import sum as tsum
from tensortrax.math.linalg import eigh
from tensortrax.math.special import from_triu_1d

from .....math import cross


def saint_venant_kirchhoff_orthotropic(C, mu, lmbda, r1, r2, r3=None, k=2):
    r"""Strain energy function of the orthotropic hyperelastic
    `Saint-Venant Kirchhoff <https://en.wikipedia.org/wiki/Hyperelastic_material#Saint_Venant-Kirchhoff_model>`_
    material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : list of float
        List of the three second Lamé parameters :math:`\mu_1,\mu_2, \mu_3`.
    lmbda : list of float
        List of six (upper triangle) first Lamé parameters :math:`\lambda_{11},
        \lambda_{12}, \lambda_{13}, \lambda_{22}, \lambda_{23}, \lambda_{33}`.
    r1 : list of float
        First normal vector of planes of symmetry.
    r2 : list of float
        Second normal vector of planes of symmetry.
    r3 : list of float or None, optional
        Third normal vector of planes of symmetry. If None, the third normal vector
        is evaluated as :math:`r_1 \times r_2`. Default is None.
    k : float, optional
        Strain exponent (default is 2). If 2, the Green-Lagrange strain measure is used.
        For any other value, the family of Seth-Hill strains is used.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-svk-ortho`

    ..  math::
        :label: psi-svk-ortho

        \psi = \sum_{a=1}^3
            \mu_a \boldsymbol{E} : \boldsymbol{r}_a \otimes \boldsymbol{r}_a +
            \sum_{a=1}^3 \sum_{b=1}^3 \frac{\lambda_{ab}}{2}
                \left(\boldsymbol{E} : \boldsymbol{r}_a \otimes \boldsymbol{r}_a \right)
                \left(\boldsymbol{E} : \boldsymbol{r}_b \otimes \boldsymbol{r}_b \right)

    Examples
    --------
    ..  warning::
        The orthotropic Saint-Venant Kirchhoff material formulation is unstable for
        large strains.

    ..  pyvista-plot::
        :context:

        >>> import felupe.constitution.tensortrax as mat
        >>> import felupe as fem
        >>>
        >>> r = fem.math.rotation_matrix(0, axis=2)
        >>> lmbda, mu = fem.constitution.lame_converter_orthotropic(
        ...     E=[10, 10, 10],
        ...     nu=[0.3, 0.3, 0.3],
        ...     G=[1, 1, 1],
        ... )
        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.saint_venant_kirchhoff_orthotropic,
        ...     mu=mu,
        ...     lmbda=lmbda,
        ...     r1=r[:, 0],
        ...     r2=r[:, 1],
        ...     r3=r[:, 2],
        ... )
        >>> ax = umat.plot(ux=fem.math.linsteps([1, 1.1], num=10), ps=None, bx=None)

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
    eye = base.eye

    if k == 2:
        E = (C - eye(C)) / 2

    else:
        λ2, M = eigh(C)
        if k == 0:
            E = tsum(log(λ2) / 2 * M, axis=0)
        else:
            E = tsum((λ2 ** (k / 2) - 1) / k * M, axis=0)

    μ = array(mu)
    λ = from_triu_1d(array(lmbda))

    if r3 is None:
        r3 = cross(r1, r2)

    r = array([r1, r2, r3])
    Err = einsum("ai...,ij...,aj...->a...", r, E, r)

    λI1 = einsum("ab,a...,b...->...", λ / 2, Err, Err)
    μI2 = einsum("a,ai...,ij...,aj...->...", μ, r, E @ E, r)

    return μI2 + λI1
