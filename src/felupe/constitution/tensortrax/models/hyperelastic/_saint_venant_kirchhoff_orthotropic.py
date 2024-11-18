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
from tensortrax.math import base, einsum
from tensortrax.math.special import from_triu_1d


def saint_venant_kirchhoff_orthotropic(C, mu, lmbda, r1, r2, r3):
    r"""Strain energy function of the orthotropic hyperelastic
    `Saint-Venant Kirchhoff <https://en.wikipedia.org/wiki/Hyperelastic_material#Saint_Venant-Kirchhoff_model>`_
    material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : list of float
        List of the three shear moduli.
    lmbda : list of float
        The six (upper triangle) orthotropic moduli.
    r1 : list of float
        First normal vector of planes of symmetry.
    r2 : list of float
        Second normal vector of planes of symmetry.
    r3 : list of float
        Third normal vector of planes of symmetry.

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
        >>> r = fem.math.rotation_matrix(30, axis=2)
        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.saint_venant_kirchhoff_orthotropic,
        ...     mu=[1, 1, 1],
        ...     lmbda=[20, 20, 20, 20, 20, 20],
        ...     r1=r[:, 0],
        ...     r2=r[:, 1],
        ...     r3=r[:, 2],
        ... )
        >>> ax = umat.plot(incompressible=False)

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
    r = array([r1, r2, r3])

    E = (C - base.eye(C)) / 2
    Err = einsum("ai...,ij...,aj...->a...", r, E, r)

    I1 = einsum("a...,b...->ab...", Err, Err)
    I2 = einsum("ai...,ij...,aj...->a...", r, E @ E, r)

    μ = array(mu)
    λ = from_triu_1d(array(lmbda))

    return einsum("a,a...->...", μ, I2) + einsum("ab,ab...->...", λ, I1**2 / 2)
