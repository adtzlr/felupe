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

from ....math import cdya_ik, dya, identity, trace


def linear_elastic(dε, εn, σn, ζn, λ, μ, **kwargs):
    r"""3D linear-elastic material formulation to be used in
    :class:`~felupe.MaterialStrain`.

    Arguments
    ---------
    dε : ndarray
        Strain increment.
    εn : ndarray
        Old strain tensor.
    σn : ndarray
        Old stress tensor.
    ζn : list
        List of old state variables.
    λ : float
        First Lamé-constant.
    μ : float
        Second Lamé-constant (shear modulus).

    Returns
    -------
    dσdε : ndarray
        Elasticity tensor.
    σ : ndarray
        (New) stress tensor.
    ζ : list
        List of new state variables.

    Notes
    -----

    1.  Given state in point :math:`\boldsymbol{x} (\boldsymbol{\sigma}_n)` (valid).

    2.  Given strain increment :math:`\Delta\boldsymbol{\varepsilon}`, so that
        :math:`\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}_n + \Delta\boldsymbol{\varepsilon}`.

    3.  Evaluation of the stress :math:`\boldsymbol{\sigma}` and the algorithmic
        consistent tangent modulus :math:`\mathbb{C}` (=``dσdε``).

        ..  math::

            \mathbb{C} &= \lambda \ \boldsymbol{1} \otimes \boldsymbol{1} +
                2 \mu \ \boldsymbol{1} \odot \boldsymbol{1}

            \boldsymbol{\sigma} &= \boldsymbol{\sigma}_n
                + \mathbb{C} : \Delta\boldsymbol{\varepsilon}

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.MaterialStrain(material=fem.linear_elastic, λ=2.0, μ=1.0)
        >>> ax = umat.plot()

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    See Also
    --------
    MaterialStrain : A strain-based user-defined material definition with a given
        function for the stress tensor and the (fourth-order) elasticity tensor.

    """

    # change of stress due to change of strain
    eye = identity(dim=3, shape=(1, 1))
    dσ = 2 * μ * dε + λ * trace(dε) * eye

    # update stress
    σ = σn + dσ

    # evaluate elasticity tensor
    if kwargs["tangent"]:
        dσdε = 2 * μ * cdya_ik(eye, eye) + λ * dya(eye, eye)
    else:
        dσdε = None

    # update state variables (not used here)
    ζ = ζn

    return dσdε, σ, ζ
