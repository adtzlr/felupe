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

from ..math import cdya_ik, ddot, dya, identity, sqrt, trace


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


def linear_elastic_plastic_isotropic_hardening(dε, εn, σn, ζn, λ, μ, σy, K, **kwargs):
    r"""Linear-elastic-plastic material formulation with linear isotropic
    hardening (return mapping algorithm) to be used in :class:`~felupe.MaterialStrain`.

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
    σy : float
        Initial yield stress.
    K : float
        Isotropic hardening modulus.

    Returns
    -------
    dσdε : ndarray
        Algorithmic consistent elasticity tensor.
    σ : ndarray
        (New) stress tensor.
    ζ : list
        List of new state variables.

    Notes
    -----

    1.  Given state in point :math:`x (\sigma_n, \zeta_n=[\varepsilon^p_n, \alpha_n])`
        (valid).

    2.  Given strain increment :math:`\Delta\varepsilon`, so that
        :math:`\varepsilon = \varepsilon_n + \Delta\varepsilon`.

    3.  Evaluation of the hypothetic trial state:

        ..  math::

            \mathbb{C} &= \lambda\ \boldsymbol{1} \otimes \boldsymbol{1}
                + 2 \mu\ \boldsymbol{1} \odot \boldsymbol{1}

            \sigma &= \sigma_n + \mathbb{C} : \Delta\varepsilon

            s &= \text{dev}(\sigma)

            \varepsilon^p &= \varepsilon^p_n

            \alpha &= \alpha_n

            f &= ||s|| - \sqrt{\frac{2}{3}}\ (\sigma_y + K \alpha)

    4.  If :math:`f \le 0`, then elastic step:

        Set :math:`y = y_n + \Delta y, y=(\sigma, \zeta=[\varepsilon^p, \alpha])`,

        algorithmic consistent tangent modulus :math:`d\sigma d\varepsilon`.

        ..  math::

            d\sigma d\varepsilon = \mathbb{C}

        Else:

        ..  math::

            d\gamma &= \frac{f}{2\mu + \frac{2}{3} K}

            n &= \frac{s}{||s||}

            \sigma &= \sigma - 2\mu \Delta\gamma n

            \varepsilon^p &= \varepsilon^p_n + \Delta\gamma n

            \alpha &= \alpha_n + \sqrt{\frac{2}{3}}\ \Delta\gamma

        Algorithmic consistent tangent modulus:

        ..  math::

            d\sigma d\varepsilon = \mathbb{C}
                - \frac{2 \mu}{1 + \frac{K}{3 \mu}} n \otimes n
                - \frac{2 \mu \Delta\gamma}{||s||} \left[
                    2 \mu \left( \boldsymbol{1} \odot \boldsymbol{1}
                     - \frac{1}{3} \boldsymbol{1} \otimes \boldsymbol{1}
                    - n \otimes n \right)
                \right]

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.MaterialStrain(
        ...     material=fem.linear_elastic_plastic_isotropic_hardening,
        ...     λ=2.0,
        ...     μ=1.0,
        ...     σy=1.0,
        ...     K=0.1,
        ...     dim=3,
        ...     statevars=(1, (3, 3)),
        ... )

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

    eye = identity(dε)

    # elasticity tensor
    if kwargs["tangent"]:
        dσdε = np.zeros((3, 3, 3, 3, *dε.shape[2:]))
        dσdε[:] = λ * dya(eye, eye) + 2 * μ * cdya_ik(eye, eye)
    else:
        dσdε = None

    # elastic hypothetic (trial) stress and deviatoric stress
    dσ = 2 * μ * dε + λ * trace(dε) * eye
    σ = σn + dσ
    s = σ - 1 / 3 * trace(σ) * eye

    # unpack old state variables
    α, εp = ζn

    # hypothetic (trial) yield function
    norm_s = sqrt(ddot(s, s))
    f = norm_s - sqrt(2 / 3) * (σy + K * α)

    ζ = ζn

    # check yield function and create a mask where plasticity occurs
    mask = (f > 0)[0]

    # update stress, tangent and state due to plasticity
    if np.any(mask):
        dγ = f / (2 * μ + 2 / 3 * K)
        n = s / norm_s
        εp = εp + dγ * n
        α = α + sqrt(2 / 3) * dγ

        # stress
        σ[..., mask] = (σ - 2 * μ * dγ * n)[..., mask]

        # algorithmic consistent tangent modulus
        if kwargs["tangent"]:
            dσdε[..., mask] = (
                dσdε
                - 2 * μ / (1 + K / (3 * μ)) * dya(n, n)
                - 2
                * μ
                * dγ
                / norm_s
                * (
                    2 * μ * (cdya_ik(eye, eye) - 1 / 3 * dya(eye, eye))
                    - 2 * μ * dya(n, n)
                )
            )[..., mask]

        # update list of state variables
        ζ[0][..., mask] = α[..., mask]
        ζ[1][..., mask] = εp[..., mask]

    return dσdε, σ, ζ
