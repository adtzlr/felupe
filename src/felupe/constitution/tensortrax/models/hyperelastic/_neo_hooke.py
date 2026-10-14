# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import trace
from tensortrax.math.linalg import det


def neo_hooke(C, mu):
    r"""Strain energy function of the isotropic hyperelastic
    `Neo-Hookean <https://en.wikipedia.org/wiki/Neo-Hookean_solid>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : float
        Shear modulus.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-nh`.

    ..  math::
        :label: psi-nh

        \psi = \frac{\mu}{2} \left(\text{tr}\left(\hat{\boldsymbol{C}}\right) - 3\right)

    Examples
    --------
    First, choose the desired automatic differentiation backend

    ..  plot::
        :context: close-figs

        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    and create the hyperelastic material.

    ..  plot::
        :context: close-figs

        >>> umat = mat.Hyperelastic(mat.models.hyperelastic.neo_hooke, mu=1.0)
        >>> ax = umat.plot(incompressible=True)

    """
    return mu / 2 * (det(C) ** (-1 / 3) * trace(C) - 3)
