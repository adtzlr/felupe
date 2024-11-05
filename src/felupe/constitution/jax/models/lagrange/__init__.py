"""
Strain-energy density functions for strain energy-gradient (stress) model formulations.

This module contains material model formulations to be used as the ``fun``-argument in
:func:`~felupe.constitution.jax.Material`. The gradient as well as the hessian of
the strain energy density function is carried out by automatic differentiation using
:mod:`jax`. Hence, all math-functions must be taken from :mod:`jax.numpy`.
"""

from ._morph import morph
from ._morph_representative_directions import morph_representative_directions
from ._morph_uniaxial import morph_uniaxial

__all__ = [
    "morph",
    "morph_representative_directions",
    "morph_uniaxial",
]

# default (stable) material parameters
morph.kwargs = dict(p=[0, 0, 0, 0, 0, 1, 0, 0])
morph_representative_directions.kwargs = dict(p=[0, 0, 0, 0, 0, 1, 0, 0])
morph_uniaxial.kwargs = dict(p=[0, 0, 0, 0, 0, 1, 0, 0])
