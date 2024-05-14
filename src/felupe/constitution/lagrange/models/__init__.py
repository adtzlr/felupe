"""
Strain-energy density functions for hyperlastic model formulations.

This module contains material model formulations to be used as the ``fun``-argument in :func:`~felupe.MaterialAD`. The gradient as well as the hessian of the strain energy
density function is carried out by automatic differentiation using :mod:`tensortrax`.
Hence, all math-functions must be taken from :mod:`tensortrax.math`.
"""

from ._morph import morph

__all__ = [
    "morph",
]

# default (stable) material parameters
morph.kwargs = dict(p=[0, 0, 0, 0, 0, 1, 0, 0])
