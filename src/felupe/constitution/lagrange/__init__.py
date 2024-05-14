"""
Total Lagrange and Updated Lagrange material model formulations.

This module contains automatic-differentiation (ad) material model formulations which
have to be defined by the first Piola-Kirchhoff stress tensor in terms of the
deformation gradient. Total Lagrange and Updated Lagrange formulations are available
via function decorators.
"""
from ._material_ad import MaterialAD
from ._total_lagrange import total_lagrange
from ._updated_lagrange import updated_lagrange

__all__ = [
    "total_lagrange",
    "updated_lagrange",
]
