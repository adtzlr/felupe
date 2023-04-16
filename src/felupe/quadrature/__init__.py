from ._base import Scheme
from ._gausslegendre import GaussLegendre, GaussLegendreBoundary
from ._tetra import Tetrahedron
from ._triangle import Triangle

__all__ = [
    "Scheme",
    "GaussLegendre",
    "GaussLegendreBoundary",
    "Tetrahedron",
    "Triangle",
]
