from ._gausslegendre import GaussLegendre, GaussLegendreBoundary
from ._scheme import Scheme
from ._sphere import BazantOh
from ._tetra import Tetrahedron
from ._triangle import Triangle

__all__ = [
    "Scheme",
    "GaussLegendre",
    "GaussLegendreBoundary",
    "Tetrahedron",
    "Triangle",
    "BazantOh",
]
