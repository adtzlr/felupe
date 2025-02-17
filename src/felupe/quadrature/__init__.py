from ._gauss_legendre import GaussLegendre, GaussLegendreBoundary
from ._gauss_lobatto import GaussLobatto, GaussLobattoBoundary
from ._scheme import Scheme
from ._sphere import BazantOh
from ._tetra import Tetrahedron
from ._triangle import Triangle

__all__ = [
    "Scheme",
    "GaussLegendre",
    "GaussLegendreBoundary",
    "GaussLobatto",
    "GaussLobattoBoundary",
    "Tetrahedron",
    "Triangle",
    "BazantOh",
]
