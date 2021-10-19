from .__about__ import __version__
from . import math
from . import mesh
from . import quadrature
from . import dof
from . import element
from . import tools
from . import constitution
from . import solve
from . import region

from .region import Region
from ._assembly import (
    IntegralForm,
    IntegralFormMixed,
    IntegralFormAxisymmetric,
)
from ._field import (
    Field,
    FieldAxisymmetric,
    FieldMixed,
)
from .dof import Boundary, MultiPointConstraint, MultiPointContact
from .element import (
    Line,
    #
    Quad,
    ConstantQuad,
    #
    ArbitraryOrderLagrange as ArbitraryOrderLagrangeElement,
    #
    Hexahedron,
    ConstantHexahedron,
    QuadraticHexahedron,
    TriQuadraticHexahedron,
    #
    Triangle,
    TriangleMINI,
    QuadraticTriangle,
    #
    Tetra,
    TetraMINI,
    QuadraticTetra,
)
from .quadrature import (
    GaussLegendre,
    Triangle as TriangleQuadrature,
    Tetrahedron as TetrahedronQuadrature,
)
from .mesh import (
    Mesh,
    Rectangle,
    Cube,
)

__all__ = [
    "__version__",
]
