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
from . import mechanics

from .region import (
    Region,
    RegionBoundary,
    RegionQuad,
    RegionHexahedron,
    RegionQuadBoundary,
    RegionHexahedronBoundary,
    RegionConstantQuad,
    RegionConstantHexahedron,
    RegionQuadraticHexahedron,
    RegionTriQuadraticHexahedron,
    RegionTriangle,
    RegionTetra,
    RegionQuadraticTriangle,
    RegionQuadraticTetra,
    RegionTriangleMINI,
    RegionTetraMINI,
    RegionLagrange,
)
from ._assembly import (
    IntegralForm,
    IntegralFormMixed,
    IntegralFormAxisymmetric,
    LinearForm,
    BilinearForm,
    LinearFormMixed,
    BilinearFormMixed,
    BaseForm,
    Form,
)
from ._basis import (
    Basis,
    BasisMixed,
)
from ._field import (
    Field,
    FieldAxisymmetric,
    FieldMixed,
    FieldsMixed,
)
from .dof import Boundary

try:
    from .dof import (
        MultiPointConstraint,
        MultiPointContact,
    )
except:
    pass
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
    GaussLegendreBoundary,
    Triangle as TriangleQuadrature,
    Tetrahedron as TetrahedronQuadrature,
)
from .mesh import (
    Mesh,
    Rectangle,
    Cube,
)
from .constitution import (
    NeoHooke,
    LinearElastic,
    LinearElasticPlaneStress,
    LinearElasticPlaneStrain,
    ThreeFieldVariation,
    LineChange,
    AreaChange,
    VolumeChange,
    MatadiMaterial,
)
from .tools import (
    newtonrhapson,
    save,
    topoints,
    project,
)
from .mechanics import (
    SolidBody,
    SolidBodyPressure,
)

__all__ = [
    "__version__",
]
