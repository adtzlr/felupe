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
    # IntegralForm,
    IntegralFormMixed as IntegralForm,
    # IntegralFormAxisymmetric,
    # LinearForm,
    # BilinearForm,
    # LinearFormMixed as LinearForm,
    # BilinearFormMixed as BilinearForm,
    # BaseForm,
    Form,
)
from ._basis import (
    BasisMixed as Basis,
)
from ._field import (
    Field,
    FieldAxisymmetric,
    FieldPlaneStrain,
    FieldContainer,
    FieldsMixed,
)
from .dof import Boundary
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
    MeshContainer,
    Rectangle,
    Cube,
    Grid,
)
from .constitution import (
    UserMaterial,
    UserMaterialStrain,
    NeoHooke,
    OgdenRoxburgh,
    LinearElastic,
    LinearElasticPlaneStress,
    LinearElasticPlaneStrain,
    LinearElasticPlasticIsotropicHardening,
    ThreeFieldVariation,
    LineChange,
    AreaChange,
    VolumeChange,
)
from .tools import (
    newtonrhapson,
    save,
    topoints,
    project,
)
from .mechanics import (
    SolidBody,
    SolidBodyNearlyIncompressible,
    StateNearlyIncompressible,
    SolidBodyPressure,
    SolidBodyGravity,
    PointLoad,
    Step,
    Job,
    CharacteristicCurve,
)

try:
    from .mechanics import (
        MultiPointConstraint,
        MultiPointContact,
    )
except:
    pass

__all__ = [
    "__version__",
]
