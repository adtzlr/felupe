from . import (
    constitution,
    dof,
    element,
    math,
    mechanics,
    mesh,
    quadrature,
    region,
    solve,
    tools,
)
from .__about__ import __version__
from ._assembly import Form
from ._assembly import (
    IntegralFormMixed as IntegralForm,  # IntegralForm,; IntegralFormAxisymmetric,; LinearForm,; BilinearForm,; LinearFormMixed as LinearForm,; BilinearFormMixed as BilinearForm,; BaseForm,
)
from ._basis import BasisMixed as Basis
from ._field import (
    Field,
    FieldAxisymmetric,
    FieldContainer,
    FieldPlaneStrain,
    FieldsMixed,
)
from .constitution import (
    AreaChange,
    LinearElastic,
    LinearElasticPlaneStrain,
    LinearElasticPlaneStress,
    LinearElasticPlasticIsotropicHardening,
    LineChange,
    NeoHooke,
    OgdenRoxburgh,
    ThreeFieldVariation,
    UserMaterial,
    UserMaterialStrain,
    VolumeChange,
)
from .dof import Boundary
from .element import ArbitraryOrderLagrange as ArbitraryOrderLagrangeElement
from .element import (
    BiQuadraticQuad,
    ConstantHexahedron,
    ConstantQuad,
    Hexahedron,
    Line,
    Quad,
    QuadraticHexahedron,
    QuadraticQuad,
    QuadraticTetra,
    QuadraticTriangle,
    Tetra,
    TetraMINI,
    Triangle,
    TriangleMINI,
    TriQuadraticHexahedron,
)
from .mesh import Cube, Grid, Mesh, MeshContainer, Rectangle
from .quadrature import GaussLegendre, GaussLegendreBoundary
from .quadrature import Tetrahedron as TetrahedronQuadrature
from .quadrature import Triangle as TriangleQuadrature
from .region import (
    Region,
    RegionBiQuadraticQuad,
    RegionBoundary,
    RegionConstantHexahedron,
    RegionConstantQuad,
    RegionHexahedron,
    RegionHexahedronBoundary,
    RegionLagrange,
    RegionQuad,
    RegionQuadBoundary,
    RegionQuadraticHexahedron,
    RegionQuadraticQuad,
    RegionQuadraticTetra,
    RegionQuadraticTriangle,
    RegionTetra,
    RegionTetraMINI,
    RegionTriangle,
    RegionTriangleMINI,
    RegionTriQuadraticHexahedron,
)

try:
    from .constitution import UserMaterialHyperelastic
except:
    pass
from .mechanics import (
    CharacteristicCurve,
    Job,
    PointLoad,
    SolidBody,
    SolidBodyGravity,
    SolidBodyNearlyIncompressible,
    SolidBodyPressure,
    StateNearlyIncompressible,
    Step,
)
from .tools import newtonrhapson, project, save, topoints

try:
    from .mechanics import MultiPointConstraint, MultiPointContact
except:
    pass

__all__ = [
    "__version__",
]
