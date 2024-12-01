from . import (
    assembly,
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
    view,
)
from .__about__ import __version__
from .assembly import IntegralForm
from .assembly.expression import Form
from .constitution import (
    AreaChange,
    CompositeMaterial,
    ConstitutiveMaterial,
    Laplace,
    LinearElastic,
    LinearElasticLargeStrain,
    LinearElasticOrthotropic,
    LinearElasticPlaneStress,
    LinearElasticPlasticIsotropicHardening,
    LineChange,
    Material,
    MaterialStrain,
    NearlyIncompressible,
    NeoHooke,
    NeoHookeCompressible,
    OgdenRoxburgh,
    ThreeFieldVariation,
    ViewMaterial,
    ViewMaterialIncompressible,
    VolumeChange,
    Volumetric,
    linear_elastic,
    linear_elastic_plastic_isotropic_hardening,
)
from .dof import Boundary, BoundaryDict
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
from .field import (
    Field,
    FieldAxisymmetric,
    FieldContainer,
    FieldDual,
    FieldPlaneStrain,
    FieldsMixed,
)
from .mechanics import (
    CharacteristicCurve,
    FormItem,
    FreeVibration,
    Job,
    MultiPointConstraint,
    MultiPointContact,
    PointLoad,
    SolidBody,
    SolidBodyCauchyStress,
    SolidBodyForce,
    SolidBodyGravity,
    SolidBodyNearlyIncompressible,
    SolidBodyPressure,
    StateNearlyIncompressible,
    Step,
)
from .mesh import Circle, Cube, Grid, Mesh, MeshContainer, Point, Rectangle
from .quadrature import BazantOh, GaussLegendre, GaussLegendreBoundary
from .quadrature import Tetrahedron as TetrahedronQuadrature
from .quadrature import Triangle as TriangleQuadrature
from .region import (
    Region,
    RegionBiQuadraticQuad,
    RegionBiQuadraticQuadBoundary,
    RegionBoundary,
    RegionConstantHexahedron,
    RegionConstantQuad,
    RegionHexahedron,
    RegionHexahedronBoundary,
    RegionLagrange,
    RegionQuad,
    RegionQuadBoundary,
    RegionQuadraticHexahedron,
    RegionQuadraticHexahedronBoundary,
    RegionQuadraticQuad,
    RegionQuadraticQuadBoundary,
    RegionQuadraticTetra,
    RegionQuadraticTriangle,
    RegionTetra,
    RegionTetraMINI,
    RegionTriangle,
    RegionTriangleMINI,
    RegionTriQuadraticHexahedron,
    RegionTriQuadraticHexahedronBoundary,
)
from .tools import hello_world, newtonrhapson, project, runs_on, save, topoints
from .view import ViewField, ViewMesh
from .view import ViewSolid
from .view import ViewSolid as View
from .view import ViewXdmf

__all__ = [
    "__version__",
    "constitution",
    "dof",
    "element",
    "hello_world",
    "math",
    "mechanics",
    "mesh",
    "quadrature",
    "region",
    "solve",
    "tools",
    "Form",
    "FormItem",
    "FreeVibration",
    "IntegralForm",
    "Basis",
    "Field",
    "FieldDual",
    "FieldAxisymmetric",
    "FieldContainer",
    "FieldPlaneStrain",
    "FieldsMixed",
    "AreaChange",
    "Laplace",
    "LinearElastic",
    "LinearElasticLargeStrain",
    "LinearElasticOrthotropic",
    "LinearElasticPlaneStress",
    "LinearElasticPlasticIsotropicHardening",
    "LineChange",
    "CompositeMaterial",
    "Volumetric",
    "NeoHooke",
    "NeoHookeCompressible",
    "OgdenRoxburgh",
    "ThreeFieldVariation",
    "NearlyIncompressible",
    "Material",
    "MaterialStrain",
    "ViewMaterial",
    "ViewMaterialIncompressible",
    "ConstitutiveMaterial",
    "constitutive_material",
    "VolumeChange",
    "linear_elastic",
    "linear_elastic_plastic_isotropic_hardening",
    "Boundary",
    "BoundaryDict",
    "ArbitraryOrderLagrangeElement",
    "BiQuadraticQuad",
    "ConstantHexahedron",
    "ConstantQuad",
    "Hexahedron",
    "Line",
    "Quad",
    "QuadraticHexahedron",
    "QuadraticQuad",
    "QuadraticTetra",
    "QuadraticTriangle",
    "Tetra",
    "TetraMINI",
    "Triangle",
    "TriangleMINI",
    "TriQuadraticHexahedron",
    "Circle",
    "Cube",
    "Grid",
    "Mesh",
    "MeshContainer",
    "Point",
    "Rectangle",
    "CharacteristicCurve",
    "Job",
    "PointLoad",
    "SolidBody",
    "SolidBodyCauchyStress",
    "SolidBodyGravity",
    "SolidBodyForce",
    "SolidBodyNearlyIncompressible",
    "SolidBodyPressure",
    "StateNearlyIncompressible",
    "Step",
    "MultiPointConstraint",
    "MultiPointContact",
    "GaussLegendre",
    "GaussLegendreBoundary",
    "TetrahedronQuadrature",
    "TriangleQuadrature",
    "BazantOh",
    "Region",
    "RegionBiQuadraticQuad",
    "RegionBiQuadraticQuadBoundary",
    "RegionBoundary",
    "RegionConstantHexahedron",
    "RegionConstantQuad",
    "RegionHexahedron",
    "RegionHexahedronBoundary",
    "RegionLagrange",
    "RegionQuad",
    "RegionQuadBoundary",
    "RegionQuadraticHexahedron",
    "RegionQuadraticHexahedronBoundary",
    "RegionQuadraticQuad",
    "RegionQuadraticQuadBoundary",
    "RegionQuadraticTetra",
    "RegionQuadraticTriangle",
    "RegionTetra",
    "RegionTetraMINI",
    "RegionTriangle",
    "RegionTriangleMINI",
    "RegionTriQuadraticHexahedron",
    "RegionTriQuadraticHexahedronBoundary",
    "newtonrhapson",
    "project",
    "save",
    "topoints",
    "View",
    "ViewField",
    "ViewMesh",
    "ViewXdmf",
    "ViewSolid",
    "runs_on",
]

try:
    from .constitution import (
        Hyperelastic,
        MaterialAD,
        alexander,
        anssari_benam_bucchi,
        arruda_boyce,
        blatz_ko,
        constitutive_material,
        extended_tube,
        finite_strain_viscoelastic,
        isochoric_volumetric_split,
        lopez_pamies,
        miehe_goektepe_lulei,
        mooney_rivlin,
        morph,
        morph_representative_directions,
        neo_hooke,
        ogden,
        ogden_roxburgh,
        saint_venant_kirchhoff,
        saint_venant_kirchhoff_orthotropic,
        storakers,
        third_order_deformation,
        total_lagrange,
        updated_lagrange,
        van_der_waals,
        yeoh,
    )

    __all_tensortrax = [
        "Hyperelastic",
        "total_lagrange",
        "updated_lagrange",
        "MaterialAD",
        "alexander",
        "anssari_benam_bucchi",
        "arruda_boyce",
        "blatz_ko",
        "extended_tube",
        "finite_strain_viscoelastic",
        "isochoric_volumetric_split",
        "lopez_pamies",
        "miehe_goektepe_lulei",
        "mooney_rivlin",
        "morph",
        "morph_representative_directions",
        "neo_hooke",
        "ogden",
        "ogden_roxburgh",
        "saint_venant_kirchhoff",
        "saint_venant_kirchhoff_orthotropic",
        "storakers",
        "third_order_deformation",
        "van_der_waals",
        "yeoh",
    ]
    __all__.extend(__all_tensortrax)
except ImportError:
    pass
