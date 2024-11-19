from ._base import CompositeMaterial, ConstitutiveMaterial, constitutive_material
from ._kinematics import AreaChange, LineChange, VolumeChange
from ._material import Material
from ._mixed import NearlyIncompressible, ThreeFieldVariation
from ._view import ViewMaterial, ViewMaterialIncompressible
from .hyperelasticity import NeoHooke, NeoHookeCompressible, OgdenRoxburgh, Volumetric
from .linear_elasticity import (
    LinearElastic,
    LinearElasticLargeStrain,
    LinearElasticOrthotropic,
    LinearElasticPlaneStrain,
    LinearElasticPlaneStress,
    LinearElasticTensorNotation,
    lame_converter,
    lame_converter_orthotropic,
)
from .poisson import Laplace
from .small_strain import MaterialStrain
from .small_strain.models import (
    LinearElasticPlasticIsotropicHardening,
    linear_elastic,
    linear_elastic_plastic_isotropic_hardening,
)

__all__ = [
    "NeoHooke",
    "NeoHookeCompressible",
    "Laplace",
    "LinearElastic",
    "LinearElasticLargeStrain",
    "LinearElasticOrthotropic",
    "LinearElasticPlaneStrain",
    "LinearElasticPlaneStress",
    "LinearElasticTensorNotation",
    "lame_converter",
    "lame_converter_orthotropic",
    "OgdenRoxburgh",
    "LinearElasticPlasticIsotropicHardening",
    "Material",
    "MaterialStrain",
    "AreaChange",
    "LineChange",
    "VolumeChange",
    "ThreeFieldVariation",
    "NearlyIncompressible",
    "linear_elastic",
    "linear_elastic_plastic_isotropic_hardening",
    "ViewMaterial",
    "ViewMaterialIncompressible",
    "ConstitutiveMaterial",
    "constitutive_material",
    "CompositeMaterial",
    "Volumetric",
]
try:
    from .tensortrax import Hyperelastic
    from .tensortrax import Material as MaterialAD
    from .tensortrax import isochoric_volumetric_split, total_lagrange, updated_lagrange
    from .tensortrax.models.hyperelastic import (
        alexander,
        anssari_benam_bucchi,
        arruda_boyce,
        blatz_ko,
        extended_tube,
        finite_strain_viscoelastic,
        lopez_pamies,
        miehe_goektepe_lulei,
        mooney_rivlin,
        neo_hooke,
        ogden,
        ogden_roxburgh,
        saint_venant_kirchhoff,
        saint_venant_kirchhoff_orthotropic,
        storakers,
        third_order_deformation,
        van_der_waals,
        yeoh,
    )
    from .tensortrax.models.lagrange import morph, morph_representative_directions

    __all_tensortrax = [
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
        "MaterialAD",
        "Hyperelastic",
        "total_lagrange",
        "updated_lagrange",
    ]
    __all__.extend(__all_tensortrax)

except ImportError:
    pass
