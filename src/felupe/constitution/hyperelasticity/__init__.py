# SPDX-License-Identifier: GPL-3.0-or-later

"""
constitution.hyperelasticity
============================
This module contains manually (no automatic differentiation) defined isotropic
hyperelastic constitutive material formulations with gradients and hessians of strain
energy density functions. For more sophisticated material model formulations see
:class:`~felupe.Hyperelastic`.
"""

from ._neo_hooke_compressible import NeoHookeCompressible
from ._neo_hooke_nearly_incompressible import NeoHooke
from ._ogden_roxburgh import OgdenRoxburgh
from ._volumetric import Volumetric
from ._volumetric_strain_energy import VolumetricStrainEnergy

__all__ = [
    "NeoHooke",
    "NeoHookeCompressible",
    "OgdenRoxburgh",
    "Volumetric",
    "VolumetricStrainEnergy",
]
