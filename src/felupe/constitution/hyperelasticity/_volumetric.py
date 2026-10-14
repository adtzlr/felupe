# SPDX-License-Identifier: GPL-3.0-or-later

from ._neo_hooke_nearly_incompressible import NeoHooke


class Volumetric(NeoHooke):
    "Neo-Hookean material formulation with deactivated shear modulus."

    def __init__(self, bulk, parallel=False):
        super().__init__(mu=None, bulk=bulk, parallel=parallel)
