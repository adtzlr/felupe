# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

from ._neo_hooke_nearly_incompressible import NeoHooke


class Volumetric(NeoHooke):
    "Neo-Hookean material formulation with deactivated shear modulus."

    def __init__(self, bulk, parallel=False):
        super().__init__(mu=None, bulk=bulk, parallel=parallel)
