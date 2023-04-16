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

import numpy as np

from ._base import Element


class Line(Element):
    def __init__(self):
        super().__init__(shape=(2, 1))
        self.points = np.array([-1, 1], dtype=float)

    def function(self, rv):
        "linear line shape functions"
        (r,) = rv
        return np.array([(1 - r), (1 + r)]) * 0.5

    def gradient(self, rv):
        "linear line gradient of shape functions"
        (r,) = rv
        return np.array([[-1], [1]]) * 0.5
