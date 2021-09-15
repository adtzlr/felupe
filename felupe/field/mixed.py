# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:29:15 2021

@author: adutz
"""
# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np


class FieldMixed:
    def __init__(self, fields):
        self.fields = fields
        self.values = tuple(f.values for f in self.fields)

    def extract(self, grad=True, sym=False, add_identity=True):
        "extract gradient or interpolated field values at quadrature points."

        if isinstance(grad, bool):
            grad = (grad,)

        grads = np.pad(grad, (0, len(self.fields) - 1))
        return tuple(
            f.extract(g, sym, add_identity) for g, f in zip(grads, self.fields)
        )
