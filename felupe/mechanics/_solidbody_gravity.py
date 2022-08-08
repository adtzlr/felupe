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

from .._assembly import IntegralFormMixed
from ..constitution import AreaChange
from ._helpers import Assemble, Results


class SolidBodyGravity:
    "A SolidBody with methods for the assembly of sparse vectors/matrices."

    def __init__(self, field, gravity, density):

        self.field = field
        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self._form = IntegralFormMixed
        
        self.results.gravity = np.array(gravity)
        self.results.density = density

    def _vector(
        self, field=None, parallel=False, jit=False, resize=None
    ):

        if field is not None:
            self.field = field
        
        self.results.force = self._form(
            fun=[self.results.density * self.results.gravity.reshape(-1, 1, 1)],
            v=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
        ).assemble(parallel=parallel, jit=jit)
        
        if resize is not None:
            self.results.force.resize(*resize.shape)

        return self.results.force

    def _matrix(
        self, field=None, parallel=False, jit=False, resize=None
    ):
        
        from scipy.sparse import csr_matrix

        if field is not None:
            self.field = field

        self.results.stiffness = csr_matrix(([0], ([0], [0])))
        
        if resize is not None:
            self.results.stiffness.resize(*resize.shape)

        return self.results.stiffness