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
from scipy.sparse import bmat, vstack

from .._field._base import Field
from .._field._axi import FieldAxisymmetric
from .._field._planestrain import FieldPlaneStrain
from ._base import IntegralForm
from ._axi import IntegralFormAxisymmetric


class IntegralFormMixed:
    def __init__(self, fun, v, dV, u=None, grad_v=None, grad_u=None):

        self.fun = fun
        self.v = v.fields
        self.nv = len(self.v)
        self.dV = dV

        if u is not None:
            self.u = u.fields
            self.nu = len(self.u)
        else:
            self.u = None
            self.nu = None

        IntForm = {
            Field: IntegralForm,
            FieldPlaneStrain: IntegralForm,
            FieldAxisymmetric: IntegralFormAxisymmetric,
        }[type(self.v[0])]

        if isinstance(self.v[0], FieldAxisymmetric):
            for i in range(1, len(self.v)):
                self.v[i].radius = self.v[0].radius

        if grad_v is None:
            self.grad_v = np.zeros_like(self.v, dtype=bool)
            self.grad_v[0] = True
        else:
            self.grad_v = grad_v

        if grad_u is None and u is not None:
            self.grad_u = np.zeros_like(self.u, dtype=bool)
            self.grad_u[0] = True
        else:
            self.grad_u = grad_u

        self.forms = []

        if len(fun) == self.nv and u is None:
            # LinearForm
            self.mode = 1
            self.i = np.arange(self.nv)
            self.j = np.zeros_like(self.i)

            for fun, v, grad_v in zip(self.fun, self.v, self.grad_v):
                f = IntForm(fun=fun, v=v, dV=self.dV, grad_v=grad_v)
                self.forms.append(f)

        elif len(fun) == np.sum(1 + np.arange(self.nv)) and u is not None:
            # BilinearForm
            self.mode = 2
            self.i, self.j = np.triu_indices(self.nv)

            for a, (i, j) in enumerate(zip(self.i, self.j)):
                f = IntForm(
                    self.fun[a],
                    v=self.v[i],
                    dV=self.dV,
                    u=self.u[j],
                    grad_v=self.grad_v[i],
                    grad_u=self.grad_u[j],
                )
                self.forms.append(f)
        else:
            raise ValueError("Unknown input format.")

    def assemble(self, values=None, parallel=False, jit=False, block=True):

        out = []

        if values is None:
            values = [None] * len(self.forms)

        for val, form in zip(values, self.forms):
            out.append(form.assemble(val, parallel=parallel, jit=jit))

        if block and self.mode == 2:
            K = np.zeros((self.nv, self.nv), dtype=object)
            for a, (i, j) in enumerate(zip(self.i, self.j)):
                K[i, j] = out[a]
                if i != j:
                    K[j, i] = out[a].T

            return bmat(K).tocsr()

        if block and self.mode == 1:
            return vstack(out).tocsr()

        else:
            return out

    def integrate(self, parallel=False, jit=False):

        out = []
        for form in self.forms:
            out.append(form.integrate(parallel=parallel, jit=jit))

        return out
