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

from .._field._base import Field
from .._field._axi import FieldAxisymmetric
from ._base import IntegralForm


class IntegralFormAxisymmetric(IntegralForm):
    def __init__(self, fun, v, dV, u=None, grad_v=True, grad_u=True):

        R = v.radius
        self.dV = 2 * np.pi * R * dV

        if u is None:

            if isinstance(v, FieldAxisymmetric):

                self.mode = 1

                if grad_v:
                    fun_2d = fun[:-1, :-1]
                    fun_zz = fun[(-1,), (-1,)] / R
                else:
                    fun_2d = fun[:-1]
                    fun_zz = fun[-1].reshape(1, *fun[-1].shape) / R

                form_a = IntegralForm(fun_2d, v, self.dV, grad_v=grad_v)
                form_b = IntegralForm(fun_zz, v.scalar, self.dV)

                self.forms = [form_a, form_b]

            else:

                self.mode = 10

                form_a = IntegralForm(fun, v, self.dV, grad_v=False)
                self.forms = [
                    form_a,
                ]

        else:

            if isinstance(v, FieldAxisymmetric) and isinstance(u, FieldAxisymmetric):

                self.mode = 2

                if grad_v and grad_u:

                    form_aa = IntegralForm(
                        fun[:-1, :-1, :-1, :-1], v, self.dV, u, True, True
                    )
                    form_bb = IntegralForm(
                        fun[-1, -1, -1, -1] / R**2,
                        v.scalar,
                        self.dV,
                        u.scalar,
                        False,
                        False,
                    )
                    form_ba = IntegralForm(
                        fun[-1, -1, :-1, :-1] / R, v.scalar, self.dV, u, False, True
                    )
                    form_ab = IntegralForm(
                        fun[:-1, :-1, -1, -1] / R, v, self.dV, u.scalar, True, False
                    )

                if not grad_v and grad_u:

                    form_aa = IntegralForm(
                        fun[:-1, :-1, :-1], v, self.dV, u, False, True
                    )
                    form_bb = IntegralForm(
                        fun[-1, -1, -1] / R**2,
                        v.scalar,
                        self.dV,
                        u.scalar,
                        False,
                        False,
                    )
                    form_ba = IntegralForm(
                        fun[-1, :-1, :-1] / R, v.scalar, self.dV, u, False, True
                    )
                    form_ab = IntegralForm(
                        fun[:-1, -1, -1] / R, v, self.dV, u.scalar, False, False
                    )

                self.forms = [form_aa, form_bb, form_ba, form_ab]

            elif isinstance(v, FieldAxisymmetric) and isinstance(u, Field):

                self.mode = 30

                form_a = IntegralForm(fun[:-1, :-1], v, self.dV, u, True, False)
                form_b = IntegralForm(
                    fun[-1, -1] / R, v.scalar, self.dV, u, False, False
                )

                self.forms = [form_a, form_b]

            elif isinstance(v, Field) and isinstance(u, Field):

                self.mode = 40

                form_a = IntegralForm(fun, v, self.dV, u, False, False)

                self.forms = [
                    form_a,
                ]

    def integrate(self, parallel=False, jit=False):
        values = [form.integrate(parallel=parallel, jit=jit) for form in self.forms]

        if self.mode == 1:
            values[0] += np.pad(values[1], ((0, 0), (1, 0), (0, 0)))
            val = values[0]

        if self.mode == 30:
            if len(values[0].shape) > 4:
                values[0] = values[0][:, :, 0, 0]
            if len(values[1].shape) > 4:
                values[1] = values[1][:, :, 0, 0]
            a, b, e = values[1].shape
            values[1] = values[1].reshape(a, 1, b, e)

            values[0] += np.pad(values[1], ((0, 0), (1, 0), (0, 0), (0, 0)))
            val = values[0]

        elif self.mode == 2:
            a, b, e = values[1].shape
            values[1] = values[1].reshape(a, 1, b, 1, e)
            values[1] = np.pad(values[1], ((0, 0), (1, 0), (0, 0), (1, 0), (0, 0)))

            a, b, i, e = values[2].shape
            values[2] = values[2].reshape(a, 1, b, i, e)
            values[2] = np.pad(values[2], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))

            a, i, b, e = values[3].shape
            values[3] = values[3].reshape(a, i, b, 1, e)
            values[3] = np.pad(values[3], ((0, 0), (0, 0), (0, 0), (1, 0), (0, 0)))

            for i in range(1, len(values)):
                values[0] += values[i]

            val = values[0]

        elif self.mode == 10 or self.mode == 40:
            val = values[0]

        return val

    def assemble(self, values=None, parallel=False, jit=False):
        if values is None:
            values = self.integrate(parallel=parallel, jit=jit)
        return self.forms[0].assemble(values)
