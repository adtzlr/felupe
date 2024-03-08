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

from threading import Thread

import numpy as np

from .._cartesian import IntegralFormCartesian


class BilinearForm:
    r"""A bilinear form object with methods for integration and assembly of
    matrices.

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Parameters
    ----------
    v : BasisField
        Basis (shape) functions and gradients of a field.
    u : BasisField
        Basis (shape) functions and gradients of a field.
    dx : ndarray or None, optional (default is None)
        Array with (numerical) differential volumes.

    """

    def __init__(self, v, u, dx=None, **kwargs):
        self.v = v
        self.u = u
        self.dx = dx

        self._form = IntegralFormCartesian(
            fun=None, v=v.field, dV=self.dx, u=u.field, **kwargs
        )

    def integrate(self, weakform, kwargs={}, parallel=False, sym=False):
        r"""Return evaluated (but not assembled) integrals.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, u, **kwargs)``.
        kwargs : dict, optional
            Optional named arguments for callable weakform
        parallel : bool, optional
            Flag to activate parallel threading (default is False).
        sym : bool, optional
            Flag to active symmetric integration/assembly (default is False).

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) matrix values.
        """

        values = np.zeros(
            (
                len(self.v.basis),
                self.v.basis.shape[-4],
                len(self.u.basis),
                self.u.basis.shape[-4],
                *self.u.basis.shape[-2:],
            )
        )

        if not parallel:
            for a, vbasis in enumerate(self.v.basis):
                for i, vb in enumerate(vbasis):
                    for b, ubasis in enumerate(self.u.basis):
                        for j, ub in enumerate(ubasis):
                            if sym:
                                v = type(self.v.basis)(vb, self.v.basis.grad[a, i])
                                u = type(self.u.basis)(ub, self.u.basis.grad[b, j])
                                if len(vbasis) * a + i <= len(ubasis) * b + j:
                                    values[a, i, b, j] = values[b, j, a, i] = (
                                        weakform(v, u, **kwargs) * self.dx
                                    )

                            else:
                                v = type(self.v.basis)(vb, self.v.basis.grad[a, i])
                                u = type(self.u.basis)(ub, self.u.basis.grad[b, j])
                                values[a, i, b, j] = weakform(v, u, **kwargs) * self.dx

        else:
            idx_a, idx_i, idx_b, idx_j = np.indices(values.shape[:4])
            aibj = zip(idx_a.ravel(), idx_i.ravel(), idx_b.ravel(), idx_j.ravel())

            if sym:
                len_vbasis = values.shape[1]
                len_ubasis = values.shape[3]

                mask = len_vbasis * idx_a + idx_i <= len_ubasis * idx_b + idx_j
                idx_a, idx_i, idx_b, idx_j = (
                    idx_a[mask],
                    idx_i[mask],
                    idx_b[mask],
                    idx_j[mask],
                )

            def contribution(values, a, i, b, j, sym, kwargs):
                v = type(self.v.basis)(self.v.basis[a, i], self.v.basis.grad[a, i])
                u = type(self.u.basis)(self.u.basis[b, j], self.u.basis.grad[b, j])
                if sym:
                    values[a, i, b, j] = values[b, j, a, i] = (
                        weakform(v, u, **kwargs) * self.dx
                    )

                else:
                    values[a, i, b, j] = weakform(v, u, **kwargs) * self.dx

            threads = [
                Thread(target=contribution, args=(values, a, i, b, j, sym, kwargs))
                for a, i, b, j in aibj
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

        return values.sum(-2)
