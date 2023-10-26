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
        An object with basis functions (gradients) of a field.
    grad_v : bool, optional (default is False)
        Flag to use the gradient of ``v``.
    u : BasisField
        An object with basis function (gradients) of a field.
    grad_u : bool, optional (default is False)
        Flag to use the gradient of ``u``.
    dx : ndarray or None, optional (default is None)
        Array with (numerical) differential volumes.

    """

    def __init__(self, v, u, grad_v=False, grad_u=False, dx=None):
        self.v = v
        self.grad_v = grad_v
        self.u = u
        self.grad_u = grad_u
        self.dx = dx

        self._form = IntegralFormCartesian(
            None, v.field, self.dx, u.field, grad_v, grad_u
        )

    def integrate(self, weakform, args=(), kwargs={}, parallel=False, sym=False):
        r"""Return evaluated (but not assembled) integrals.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, *args, **kwargs)``.
        args : tuple, optional
            Optional arguments for callable weakform
        kwargs : dict, optional
            Optional named arguments for callable weakform
        parallel : bool, optional (default is False)
            Flag to activate parallel threading.
        sym : bool, optional (default is False)
            Flag to active symmetric integration/assembly.

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) matrix values.
        """

        if self.grad_v:
            v = self.v.grad
        else:
            v = self.v.basis

        if self.grad_u:
            u = self.u.grad
        else:
            u = self.u.basis

        values = np.zeros((len(v), v.shape[-4], len(u), u.shape[-4], *u.shape[-2:]))

        if not parallel:
            for a, vbasis in enumerate(v):
                for i, vb in enumerate(vbasis):
                    for b, ubasis in enumerate(u):
                        for j, ub in enumerate(ubasis):
                            if sym:
                                if len(vbasis) * a + i <= len(ubasis) * b + j:
                                    values[a, i, b, j] = values[b, j, a, i] = (
                                        weakform(vb, ub, *args, **kwargs) * self.dx
                                    )

                            else:
                                values[a, i, b, j] = (
                                    weakform(vb, ub, *args, **kwargs) * self.dx
                                )

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

            def contribution(values, a, i, b, j, sym, args, kwargs):
                if sym:
                    values[a, i, b, j] = values[b, j, a, i] = (
                        weakform(v[a, i], u[b, j], *args, **kwargs) * self.dx
                    )

                else:
                    values[a, i, b, j] = (
                        weakform(v[a, i], u[b, j], *args, **kwargs) * self.dx
                    )

            threads = [
                Thread(
                    target=contribution, args=(values, a, i, b, j, sym, args, kwargs)
                )
                for a, i, b, j in aibj
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

        return values.sum(-2)
