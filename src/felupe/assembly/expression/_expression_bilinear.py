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

from .._integral import IntegralForm
from ._bilinear import BilinearForm


class BilinearFormExpression:
    r"""A bilinear form object with methods for integration and assembly of matrices.

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Parameters
    ----------
    v : FieldContainer
        A field container.
    grad_v : tuple of bool, optional (default is None)
        Tuple of flags to use the gradients of ``v``.
    u : FieldContainer
        A field container.
    grad_u : tuple of bool, optional (default is None)
        Flag to use the gradients of ``u``.

    """

    def __init__(self, v, u, grad_v=None, grad_u=None):
        self.v = v
        self.u = u
        self.dx = self.v[0].region.dV

        self.nv = len(v.fields)
        self.i, self.j = np.triu_indices(self.nv)

        def _set_first_grad_true(grad, fields):
            if grad is None:
                grad = np.zeros_like(fields, dtype=bool)
                grad[0] = True
            return grad

        self.grad_v = _set_first_grad_true(grad_v, self.v.fields)
        self.grad_u = _set_first_grad_true(grad_u, self.u.fields)

        self._form = IntegralForm(
            fun=np.zeros(len(self.i)),
            v=self.v,
            dV=self.dx,
            u=self.u,
            grad_v=self.grad_v,
            grad_u=self.grad_u,
        )

        self._bilinearform = []

        for a, (i, j) in enumerate(zip(self.i, self.j)):
            self._bilinearform.append(
                BilinearForm(
                    v=self.v[i],
                    u=self.u[j],
                    grad_v=self.grad_v[i],
                    grad_u=self.grad_u[j],
                    dx=self.dx,
                )
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

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) matrix values.
        """

        return [
            form.integrate(fun, args, kwargs, parallel=parallel, sym=sym)
            for form, fun in zip(self._bilinearform, weakform)
        ]

    def assemble(self, weakform, args=(), kwargs={}, parallel=False, sym=False):
        r"""Return the assembled integral as matrix.

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

        Returns
        -------
        values : csr_matrix
            The assembled matrix.
        """

        values = self.integrate(weakform, args, kwargs, parallel=parallel, sym=sym)

        return self._form.assemble(values)
