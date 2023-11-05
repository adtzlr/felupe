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
from ._linear import LinearForm


class LinearFormExpression:
    r"""A linear form object with methods for integration and assembly of vectors.

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Parameters
    ----------
    v : FieldContainer
        A field container.
    grad_v : tuple of bool, optional (default is None)
        Flag to use the gradients of ``v``.


    """

    def __init__(self, v, grad_v=None):
        self.v = v
        self.dx = self.v[0].region.dV
        self._form = IntegralForm(
            np.zeros(len(v.fields)), self.v, self.dx, grad_v=grad_v
        )

        if grad_v is None:
            self.grad_v = np.zeros_like(self.v.fields, dtype=bool)
            self.grad_v[0] = True
        else:
            self.grad_v = grad_v

        self._linearform = [
            LinearForm(v=vi, grad_v=gvi, dx=self.dx)
            for vi, gvi in zip(self.v, self.grad_v)
        ]

    def integrate(self, weakform, args=(), kwargs={}, parallel=False):
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
            Integrated (but not assembled) vector values.
        """

        return [
            form.integrate(fun, args, kwargs, parallel=parallel)
            for form, fun in zip(self._linearform, weakform)
        ]

    def assemble(self, weakform, args=(), kwargs={}, parallel=False):
        r"""Return the assembled integral as vector.

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
            The assembled vector.
        """

        values = self.integrate(weakform, args, kwargs, parallel=parallel)

        return self._form.assemble(values)
