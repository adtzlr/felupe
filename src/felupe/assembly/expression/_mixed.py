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
from ._linear import LinearForm


class LinearFormExpression:
    r"""A linear form object with methods for integration and assembly of
    vectors where ``v`` is a basis object on a field container.

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Parameters
    ----------
    v : Basis
        Basis (shape) functions and gradients of a field.


    """

    def __init__(self, v):
        self.v = v
        self.dx = self.v.field[0].region.dV
        self._form = IntegralForm(np.zeros(len(v.field.fields)), self.v.field, self.dx)

        self._linearform = [LinearForm(v=vi, dx=self.dx) for vi in self.v]

    def integrate(self, weakform, kwargs=None, parallel=False):
        r"""Return evaluated (but not assembled) integrals.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, **kwargs)``.
        kwargs : dict or None, optional
            Optional named arguments for callable weakform (default is None).
        parallel : bool, optional
            Flag to activate parallel threading (default is False).

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) vector values.
        """

        if kwargs is None:
            kwargs = {}

        return [
            form.integrate(fun, kwargs, parallel=parallel)
            for form, fun in zip(self._linearform, weakform)
        ]

    def assemble(self, weakform, kwargs=None, parallel=False):
        r"""Return the assembled integral as vector.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, **kwargs)``.
        kwargs : dict or None, optional
            Optional named arguments for callable weakform (default is None).
        parallel : bool, optional
            Flag to activate parallel threading (default is False).

        Returns
        -------
        values : csr_matrix
            The assembled vector.
        """

        values = self.integrate(weakform, kwargs, parallel=parallel)

        return self._form.assemble(values)


class BilinearFormExpression:
    r"""A bilinear form object with methods for integration and assembly of
    matrices where ``v`` is a mixed-basis object on mixed-fields.

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Parameters
    ----------
    v : Basis
        Basis (shape) functions and gradients of a field.
    u : Basis
        Basis (shape) functions and gradients of a field.

    """

    def __init__(self, v, u):
        self.v = v
        self.u = u
        self.dx = self.v.field[0].region.dV

        self.nv = len(v.field.fields)
        self.i, self.j = np.triu_indices(self.nv)

        self._form = IntegralForm(
            fun=np.zeros(len(self.i)),
            v=self.v.field,
            dV=self.dx,
            u=self.u.field,
        )

        self._bilinearform = []

        for a, (i, j) in enumerate(zip(self.i, self.j)):
            self._bilinearform.append(
                BilinearForm(
                    v=self.v[i],
                    u=self.u[j],
                    dx=self.dx,
                )
            )

    def integrate(self, weakform, kwargs=None, parallel=False, sym=False):
        r"""Return evaluated (but not assembled) integrals.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, u, **kwargs)``.
        kwargs : dict or None, optional
            Optional named arguments for callable weakform (default is None).
        parallel : bool, optional (default is False)
            Flag to activate parallel threading.

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) matrix values.
        """

        if kwargs is None:
            kwargs = {}

        return [
            form.integrate(fun, kwargs, parallel=parallel, sym=sym)
            for form, fun in zip(self._bilinearform, weakform)
        ]

    def assemble(self, weakform, kwargs=None, parallel=False, sym=False):
        r"""Return the assembled integral as matrix.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, u, **kwargs)``.
        kawargs : dict or None, optional
            Optional named arguments for callable weakform (default is None).
        parallel : bool, optional
            Flag to activate parallel threading (default is False).

        Returns
        -------
        values : csr_matrix
            The assembled sparse matrix.
        """

        values = self.integrate(weakform, kwargs, parallel=parallel, sym=sym)

        return self._form.assemble(values)
