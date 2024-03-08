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


class LinearForm:
    r"""A linear form object with methods for integration and assembly of vectors.

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Parameters
    ----------
    v : BasisField
        Basis (shape) functions and gradients of a field.
    dx : ndarray or None, optional
        Array with differential volumes (default is None).
    """

    def __init__(self, v, dx=None, **kwargs):
        self.v = v
        self.dx = dx
        self._form = IntegralFormCartesian(fun=None, v=v.field, dV=self.dx, **kwargs)

    def integrate(self, weakform, kwargs={}, parallel=False):
        r"""Return evaluated (but not assembled) integrals.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, **kwargs)``.
        kawargs : dict, optional
            Optional named arguments for callable weakform
        parallel : bool, optional (default is False)
            Flag to activate parallel threading.

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) vector values.
        """

        values = np.zeros(
            (len(self.v.basis), self.v.basis.shape[-4], *self.v.basis.shape[-2:])
        )

        if not parallel:
            for a, vbasis in enumerate(self.v.basis):
                for i, vb in enumerate(vbasis):
                    v = type(self.v.basis)(vb, self.v.basis.grad[a, i])
                    values[a, i] = weakform(v, **kwargs) * self.dx

        else:
            idx_a, idx_i = np.indices(values.shape[:2])
            ai = zip(idx_a.ravel(), idx_i.ravel())

            def contribution(values, a, i, kwargs):
                v = type(self.v.basis)(self.v.basis[a, i], self.v.basis.grad[a, i])
                values[a, i] = weakform(v, **kwargs) * self.dx

            threads = [
                Thread(target=contribution, args=(values, a, i, kwargs)) for a, i in ai
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

        return values.sum(-2)
