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
    r"""A linear form object with methods for integration and assembly of
    vectors.

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Parameters
    ----------
    v : BasisField
        An object with basis functions (gradients) of a field.
    grad_v : bool, optional (default is False)
        Flag to use the gradient of ``v``.
    dx : ndarray or None, optional (default is None)
        Array with (numerical) differential volumes.

    """

    def __init__(self, v, grad_v=False, dx=None):
        self.v = v
        self.grad_v = grad_v
        self.dx = dx

        self._form = IntegralFormCartesian(
            fun=None, v=v.field, dV=self.dx, grad_v=grad_v
        )

    def integrate(self, weakform, args=(), kwargs={}, parallel=False):
        r"""Return evaluated (but not assembled) integrals.

        Parameters
        ----------
        weakform : callable
            A callable function ``weakform(v, *args, **kwargs)``.
        args : tuple, optional
            Optional arguments for callable weakform
        kawargs : dict, optional
            Optional named arguments for callable weakform
        parallel : bool, optional (default is False)
            Flag to activate parallel threading.

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) vector values.
        """

        if self.grad_v:
            v = self.v.grad
        else:
            v = self.v.basis

        values = np.zeros((len(v), *v.shape[-3:]))

        if not parallel:
            for a, vbasis in enumerate(v):
                for i, vb in enumerate(vbasis):
                    values[a, i] = weakform(vb, *args, **kwargs) * self.dx

        else:
            idx_a, idx_i = np.indices(values.shape[:2])
            ai = zip(idx_a.ravel(), idx_i.ravel())

            def contribution(values, a, i, args, kwargs):
                values[a, i] = weakform(v[a, i], *args, **kwargs) * self.dx

            threads = [
                Thread(target=contribution, args=(values, a, i, args, kwargs))
                for a, i in ai
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

        return values.sum(-2)
