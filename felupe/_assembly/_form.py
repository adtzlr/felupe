# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:26:08 2022

@author: adutz
"""
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
from threading import Thread

from ._base import IntegralForm
from ._mixed import IntegralFormMixed


class LinearForm:
    r"""A linear form object with methods for integration and assembly of
    vectors.

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Parameters
    ----------
    v : Basis
        An object with basis functions (gradients) of a field.
    grad_v : bool, optional (default is False)
        Flag to use the gradient of ``v``.


    """

    def __init__(self, v, grad_v=False, dx=None):
        self.v = v
        self.grad_v = grad_v
        if dx is None:
            self.dx = v.field.region.dV
        else:
            self.dx = dx

        self._form = IntegralForm(fun=None, v=v.field, dV=self.dx, grad_v=grad_v)

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

    def assemble(self, weakform, args=(), kwargs={}, parallel=False):
        r"""Return the assembled integral as vector.

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
        values : csr_matrix
            The assembled vector.
        """

        values = self.integrate(weakform, args, kwargs, parallel=parallel)

        return self._form.assemble(values)


class BilinearForm:
    r"""A bilinear form object with methods for integration and assembly of
    matrices.

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Parameters
    ----------
    v : Basis
        An object with basis functions (gradients) of a field.
    grad_v : bool, optional (default is False)
        Flag to use the gradient of ``v``.
    u : Basis
        An object with basis function (gradients) of a field.
    grad_u : bool, optional (default is False)
        Flag to use the gradient of ``u``.

    """

    def __init__(self, v, u, grad_v=False, grad_u=False, dx=None):
        self.v = v
        self.grad_v = grad_v
        self.u = u
        self.grad_u = grad_u
        if dx is None:
            self.dx = v.field.region.dV
        else:
            self.dx = dx

        self._form = IntegralForm(None, v.field, self.dx, u.field, grad_v, grad_u)

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

        values = np.zeros((len(v), v.shape[-3], len(u), *u.shape[-3:]))

        if not parallel:

            for a, vbasis in enumerate(v):
                for i, vb in enumerate(vbasis):

                    for b, ubasis in enumerate(u):
                        for j, ub in enumerate(ubasis):

                            values[a, i, b, j] = (
                                weakform(vb, ub, *args, **kwargs) * self.dx
                            )

        else:

            idx_a, idx_i, idx_b, idx_j = np.indices(values.shape[:4])
            aibj = zip(idx_a.ravel(), idx_i.ravel(), idx_b.ravel(), idx_j.ravel())

            def contribution(values, a, i, b, j, args, kwargs):
                values[a, i, b, j] = (
                    weakform(v[a, i], u[b, j], *args, **kwargs) * self.dx
                )

            threads = [
                Thread(target=contribution, args=(values, a, i, b, j, args, kwargs))
                for a, i, b, j in aibj
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

        return values.sum(-2)

    def assemble(self, weakform, args=(), kwargs={}, parallel=False):
        r"""Return the assembled integral as sparse matrix.

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
        values : csr_matrix
            The assembled sparse matrix.
        """

        values = self.integrate(weakform, args, kwargs, parallel=parallel)

        return self._form.assemble(values)


class LinearFormMixed:
    r"""A linear form object with methods for integration and assembly of
    vectors where ``v`` is a mixed-basis object on mixed-fields.

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Parameters
    ----------
    v : BasisMixed
        An object with basis functions (gradients) of a field.
    grad_v : tuple of bool, optional (default is None)
        Flag to use the gradients of ``v``.


    """

    def __init__(self, v, grad_v=None):
        self.v = v
        self.dx = self.v.field[0].region.dV
        self._form = IntegralFormMixed(
            np.zeros(len(v.field.fields)), self.v.field, self.dx, grad_v=grad_v
        )

        if grad_v is None:
            self.grad_v = np.zeros_like(self.v.field.fields, dtype=bool)
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
        kawargs : dict, optional
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
        kawargs : dict, optional
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


class BilinearFormMixed:
    r"""A bilinear form object with methods for integration and assembly of
    matrices where ``v`` is a mixed-basis object on mixed-fields.

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Parameters
    ----------
    v : BasisMixed
        An object with basis functions (gradients) of a field.
    grad_v : tuple of bool, optional (default is None)
        Tuple of flags to use the gradients of ``v``.
    u : Basis
        An object with basis function (gradients) of a field.
    grad_u : tuple of bool, optional (default is None)
        Flag to use the gradients of ``u``.

    """

    def __init__(self, v, u, grad_v=None, grad_u=None):
        self.v = v
        self.u = u
        self.dx = self.v.field[0].region.dV

        self.nv = len(v.field.fields)
        self.i, self.j = np.triu_indices(self.nv)

        def _set_first_grad_true(grad, fields):
            if grad is None:
                grad = np.zeros_like(fields, dtype=bool)
                grad[0] = True
            return grad

        self.grad_v = _set_first_grad_true(grad_v, self.v.field.fields)
        self.grad_u = _set_first_grad_true(grad_u, self.u.field.fields)

        self._form = IntegralFormMixed(
            fun=np.zeros(len(self.i)),
            v=self.v.field,
            dV=self.dx,
            u=self.u.field,
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
            Integrated (but not assembled) matrix values.
        """

        return [
            form.integrate(fun, args, kwargs, parallel=parallel)
            for form, fun in zip(self._bilinearform, weakform)
        ]

    def assemble(self, weakform, args=(), kwargs={}, parallel=False):
        r"""Return the assembled integral as matrix.

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
        values : csr_matrix
            The assembled matrix.
        """

        values = self.integrate(weakform, args, kwargs, parallel=parallel)

        return self._form.assemble(values)
