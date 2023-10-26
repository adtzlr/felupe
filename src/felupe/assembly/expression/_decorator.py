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

from ._expression import FormExpression


def FormExpressionDecorator(
    v, u=None, grad_v=None, grad_u=None, dx=None, args=(), kwargs={}, parallel=False
):
    r"""A linear or bilinear form object as function decorator on a weak-form
    with methods for integration and assembly of vectors or sparse matrices.

    Linear Form:

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Bilinear Form:

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Parameters
    ----------
    v : Field or FieldMixed
        An object with interpolation or gradients of a field. May be
        updated during integration / assembly.
    u : Field or FieldMixed
        An object with interpolation or gradients of a field. May be
        updated during integration / assembly.
    grad_v : bool, optional (default is None)
        Flag to use the gradient of ``v``.
    grad_u : bool, optional (default is None)
        Flag to use the gradient of ``u``.
    dx : ndarray or None, optional (default is None)
        Array with (numerical) differential volumes.
    args : tuple, optional (default is ())
        Tuple with initial optional weakform-arguments. May be updated during
        integration / assembly.
    kwargs : dict, optional (default is {})
        Dictionary with initial optional weakform-keyword-arguments. May be
        updated during integration / assembly.

    Returns
    -------
    BaseForm
        A form object based on LinearForm, LinearFormMixed, BilinearForm or
        BilinearFormMixed with methods for integration and assembly.

    """

    def form(weakform):
        return FormExpression(
            weakform=weakform,
            v=v,
            u=u,
            grad_v=grad_v,
            grad_u=grad_u,
            dx=dx,
            args=args,
            kwargs=kwargs,
            parallel=parallel,
        )

    return form
