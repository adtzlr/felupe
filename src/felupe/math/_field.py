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

from ._tensor import dot, eigh, eigvalsh
from ._tensor import tovoigt as tensor_to_vector
from ._tensor import transpose


def displacement(field, dim=3):
    "Return the values of the first field."

    u = field[0].values
    return np.pad(u, ((0, 0), (0, dim - u.shape[1])))


def deformation_gradient(field):
    "Return the deformation gradient of the first field."
    return field[0].extract(grad=True, sym=False, add_identity=True)


def strain(field, fun=lambda stretch: np.log(stretch), tensor=True, asvoigt=False):
    r"""Return Lagrangian strain tensors or their principal values of the first field.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement as first field.
    fun : callable, optional
        A callable for the one-dimensional strain-stretch relation. Function signature
        must be ``lambda stretch: strain`` (default is the logarithmic strain-stretch
        relation ``lambda stretch: np.log(stretch)``).
    tensor : bool, optional
        Assemble and return the strain tensors if True or return their principal values
        only if False. Default is True.
    asvoigt : bool, optional
        Return the symmetric strain tensors in reduced vector (Voigt) storage. Default
        is False.

    Returns
    -------
    ndarray of shape (N, N, ...) tensor, (N!, ...) asvoigt or (N, ...) princ. values
        The strain tensors or their principal values.

    Notes
    -----
    The right Cauchy-Green deformation tensor is defined by the dot-products of the
    column vectors of the deformation gradient tensor and enables a quadratic length
    measure, see Eq. :eq:`right-cauchy-green-deformation-tensor`.

    .. math::
       :label: right-cauchy-green-deformation-tensor

       \boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}

    The principal stretches are obtained as the square-roots of the eigenvalues of the
    right Cauchy-Green deformation tensor, see Eq. :eq:`principal-stretches`.

    .. math::
       :label: principal-stretches

       \lambda^2_\alpha, \boldsymbol{N}_\alpha = \text{eig}\left( \boldsymbol{C} \right)

    The Lagrangian strain tensor is assembled with the one-dimensional strain-stretch
    relation from Eq. :eq:`principal-strain` and the eigenbases as dyadic products of
    the eigenvectors, see Eq. :eq:`strain-tensor`.

    .. math::
       :label: principal-strain

       E_\alpha = \text{f}\left( \lambda_\alpha \right)

    .. math::
       :label: strain-tensor

       \boldsymbol{E} = \sum_\alpha E_\alpha \
           \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

    """

    F = deformation_gradient(field)
    C = dot(transpose(F), F)

    if tensor:
        w, N = eigh(C)
        stretch = np.sqrt(w)
        tensor = np.einsum("a...,ai...,aj...->ij...", fun(stretch), N, N)
        if asvoigt:
            # double the off-diagonal items in Voigt-notation for strain tensors
            tensor = tensor_to_vector(tensor, strain=True)
        return tensor
    else:
        stretch = np.sqrt(eigvalsh(C))
        return fun(stretch)


def extract(field, grad=True, sym=False, add_identity=True):
    "Extract gradient or interpolated field values at quadrature points."
    return field.extract(grad=grad, sym=sym, add_identity=add_identity)


def values(field):
    "Return values of a field or a tuple of fields."

    return np.concatenate([f.values.ravel() for f in field.fields])


def norm(array, axis=None):
    "Calculate the norm of an array or the norms of a list of arrays."
    if isinstance(array, list):
        return np.array([np.linalg.norm(arr, axis=axis) for arr in array])
    else:
        return np.linalg.norm(array, axis=axis)


def interpolate(field):
    "Interpolate method of a field."
    return field.interpolate()


def grad(x, **kwargs):
    "Return the gradient of a field or the gradient of a basis-array."
    if callable(x.grad):
        return x.grad(**kwargs)
    else:
        return x.grad
