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
from scipy.sparse import bmat, vstack

from ..field._axi import FieldAxisymmetric
from ..field._base import Field
from ..field._planestrain import FieldPlaneStrain
from ._axi import IntegralFormAxisymmetric
from ._cartesian import IntegralFormCartesian


class IntegralForm:
    r"""Mixed-field integral form container with methods for integration and assembly.
    It is constructed by a list of function results ``[fun, ...]``, a list of test
    fields ``[v, ...]``, differential volumes ``dV`` and optionally a list of trial
    fields ``[u, ...]``. For the lists of fields, gradients may be passed by setting the
    respective list items in ``grad_v`` and ``grad_u`` to True.

    Linearform
    ----------
    A linear form is either defined by the dot product of a given (vector-valued)
    function :math:`\boldsymbol{f}` and the (vector) field :math:`\boldsymbol{v}`

    ..  math::

        L(\boldsymbol{v}) = \int_V \boldsymbol{f} \cdot \boldsymbol{v} ~ dV


    or by the double-dot product of a (matrix-valued) function :math:`\boldsymbol{F}`
    and the gradient of the field values :math:`\boldsymbol{\nabla v}`.

    ..  math::

        L(\boldsymbol{v}) = \int_V \boldsymbol{F} : \boldsymbol{\nabla v} ~ dV

    Bilinearform
    ------------
    A bilinear form is either defined by the dot products of a given (matrix-valued)
    function :math:`\boldsymbol{F}` and the (vector) fields :math:`\boldsymbol{v}` and
    :math:`\boldsymbol{u}`

    ..  math::

        a(\boldsymbol{v}, \boldsymbol{u}) =
            \int_V \boldsymbol{v} \cdot \boldsymbol{F} \cdot \boldsymbol{u} ~ dV


    or by the double-dot products of a (tensor-valued) function :math:`\boldsymbol{F}`
    and the field values :math:`\boldsymbol{v}` and :math:`\boldsymbol{u}` or their
    gradients :math:`\boldsymbol{\nabla v}` and :math:`\boldsymbol{\nabla u}`.

    ..  math::

        a(\boldsymbol{v}, \boldsymbol{u}) &=
            \int_V \boldsymbol{\nabla v} : \boldsymbol{F} \cdot \boldsymbol{u} ~ dV

        a(\boldsymbol{v}, \boldsymbol{u}) &=
            \int_V \boldsymbol{v} \cdot \boldsymbol{F} : \boldsymbol{\nabla u} ~ dV

        a(\boldsymbol{v}, \boldsymbol{u}) &=
            \int_V \boldsymbol{\nabla v} : \mathbb{F} : \boldsymbol{\nabla u} ~ dV


    Arguments
    ---------
    fun : list of array
        The list of pre-evaluated function arrays.
    v : list of Field, FieldAxisymmetric or FieldPlaneStrain
        The list of test fields.
    dV : array
        The differential volumes.
    u : list of Field, FieldAxisymmetric or FieldPlaneStrain, optional
        If a list with fields is passed, bilinear forms are created (default is None).
    grad_v : list of bool or None, optional
        List with flags to activate the gradients on the test fields ``v`` (default is
        None which enforces True for the first field and False for all following fields)
        .
    grad_u : list of bool or None, optional
        Flag to activate the gradient on the trial field ``u`` (default is None which
        enforces True for the first field and False for all following fields).
    """

    def __init__(self, fun, v, dV, u=None, grad_v=None, grad_u=None):
        self.fun = fun
        self.v = v.fields
        self.nv = len(self.v)
        self.dV = dV

        if u is not None:
            self.u = u.fields
            self.nu = len(self.u)
        else:
            self.u = None
            self.nu = None

        IntForm = {
            Field: IntegralFormCartesian,
            FieldPlaneStrain: IntegralFormCartesian,
            FieldAxisymmetric: IntegralFormAxisymmetric,
        }[type(self.v[0])]

        if isinstance(self.v[0], FieldAxisymmetric):
            for i in range(1, len(self.v)):
                self.v[i].radius = self.v[0].radius

        if grad_v is None:
            self.grad_v = np.zeros_like(self.v, dtype=bool)
            self.grad_v[0] = True
        else:
            self.grad_v = grad_v

        if grad_u is None and u is not None:
            self.grad_u = np.zeros_like(self.u, dtype=bool)
            self.grad_u[0] = True
        else:
            self.grad_u = grad_u

        self.forms = []

        if len(fun) == self.nv and u is None:
            # LinearForm
            self.mode = 1
            self.i = np.arange(self.nv)
            self.j = np.zeros_like(self.i)

            for fun, v, grad_v in zip(self.fun, self.v, self.grad_v):
                f = IntForm(fun=fun, v=v, dV=self.dV, grad_v=grad_v)
                self.forms.append(f)

        elif len(fun) == np.sum(1 + np.arange(self.nv)) and u is not None:
            # BilinearForm
            self.mode = 2
            self.i, self.j = np.triu_indices(self.nv)

            for a, (i, j) in enumerate(zip(self.i, self.j)):
                f = IntForm(
                    self.fun[a],
                    v=self.v[i],
                    dV=self.dV,
                    u=self.u[j],
                    grad_v=self.grad_v[i],
                    grad_u=self.grad_u[j],
                )
                self.forms.append(f)
        else:
            raise ValueError("Unknown input format.")

    def assemble(self, values=None, parallel=False, block=True):
        out = []

        if values is None:
            values = [None] * len(self.forms)

        for val, form in zip(values, self.forms):
            out.append(form.assemble(val, parallel=parallel))

        if block and self.mode == 2:
            K = np.zeros((self.nv, self.nv), dtype=object)
            for a, (i, j) in enumerate(zip(self.i, self.j)):
                K[i, j] = out[a]
                if i != j:
                    K[j, i] = out[a].T

            return bmat(K).tocsr()

        if block and self.mode == 1:
            return vstack(out).tocsr()

        else:
            return out

    def integrate(self, parallel=False):
        out = []
        for form in self.forms:
            out.append(form.integrate(parallel=parallel))

        return out
