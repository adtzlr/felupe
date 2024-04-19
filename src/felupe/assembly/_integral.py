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
from ..field._dual import FieldDual
from ..field._planestrain import FieldPlaneStrain
from ._axi import IntegralFormAxisymmetric
from ._cartesian import IntegralFormCartesian


class IntegralForm:
    r"""Mixed-field integral form container with methods for integration and assembly.
    It is constructed by a list of function results ``[fun, ...]``, a list of test
    fields ``[v, ...]``, differential volumes ``dV`` and optionally a list of trial
    fields ``[u, ...]``. For the lists of fields, gradients may be passed by setting the
    respective list items in ``grad_v`` and ``grad_u`` to True.

    Arguments
    ---------
    fun : list of array
        The list of pre-evaluated function arrays.
    v : FieldContainer
        The field container for the test fields.
    dV : array
        The differential volumes.
    u : FieldContainer, optional
        The field container for the trial fields. If a field container is passed,
        bilinear forms are created (default is None).
    grad_v : list of bool or None, optional
        List with flags to activate the gradients on the test fields ``v`` (default is
        None which enforces True for the first field and False for all following fields)
        .
    grad_u : list of bool or None, optional
        Flag to activate the gradient on the trial field ``u`` (default is None which
        enforces True for the first field and False for all following fields).

    Notes
    -----

    Linearform
    ~~~~~~~~~~

    A linear form is either defined by the dot product of a given (vector-valued)
    function :math:`\boldsymbol{f}` and the (vector) field :math:`\boldsymbol{v}`

    ..  math::

        L(\boldsymbol{v}) = \int_\Omega \boldsymbol{f} \cdot \boldsymbol{v} ~ dV


    or by the double-dot product of a (matrix-valued) function :math:`\boldsymbol{f}`
    and the gradient of the field values :math:`\boldsymbol{\nabla v}`.

    ..  math::

        L(\boldsymbol{v}) = \int_\Omega \boldsymbol{f} : \boldsymbol{\nabla v} ~ dV

    Bilinearform
    ~~~~~~~~~~~~

    A bilinear form is either defined by the dot products of a given (matrix-valued)
    function :math:`\boldsymbol{f}` and the (vector) fields :math:`\boldsymbol{v}` and
    :math:`\boldsymbol{u}`

    ..  math::

        a(\boldsymbol{v}, \boldsymbol{u}) =
            \int_\Omega \boldsymbol{v} \cdot \boldsymbol{f} \cdot \boldsymbol{u} ~ dV


    or by the double-dot products of a (tensor-valued) function :math:`\boldsymbol{f}`
    and the field values :math:`\boldsymbol{v}` and :math:`\boldsymbol{u}` or their
    gradients :math:`\boldsymbol{\nabla v}` and :math:`\boldsymbol{\nabla u}`.

    ..  math::

        a(\boldsymbol{v}, \boldsymbol{u}) &=
            \int_\Omega \boldsymbol{\nabla v} : \boldsymbol{f} \cdot \boldsymbol{u} ~ dV

        a(\boldsymbol{v}, \boldsymbol{u}) &=
            \int_\Omega \boldsymbol{v} \cdot \boldsymbol{f} : \boldsymbol{\nabla u} ~ dV

        a(\boldsymbol{v}, \boldsymbol{u}) &= \int_\Omega
            \boldsymbol{\nabla v} : \boldsymbol{f} : \boldsymbol{\nabla u} ~ dV

    Examples
    --------
    The stiffness matrix for a linear-elastic solid body on a cube out of hexahedrons
    is assembled as follows. First, the mesh, the region and the field objects are
    created.

    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube(n=11)
    >>> region = fem.RegionHexahedron(mesh)
    >>> displacement = fem.Field(region, dim=3)
    >>> field = fem.FieldContainer([displacement])

    The (constant) fourth-order elasticity tensor for linear-elasticity is created with
    two trailing axes, one for each quadrature point and one for each cell. Due to the
    fact that the elasticity tensor is constant, broadcasting is used for the trailing
    axes.

    ..  math::

        \frac{\boldsymbol{\partial \sigma}}{\partial \boldsymbol{\varepsilon}} =
        2 \mu \ \boldsymbol{I} \odot \boldsymbol{I} + \gamma \ \boldsymbol{I} \otimes
        \boldsymbol{I}


    >>> import numpy as np
    >>> from felupe.math import cdya, dya
    >>>
    >>> mu, lmbda = 1.0, 2.0
    >>> I = np.eye(3).reshape(3, 3, 1, 1)
    >>> dSdE = 2 * mu * cdya(I, I) + lmbda * dya(I, I)
    >>> dSdE.shape
    (3, 3, 3, 3, 1, 1)

    The integral form object provides methods for cell-wise stiffness matrices via its
    integrate-method and the system stiffness matrix via the assembly-method.

    ..  math::

        \delta W_{int} = -\int_v \delta\boldsymbol{\varepsilon} :
            \frac{\boldsymbol{\partial \sigma}}{\partial \boldsymbol{\varepsilon}} :
            \boldsymbol{\varepsilon} ~ dv


    ..  math::

        \delta\boldsymbol{\varepsilon} &= \text{sym}(\boldsymbol{\nabla v})

        \boldsymbol{\varepsilon} &= \text{sym}(\boldsymbol{\nabla u})

        \boldsymbol{\nabla v} &= \frac{\partial\boldsymbol{v}}{\partial\boldsymbol{x}}

        \boldsymbol{\nabla u} &= \frac{\partial\boldsymbol{u}}{\partial\boldsymbol{x}}

        \left( \frac{\partial v_i}{\partial x_j} \right)_{(qc)} &= \hat{v}_{ai}
            \left( \frac{\partial h_a}{\partial x_j} \right)_{(qc)}


    ..  math::

        \hat{K}_{aibk(c)} = \left( \frac{\partial h_a}{\partial x_J} \right)_{(qc)}
            \left( \frac{\partial \sigma_{ij}}{\partial \varepsilon_{kl}} \right)_{(qc)}
            \left( \frac{\partial h_b}{\partial x_L} \right)_{(qc)} ~ dv_{(qc)}


    >>> form = fem.IntegralForm([dSdE], v=field, dV=region.dV, u=field)
    >>> values = form.integrate(parallel=False)
    >>> values[0].shape
    (8, 3, 8, 3, 1000)

    The cell-wise stiffness matrices are re-used to assemble the sparse system stiffness
    matrix. The parallel keyword argument enables a threaded assembly.

    ..  math::

        \Delta\delta W_{int} = -\hat{\boldsymbol{v}} : \hat{\boldsymbol{K}} :
            \hat{\boldsymbol{u}}


    >>> K = form.assemble(values=values, parallel=False)
    >>> K.shape
    (3993, 3993)

    See Also
    --------
    felupe.IntegralFormAxisymmetric : An Integral Form for axisymmetric fields.
    felupe.IntegralFormCartesian : Single-field integral form.
    felupe.Form : A weak-form expression decorator.

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
            FieldDual: IntegralFormCartesian,
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

    def assemble(self, values=None, parallel=False, block=True, out=None):
        res = []

        if values is None:
            values = [None] * len(self.forms)

        for val, form in zip(values, self.forms):
            res.append(form.assemble(val, parallel=parallel, out=out))

        if block and self.mode == 2:
            K = np.zeros((self.nv, self.nv), dtype=object)
            for a, (i, j) in enumerate(zip(self.i, self.j)):
                K[i, j] = res[a]
                if i != j:
                    K[j, i] = res[a].T

            return bmat(K).tocsr()

        if block and self.mode == 1:
            return vstack(res).tocsr()

        else:
            return res

    def integrate(self, parallel=False, out=None):
        if out is None:
            out = [None] * len(self.forms)

        for i, form in enumerate(self.forms):
            out[i] = form.integrate(parallel=parallel, out=out[i])

        return out
