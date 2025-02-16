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

from copy import deepcopy

import numpy as np

from ..math import identity
from ..math import sym as symmetric
from ..region import RegionVertex
from ._container import FieldContainer
from ._indices import Indices


class Field:
    r"""A Field on points of a :class:`~felupe.Region` with dimension ``dim`` and
    initial point ``values``.

    Parameters
    ----------
    region : Region
        The region on which the field will be created.
    dim : int, optional
        The dimension of the field  (default is 1).
    values : float or array
        A single value for all components of the field or an array of shape
        `(region.mesh.npoints, dim)`. Default is 0.0.
    dtype : data-type or None, optional
        Data-type of the array containing the field values.
    **kwargs : dict, optional
        Extra class attributes for the field.

    Notes
    -----
    A slice of this field directly accesses the field-values as 1d-array. The
    interpolation method returns the field values evaluated at the numeric integration
    points ``q`` for each cell ``c`` in the region (so-called *trailing axes*).

    ..  math::

        u_{i(qc)} = \hat{u}_{ai}\ h_{a(qc)}

    The gradient method returns the gradient of the field values w.r.t. the
    undeformed mesh point coordinates, evaluated at the integration points of
    all cells in the region.

    ..  math::

        \left( \frac{\partial u_i}{\partial X_J} \right)_{(qc)} =
            \hat{u}_{ai} \left( \frac{\partial h_a}{\partial X_J} \right)_{(qc)}

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>>
        >>> u = displacement.interpolate()
        >>> dudX = displacement.grad()

    To obtain deformation-related quantities like the right Cauchy-Green deformation
    tensor or the principal stretches, use the math-helpers from FElupe. These
    functions operate on arrays with trailing axes.

    ..  math::

        \boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}

    ..  pyvista-plot::
        :context:

        >>> from felupe.math import dot, transpose, eigvalsh, sqrt
        >>>
        >>> F = displacement.extract(grad=True, add_identity=True)
        >>> C = dot(transpose(F), F)
        >>> λ = sqrt(eigvalsh(C))

    """

    def __init__(self, region, dim=1, values=0.0, dtype=None, **kwargs):
        self.region = region
        self.dim = dim
        self.shape = self.region.quadrature.npoints, self.region.mesh.ncells

        # set optional user-defined attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # init values
        if isinstance(values, np.ndarray):
            if values.size == dim:
                shape = (region.mesh.npoints, dim)
                self.values = np.broadcast_to(values.reshape(-1, dim), shape)
            elif len(values) == region.mesh.npoints:
                self.values = values
            else:
                raise ValueError("Wrong shape of values.")

            if dtype is not None:
                self.values = self.values.astype(dtype)

        else:  # scalar value
            self.values = np.full(
                shape=(region.mesh.npoints, dim), fill_value=values, dtype=dtype
            )

        eai, ai = self._indices_per_cell(self.region.mesh.cells, dim)
        self.indices = Indices(eai, ai, region, dim)

    def _indices_per_cell(self, cells, dim):
        "Calculate pre-defined indices for sparse matrices."

        # index of cell "c", point "a" and component "i"
        cai = (
            dim * np.repeat(cells, dim) + np.tile(np.arange(dim), cells.size)
        ).reshape(*cells.shape, dim)
        # store indices as (rows, cols) (note: sparse-matrices are always 2d)
        ai = (cai.ravel(), np.zeros_like(cai.ravel()))

        return cai, ai

    @classmethod
    def from_mesh_container(cls, mesh_container, dim=None, values=0.0):
        "Create a Field on a vertex mesh from a mesh container."

        mesh = mesh_container.as_vertex_mesh()
        region = RegionVertex(mesh)

        if dim is None:
            dim = mesh.dim

        return cls(region, dim=dim, values=values)

    def grad(self, sym=False, dtype=None, out=None, order="C"):
        r"""Gradient as partial derivative of field values w.r.t. undeformed
        coordinates, evaluated at the integration points of all cells in the region.
        Optionally, the symmetric part the gradient is evaluated.

        ..  math::

            \left( \frac{\partial u_i}{\partial X_J} \right)_{(qc)} =
                \hat{u}_{ai} \left( \frac{\partial h_a}{\partial X_J} \right)_{(qc)}

        Parameters
        ----------
        sym : bool, optional
            Calculate the symmetric part of the gradient (default is False).
        dtype : data-type or None, optional
            If provided, forces the calculation to use the data type specified. Default
            is None.
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the output. 'C' means it should be C
            contiguous. 'F' means it should be Fortran contiguous, 'A' means it should
            be 'F' if the inputs are all 'F', 'C' otherwise. 'K' means it should be as
            close to the layout as the inputs as is possible, including arbitrarily
            permuted axes. Default is 'C'.

        Returns
        -------
        ndarray of shape (i, j, q, c)
            Gradient as partial derivative of field value components ``i`` at points
            w.r.t. the undeformed coordinates ``j``, evaluated at the quadrature points
            ``q`` of cells ``c`` in the region.
        """

        # gradient dudX_IJqc as partial derivative of field values at points "aI"
        # w.r.t. undeformed coordinates "J" evaluated at quadrature point "q"
        # for each cell "c"
        g = np.einsum(
            "ca...,aJqc->...Jqc",
            self.values[self.region.mesh.cells],
            self.region.dhdX,
            dtype=dtype,
            out=out,
            order=order,
        )

        if sym:
            return symmetric(g, out=g)
        else:
            return g

    def hess(self, dtype=None, out=None, order="C"):
        r"""Hessian as second partial derivative of field values w.r.t. undeformed
        coordinates, evaluated at the integration points of all cells in the region.

        ..  math::

            \left( \frac{\partial^2 u_i}{\partial X_J~\partial X_K} \right)_{(qc)} =
                \hat{u}_{ai} \left(
                    \frac{\partial^2 h_a}{\partial X_J~\partial X_K}
                \right)_{(qc)}

        Parameters
        ----------
        dtype : data-type or None, optional
            If provided, forces the calculation to use the data type specified. Default
            is None.
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the output. 'C' means it should be C
            contiguous. 'F' means it should be Fortran contiguous, 'A' means it should
            be 'F' if the inputs are all 'F', 'C' otherwise. 'K' means it should be as
            close to the layout as the inputs as is possible, including arbitrarily
            permuted axes. Default is 'C'.

        Returns
        -------
        ndarray of shape (i, j, k, q, c)
            Hessian as partial derivative of field value components ``i`` at points
            w.r.t. the undeformed coordinates ``j`` and ``k``, evaluated at the
            quadrature points ``q`` of cells ``c`` in the region.
        """

        # hessian d2udXdX_IJKqc as second partial derivative of field values at points
        # "aI" w.r.t. undeformed coordinates "J" and "K" evaluated at quadrature point
        # "q" for each cell "c"
        h = np.einsum(
            "ca...,aJKqc->...JKqc",
            self.values[self.region.mesh.cells],
            self.region.d2hdXdX,
            dtype=dtype,
            out=out,
            order=order,
        )

        return h

    def interpolate(self, dtype=None, out=None, order="C"):
        r"""Interpolate field values located at mesh-points to the quadrature points
        ``q`` of cells ``c`` in the region.

        ..  math::

            u_{i(qc)} = \hat{u}_{ai}\ h_{a(qc)}

        Arguments
        ---------
        dtype : data-type or None, optional
            If provided, forces the calculation to use the data type specified. Default
            is None.
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the output. 'C' means it should be C
            contiguous. 'F' means it should be Fortran contiguous, 'A' means it should
            be 'F' if the inputs are all 'F', 'C' otherwise. 'K' means it should be as
            close to the layout as the inputs as is possible, including arbitrarily
            permuted axes. Default is 'C'.

        Returns
        -------
        ndarray of shape (i, q, c)
            Interpolated field value components ``i``, evaluated at the quadrature
            points ``q`` of each cell ``c`` in the region.
        """

        # interpolated field values "aI"
        # evaluated at quadrature point "q"
        # for cell "c"
        return np.einsum(
            "ca...,aqc->...qc",
            self.values[self.region.mesh.cells],
            self.region.h,
            dtype=dtype,
            out=out,
            order=order,
        )

    def extract(
        self, grad=True, sym=False, add_identity=True, dtype=None, out=None, order="C"
    ):
        """Generalized extraction method which evaluates either the gradient or the
        field values at the integration points of all cells in the region. Optionally,
        the symmetric part of the gradient is evaluated and/or the identity matrix is
        added to the gradient.

        Arguments
        ---------
        grad : bool, optional
            Flag for gradient evaluation (default is True).
        sym : bool, optional
            Flag for symmetric part if the gradient is evaluated (default is False).
        add_identity : bool, optional
            Flag for the addition of the identity matrix
            if the gradient is evaluated (default is True).
        dtype : data-type or None, optional
            If provided, forces the calculation to use the data type specified. Default
            is None.
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the output. 'C' means it should be C
            contiguous. 'F' means it should be Fortran contiguous, 'A' means it should
            be 'F' if the inputs are all 'F', 'C' otherwise. 'K' means it should be as
            close to the layout as the inputs as is possible, including arbitrarily
            permuted axes. Default is 'C'.

        Returns
        -------
        ndarray
            (Symmetric) gradient or interpolated field values evaluated at
            the integration points of each cell in the region.

        See Also
        --------
        Field.interpolate :
        Field.grad :
        """

        if grad:
            gr = self.grad(out=out, dtype=dtype, order=order)

            if sym:
                gr = symmetric(gr, out=gr)

            if add_identity:
                gr = np.add(gr, identity(gr, dtype=dtype), out=gr)

            return gr
        else:
            return self.interpolate(out=out, dtype=dtype, order=order)

    def copy(self):
        "Return a copy of the field."
        return deepcopy(self)

    def fill(self, a):
        "Fill all field values with a scalar value."
        self.values.fill(a)

    def as_container(self):
        "Create a :class:`~felupe.FieldContainer` with the field."

        return FieldContainer([self])

    def __add__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values += newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values += newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __sub__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values -= newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values -= newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __mul__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values *= newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values *= newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __truediv__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values /= newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values /= newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __iadd__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            self.values += newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values += newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __isub__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            self.values -= newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values -= newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __imul__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            self.values *= newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values *= newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __itruediv__(self, newvalues):
        if isinstance(newvalues, np.ndarray):
            self.values /= newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values /= newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __getitem__(self, dof):
        """Slice-based access to a 1d-representation of field values by
        a list of degrees of freedom `dof`."""

        return self.values.ravel()[dof]

    def __and__(self, field):
        fields = [field]
        if isinstance(field, FieldContainer):
            fields = field.fields
        elif field is None:
            fields = []

        return FieldContainer([self, *fields])
