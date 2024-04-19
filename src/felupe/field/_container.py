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

from ..tools._plot import ViewField
from ._evaluate import EvaluateFieldContainer


class FieldContainer:
    """A container for fields which holds a list or a tuple of :class:`Field`
    instances.

    Parameters
    ----------
    fields : list or tuple of Field, FieldAxisymmetric or FieldPlaneStrain
        List with fields. The region is linked to the first field.

    Attributes
    ----------
    evaluate : field.EvaluateFieldContainer
        Methods to evaluate the deformation gradient and strain measures, see
        :class:`~felupe.field.EvaluateFieldContainer` for details on the provided
        methods.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube(n=3)
    >>> region = fem.RegionHexahedron(mesh)
    >>> region_dual = fem.RegionConstantHexahedron(mesh.dual(points_per_cell=1))
    >>> displacement = fem.Field(region, dim=3)
    >>> pressure = fem.Field(region_dual)
    >>> field = fem.FieldContainer([displacement, pressure])
    >>> field
    <felupe FieldContainer object>
      Number of fields: 2
      Dimension of fields:
        Field: 3
        Field: 1

    A new :class:`~felupe.FieldContainer` is also created by one of the logical-and
    combinations of a :class:`~felupe.Field`, :class:`~felupe.FieldAxisymmetric`,
    :class:`~felupe.FieldPlaneStrain` or :class:`~felupe.FieldContainer`.

    >>> displacement & pressure
    <felupe FieldContainer object>
      Number of fields: 2
      Dimension of fields:
        Field: 3
        Field: 1

    >>> volume_ratio = fem.Field(region_dual)
    >>> field & volume_ratio  # displacement & pressure & volume_ratio
    <felupe FieldContainer object>
      Number of fields: 3
      Dimension of fields:
        Field: 3
        Field: 1
        Field: 1

    See Also
    --------
    felupe.Field : Field on points of a :class:`~felupe.Region` with dimension ``dim``
        and initial point ``values``.
    felupe.FieldAxisymmetric : An axisymmetric :class:`~felupe.Field` on points of a
        two dimensional :class:`~felupe.Region` with dimension ``dim`` (default is 2)
        and initial point ``values`` (default is 0).
    felupe.FieldPlaneStrain : A plane strain :class:`~felupe.Field` on points of a
        two dimensional :class:`~felupe.Region` with dimension ``dim`` (default is 2)
        and initial point ``values`` (default is 0).

    """

    def __init__(self, fields):
        self.fields = fields
        self.region = fields[0].region

        # get sizes of fields and calculate offsets
        self.fieldsizes = [f.indices.dof.size for f in self.fields]
        self.offsets = np.cumsum(self.fieldsizes)[:-1]

        self.evaluate = EvaluateFieldContainer(self)

    def __repr__(self):
        header = "<felupe FieldContainer object>"
        size = f"  Number of fields: {len(self.fields)}"
        fields_header = "  Dimension of fields:"
        fields = [
            f"    {type(field).__name__}: {field.dim}"
            for a, field in enumerate(self.fields)
        ]

        return "\n".join([header, size, fields_header, *fields])

    def extract(self, grad=True, sym=False, add_identity=True, out=None):
        """Generalized extraction method which evaluates either the gradient or the
        field values at the integration points of all cells in the region. Optionally,
        the symmetric part of the gradient is evaluated and/or the identity matrix is
        added to the gradient.

        Arguments
        ---------
        grad : bool or list of bool, optional
            Flag(s) for gradient evaluation(s). A boolean value is appplied on the first
            field only and all other fields are extracted with ``grad=False``. To
            enable or disable gradients per-field, use a list of boolean values instead
            (default is True).
        sym : bool, optional
            Flag for symmetric part if the gradient is evaluated (default is False).
        add_identity : bool, optional
            Flag for the addition of the identity matrix if the gradient is evaluated
            (default is True).
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).

        Returns
        -------
        tuple of ndarray
            (Symmetric) gradient or interpolated field values evaluated at
            the integration points of each cell in the region.
        """

        if isinstance(grad, bool):
            grad = (grad,)

        if out is None:
            out = [None] * len(self.fields)

        grads = np.pad(grad, (0, len(self.fields) - 1))
        return tuple(
            f.extract(g, sym, add_identity=add_identity, out=res)
            for g, f, res in zip(grads, self.fields, out)
        )

    def values(self):
        "Return the field values."
        return tuple(f.values for f in self.fields)

    def copy(self):
        "Return a copy of the field."
        return deepcopy(self)

    def link(self, other_field):
        "Link value array of other field."
        for field, newfield in zip(self.fields, other_field.fields):
            field.values = newfield.values

    def view(self, point_data=None, cell_data=None, cell_type=None, project=None):
        """View the field with optional given dicts of point- and cell-data items.

        Parameters
        ----------
        point_data : dict or None, optional
            Additional point-data dict (default is None).
        cell_data : dict or None, optional
            Additional cell-data dict (default is None).
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).
        project : callable or None, optional
            Project internal cell-data at quadrature-points to mesh-points (default is
            None).

        Returns
        -------
        felupe.ViewField
            A object which provides visualization methods for
            :class:`felupe.FieldContainer`.

        See Also
        --------
        felupe.ViewField : Visualization methods for :class:`felupe.FieldContainer`.
        felupe.project: Project given values at quadrature-points to mesh-points.
        felupe.topoints: Shift given values at quadrature-points to mesh-points.
        """

        return ViewField(
            self,
            point_data=point_data,
            cell_data=cell_data,
            cell_type=cell_type,
            project=project,
        )

    def plot(self, *args, project=None, **kwargs):
        """Plot the first field of the container.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        felupe.project: Project given values at quadrature-points to mesh-points.
        felupe.topoints: Shift given values at quadrature-points to mesh-points.
        """
        return self.view(project=project).plot(*args, **kwargs)

    def screenshot(
        self,
        *args,
        filename="field.png",
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the first field of the container.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

    def imshow(self, *args, ax=None, dpi=None, **kwargs):
        """Take a screenshot of the first field of the container, show the image data in
        a figure and return the ax.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(dpi=dpi)

        ax.imshow(self.screenshot(*args, filename=None, **kwargs))
        ax.set_axis_off()

        return ax

    def __add__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field += dfield

        return fields

    def __sub__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field -= dfield

        return fields

    def __mul__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field *= dfield

        return fields

    def __truediv__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field /= dfield

        return fields

    def __iadd__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field += dfield
        return self

    def __isub__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field -= dfield
        return self

    def __imul__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field *= dfield
        return self

    def __itruediv__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field /= dfield
        return self

    def __getitem__(self, idx):
        "Slice-based access to underlying fields."

        return self.fields[idx]

    def __len__(self):
        "Number of fields inside the container."

        return len(self.fields)

    def __and__(self, field):
        fields = [field]
        if isinstance(field, FieldContainer):
            fields = field.fields
        elif field is None:
            fields = []

        return FieldContainer([*self.fields, *fields])
