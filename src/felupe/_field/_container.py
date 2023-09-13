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


class FieldContainer:
    """A container for fields based on a list or tuple of :class:`Field`
    instances."""

    def __init__(self, fields):
        self.fields = fields
        self.region = fields[0].region

        # get sizes of fields and calculate offsets
        self.fieldsizes = [f.indices.dof.size for f in self.fields]
        self.offsets = np.cumsum(self.fieldsizes)[:-1]

    def __repr__(self):
        header = "<felupe FieldContainer object>"
        size = f"  Number of fields: {len(self.fields)}"
        fields_header = "  Dimension of fields:"
        fields = [
            f"    {type(field).__name__}: {field.dim}"
            for a, field in enumerate(self.fields)
        ]

        return "\n".join([header, size, fields_header, *fields])

    def extract(self, grad=True, sym=False, add_identity=True):
        """Generalized extraction method which evaluates either the gradient
        or the field values at the integration points of all cells
        in the region. Optionally, the symmetric part of the gradient is
        evaluated and/or the identity matrix is added to the gradient.

        Arguments
        ---------
        grad : bool, optional (default is True)
            Flag for gradient evaluation.
        sym : bool, optional (default is False)
            Flag for symmetric part if the gradient is evaluated.
        add_identity : bool, optional (default is True)
            Flag for the addition of the identity matrix
            if the gradient is evaluated.

        Returns
        -------
        array
            (Symmetric) gradient or interpolated field values evaluated at
            the integration points of each cell in the region.
        """

        if isinstance(grad, bool):
            grad = (grad,)

        grads = np.pad(grad, (0, len(self.fields) - 1))
        return tuple(
            f.extract(g, sym, add_identity) for g, f in zip(grads, self.fields)
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

    def view(self, point_data=None, cell_data=None, cell_type=None):
        """View the field with optional given dicts of point- and cell-data items.

        Parameters
        ----------
        point_data : dict or None, optional
            Additional point-data dict (default is None).
        cell_data : dict or None, optional
            Additional cell-data dict (default is None).
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).

        Returns
        -------
        felupe.ViewField
            A object which provides visualization methods for
            :class:`felupe.FieldContainer`.

        See Also
        --------
        felupe.ViewField : Visualization methods for :class:`felupe.FieldContainer`.
        """

        return ViewField(
            self, point_data=point_data, cell_data=cell_data, cell_type=cell_type
        )

    def plot(self, *args, **kwargs):
        """Plot the first field of the container.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        """
        return self.view().plot(*args, **kwargs)

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
