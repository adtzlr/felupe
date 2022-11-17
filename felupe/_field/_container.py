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

from copy import deepcopy

import numpy as np


class FieldContainer:
    """A container for fields based on a list or tuple of :class:`Field`
    instances."""

    def __init__(self, fields):

        self.fields = fields
        self.region = fields[0].region

        # get sizes of fields and calculate offsets
        self.fieldsizes = [f.indices.dof.size for f in self.fields]
        self.offsets = np.cumsum(self.fieldsizes)[:-1]

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
