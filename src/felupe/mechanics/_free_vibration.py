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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from ..dof import partition


class FreeVibration:
    """A Free-Vibration Step.

    Parameters
    ----------
    items : list of SolidBody or SolidBodyNearlyIncompressible
        A list of items with methods for the assembly of sparse stiffness and mass
        matrices.
    boundaries : dict of Boundary, optional
        A dict with :class:`~felupe.Boundary` conditions (default is None).

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>> import numpy as np
        >>> from scipy.sparse.linalg import eigsh
        >>>
        >>> mesh = fem.Rectangle(b=(5, 1), n=(50, 10))
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        >>>
        >>> boundaries = dict(left=fem.Boundary(field[0], fx=0))
        >>> dof0, dof1 = fem.dof.partition(field, boundaries)
        >>>
        >>> solid = fem.SolidBody(umat=fem.LinearElastic(E=2.5, nu=0.25), field=field)
    """

    def __init__(self, items, boundaries=None):
        self.items = items

        if boundaries is None:
            boundaries = {}

        self.boundaries = boundaries
        self.eigenvalues = None
        self.eigenvectors = None

    def evaluate(self, x0=None, k=6, solver=eigsh, tol=0, **kwargs):
        if x0 is not None:
            x = x0
        else:
            x = self.items[0].field

        # link field of items with global field
        [item.field.link(x) for item in self.items]

        # init matrix with shape from global field
        shape = (np.sum(x.fieldsizes), np.sum(x.fieldsizes))
        stiffness = csr_matrix(shape)
        mass = csr_matrix(shape)

        # assemble matrices
        for item in self.items:
            K = item.assemble.matrix(**kwargs)
            M = item.assemble.mass()

            if item.assemble.multiplier is not None:
                K *= item.assemble.multiplier

            # check and reshape matrices
            if K.shape != stiffness.shape:
                K.resize(*shape)

            if M.shape != mass.shape:
                M.resize(*shape)

            # add matrices
            stiffness += K
            mass += M

        dof0, self.dof1 = partition(x, self.boundaries)

        K = stiffness[self.dof1][:, self.dof1]
        M = mass[self.dof1][:, self.dof1]

        self.eigenvalues, self.eigenvectors = solver(A=K, k=k, M=M, sigma=0, tol=tol)

        return self

    def extract(self, n=0, x0=None, inplace=True):
        if x0 is not None:
            x = x0
        else:
            x = self.items[0].field

        if not inplace:
            x = x.copy()

        frequency = np.sqrt(self.eigenvalues[n]) / (2 * np.pi)
        x[0].values.ravel()[self.dof1] = self.eigenvectors[:, n]

        return x, frequency
