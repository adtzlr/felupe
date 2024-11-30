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
    """A Free-Vibration Step/Job.

    Parameters
    ----------
    items : list of SolidBody or SolidBodyNearlyIncompressible
        A list of items with methods for the assembly of sparse stiffness and mass
        matrices.
    boundaries : dict of Boundary, optional
        A dict with :class:`~felupe.Boundary` conditions (default is None).

    Notes
    -----
    ..  note::

        Boundary conditions with non-zero values are not supported.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> mesh = fem.Rectangle(b=(5, 1), n=(50, 10))
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        >>>
        >>> boundaries = dict(left=fem.Boundary(field[0], fx=0))
        >>> solid = fem.SolidBody(fem.LinearElastic(E=2.5, nu=0.25), field, density=1.0)
        >>> modal = fem.FreeVibration(items=[solid], boundaries=boundaries).evaluate()
        >>>
        >>> eigenvector, frequency = modal.extract(n=4, inplace=True)
        >>> solid.plot("Stress", component=0).show()
    """

    def __init__(self, items, boundaries=None):
        self.items = items

        if boundaries is None:
            boundaries = {}

        self.boundaries = boundaries
        self.eigenvalues = None
        self.eigenvectors = None

    def evaluate(self, x0=None, solver=eigsh, parallel=False, **kwargs):
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
            K = item.assemble.matrix(parallel=parallel)
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

        sigma = kwargs.get("sigma", 0)
        self.eigenvalues, self.eigenvectors = solver(A=K, M=M, sigma=sigma, **kwargs)

        return self

    def extract(self, n=0, x0=None, inplace=True):
        if x0 is not None:
            field = x0
        else:
            field = self.items[0].field

        if not inplace:
            field = field.copy()

        values = np.zeros(sum(field.fieldsizes))
        values[self.dof1] = self.eigenvectors[:, n]

        frequency = np.sqrt(self.eigenvalues[n]) / (2 * np.pi)

        [f.fill(0) for f in field]
        field += values

        return field, frequency
