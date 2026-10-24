# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from ..dof import partition
from ..tools._newton import embed


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

    The multiplier of an item, e.g. ``SolidBody(..., multiplier=2.0)``, scales both its
    stiffness and its mass matrix, i.e. the contribution of the item.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

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
        self._x0 = None

    def evaluate(self, x0=None, solver=eigsh, parallel=False, **kwargs):
        """Evaluate the eigenvalues and the eigenvectors of the free vibration.

        Parameters
        ----------
        x0 : FieldContainer or None, optional
            The field container with all unknowns (default is None). If None, the
            top-level field container ``x0`` of the field container of the first item
            is used (e.g. created by :func:`~felupe.field.merge`), otherwise the field
            container of the first item.
        solver : callable, optional
            A solver for the generalized eigenvalue problem with the signature
            ``eigenvalues, eigenvectors = solver(A, M, **kwargs)`` (default is
            :func:`scipy.sparse.linalg.eigsh`).
        parallel : bool, optional
            Flag to assemble the stiffness matrices in parallel (default is False).
        **kwargs : dict, optional
            Optional keyword-arguments for the solver. The default shift is
            ``sigma=0``.

        Returns
        -------
        FreeVibration
            The free-vibration object with the ``eigenvalues`` and the
            ``eigenvectors``.
        """

        if x0 is not None:
            x = x0
        else:
            field = self.items[0].field
            x = getattr(field, "x0", field)

        # the field container is used by default to extract the eigenvectors
        self._x0 = x

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

            # the multiplier scales the contribution of the item (stiffness and mass)
            if item.assemble.multiplier is not None:
                K *= item.assemble.multiplier
                M *= item.assemble.multiplier

            # embed the matrices into the global degrees of freedom
            K = embed(K, shape, item.field, x)
            M = embed(M, shape, item.field, x)

            # add matrices
            stiffness += K
            mass += M

        dof0, self.dof1 = partition(x, self.boundaries)

        K = stiffness[self.dof1][:, self.dof1]
        M = mass[self.dof1][:, self.dof1]

        kwargs.setdefault("sigma", 0)
        self.eigenvalues, self.eigenvectors = solver(A=K, M=M, **kwargs)

        return self

    def extract(self, n=0, x0=None, inplace=True):
        """Extract an eigenvector and its frequency.

        Parameters
        ----------
        n : int, optional
            The index of the eigenvalue and the eigenvector (default is 0).
        x0 : FieldContainer or None, optional
            The field container in which the eigenvector is stored (default is None).
            If None, the field container of :meth:`evaluate` is used.
        inplace : bool, optional
            Flag to store the eigenvector in the given field container (default is
            True). If False, a copy of the field container is used.

        Returns
        -------
        FieldContainer
            The field container with the eigenvector.
        float
            The frequency of the eigenvector.
        """

        if x0 is not None:
            field = x0
        else:
            field = self._x0

        if not inplace:
            field = field.copy()

        values = np.zeros(sum(field.fieldsizes))
        values[self.dof1] = self.eigenvectors[:, n]

        frequency = np.sqrt(self.eigenvalues[n]) / (2 * np.pi)

        [f.fill(0) for f in field]
        field += values

        return field, frequency
