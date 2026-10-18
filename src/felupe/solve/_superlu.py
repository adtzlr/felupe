# SPDX-License-Identifier: GPL-3.0-or-later

from warnings import warn

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import MatrixRankWarning, splu


class SuperLUSolver:
    r"""A sparse direct solver by :func:`scipy.sparse.linalg.splu` (SuperLU), tuned for
    the linear equation systems of the finite element method. An instance is a solver
    function ``x = solver(A, b)``, e.g. for :meth:`Job.evaluate(solver=solver)
    <felupe.Job.evaluate>`, :func:`~felupe.newtonraphson`, :func:`~felupe.project` and
    :func:`~felupe.solve.solve`.

    Parameters
    ----------
    permc_spec : str, optional
        The fill-reducing ordering of the columns. Default is ``"MMD_AT_PLUS_A"``, the
        minimum degree ordering of the structure of :math:`\boldsymbol{A}^T +
        \boldsymbol{A}`. Others are ``"COLAMD"``, ``"MMD_ATA"`` and ``"NATURAL"``.
    diag_pivot_thresh : float, optional
        The threshold for the partial pivoting in the range ``[0, 1]``. The diagonal
        entry is taken as pivot if its absolute value is at least the threshold times
        the largest absolute value of its column (0 takes the diagonal entry, 1 is the
        classical partial pivoting). Default is 0.1.
    symmetric_mode : bool, optional
        A flag to use the symmetric mode of SuperLU, i.e. diagonal pivots are preferred
        and the elimination tree is based on the structure of :math:`\boldsymbol{A}^T +
        \boldsymbol{A}`. It requires a symmetric structure of the matrix, the values
        may be non-symmetric. Default is True.
    relax : int, optional
        The relaxation parameter of the supernodes. Default is 20.
    panel_size : int, optional
        The size of the panels. Default is 20.
    options : dict or None, optional
        Additional options of SuperLU, see :func:`scipy.sparse.linalg.splu`. Default is
        None.

    Notes
    -----
    The default options are chosen for (structurally) symmetric matrices of the finite
    element method, also for matrices with non-symmetric values, e.g. of follower
    loads or of frictional contacts. The structure of the assembled matrices is
    symmetric by construction, i.e. no check of the symmetry is required. Compared to
    :func:`scipy.sparse.linalg.spsolve`, the factorization is faster by a factor of
    about two for meshes with hexahedrons and by a factor of about 1.5 for meshes with
    quads. The fill-in of the factors, i.e. the required memory, is lower. A pivot
    threshold of ``diag_pivot_thresh=0.1`` avoids pivoting by the large off-diagonal
    entries of penalty-based items, e.g. of a :class:`~felupe.MultiPointConstraint` or
    of a :class:`~felupe.ContactRigidPlane`, which would increase the fill-in.

    The solution has the same shape as :func:`scipy.sparse.linalg.spsolve`, i.e. a
    one-dimensional array for a right-hand side with the shape ``(n,)`` or ``(n, 1)``.
    The right-hand side may also be a sparse matrix (converted to a dense array) and
    multiple right-hand sides with the shape ``(n, m)`` are supported. A dense matrix is
    solved by :func:`numpy.linalg.solve`.

    Like :func:`scipy.sparse.linalg.spsolve`, an exactly singular matrix emits a
    :class:`~scipy.sparse.linalg.MatrixRankWarning` and returns NaN values, which
    raises a :class:`~felupe.NewtonConvergenceError` in :func:`~felupe.newtonraphson`.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=4)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> boundaries = fem.dof.uniaxial(field, clamped=True)
        >>> solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)
        >>> step = fem.Step(
        ...     items=[solid], ramp={boundaries["move"]: [0.1]}, boundaries=boundaries
        ... )
        >>>
        >>> solver = fem.solve.SuperLUSolver()  # the default solver
        >>> job = fem.Job(steps=[step]).evaluate(solver=solver)
        >>> solver
        SuperLUSolver(permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.1,
        symmetric_mode=True, relax=20, panel_size=20)

    The previous default solver of FElupe is :func:`scipy.sparse.linalg.spsolve`.

    >>> from scipy.sparse.linalg import spsolve
    >>>
    >>> job = fem.Job(steps=[step]).evaluate(solver=spsolve)

    See Also
    --------
    scipy.sparse.linalg.splu : Compute the LU decomposition of a sparse, square matrix.
    scipy.sparse.linalg.spsolve : Solve the sparse linear system Ax=b.
    """

    def __init__(
        self,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.1,
        symmetric_mode=True,
        relax=20,
        panel_size=20,
        options=None,
    ):
        self.permc_spec = permc_spec
        self.diag_pivot_thresh = diag_pivot_thresh
        self.symmetric_mode = symmetric_mode
        self.relax = relax
        self.panel_size = panel_size
        self.options = options

    def __repr__(self):
        args = [
            f"permc_spec={self.permc_spec!r}",
            f"diag_pivot_thresh={self.diag_pivot_thresh!r}",
            f"symmetric_mode={self.symmetric_mode!r}",
            f"relax={self.relax!r}",
            f"panel_size={self.panel_size!r}",
        ]

        if self.options is not None:
            args.append(f"options={self.options!r}")

        return f"SuperLUSolver({', '.join(args)})"

    def factorize(self, A):
        """Return the LU decomposition of a sparse matrix, see
        :func:`scipy.sparse.linalg.splu`.

        Parameters
        ----------
        A : scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse square matrix (converted to CSC format).

        Returns
        -------
        scipy.sparse.linalg.SuperLU
            The LU decomposition with the method ``solve(b)``.
        """

        options = {"SymmetricMode": self.symmetric_mode}

        if self.options is not None:
            options.update(self.options)

        return splu(
            A.tocsc(),
            permc_spec=self.permc_spec,
            diag_pivot_thresh=self.diag_pivot_thresh,
            relax=self.relax,
            panel_size=self.panel_size,
            options=options,
        )

    def __call__(self, A, b):
        """Solve the linear equation system ``A x = b``.

        Parameters
        ----------
        A : scipy.sparse.sparray or scipy.sparse.spmatrix or ndarray
            The square matrix of the linear equation system.
        b : ndarray or scipy.sparse.sparray or scipy.sparse.spmatrix
            The right-hand side(s) with the shape ``(n,)``, ``(n, 1)`` or ``(n, m)``.

        Returns
        -------
        ndarray
            The solution with the shape ``(n,)`` for a single right-hand side or
            ``(n, m)`` for multiple right-hand sides.
        """

        if issparse(b):
            b = b.toarray()

        b = np.asarray(b)

        if not issparse(A):
            x = np.linalg.solve(A, b)

        else:
            try:
                x = self.factorize(A).solve(b)

            except RuntimeError as error:
                if "singular" not in str(error):
                    raise

                # like scipy.sparse.linalg.spsolve: warn and return NaN-values
                warn("Matrix is exactly singular", MatrixRankWarning, stacklevel=2)
                x = np.full(b.shape, np.nan, dtype=np.result_type(A.dtype, b.dtype))

        # a single right-hand side returns a 1d-array, like scipy.sparse.linalg.spsolve
        if b.ndim == 2 and b.shape[1] == 1:
            x = x.ravel()

        return x
