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
        classical partial pivoting). Default is 0.0, i.e. no pivoting. A solution with
        a backward error above the tolerance is obtained again by a second
        factorization with the pivot threshold ``fallback_pivot_thresh``.
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
    refine : bool, optional
        A flag to improve the solution by one step of iterative refinement, which
        re-uses the factors of the matrix. Default is True.
    tol : float or None, optional
        The tolerance of the normwise backward error of the solution. If None, the
        square root of the machine epsilon of the lowest precision of the matrix and
        the right-hand side is used, e.g. about 1.5e-8 for double precision. Default is
        None.
    fallback_pivot_thresh : float or None, optional
        The pivot threshold of a second factorization of the matrix, which is used if
        the backward error of the solution exceeds the tolerance or if the first
        factorization fails. If None, the matrix is factorized only once and the
        backward error is not evaluated. Default is 0.1.

    Notes
    -----
    The default options are chosen for (structurally) symmetric matrices of the finite
    element method, also for matrices with non-symmetric values, e.g. of follower
    loads or of frictional contacts. The structure of the assembled matrices is
    symmetric by construction, i.e. no check of the symmetry is required. Compared to
    :func:`scipy.sparse.linalg.spsolve`, the factorization is faster by a factor of
    about two for meshes with hexahedrons and by a factor of about 1.5 for meshes with
    quads. The fill-in of the factors, i.e. the required memory, is lower.

    By default, the diagonal entries are taken as pivots (``diag_pivot_thresh=0.0``),
    i.e. the fill-reducing ordering of the columns is kept. Neither the large
    off-diagonal entries of penalty-based items, e.g. of a
    :class:`~felupe.MultiPointConstraint` or of a :class:`~felupe.ContactRigidPlane`,
    nor the zero diagonal entries of saddle point problems, e.g. of the pressure field
    of :class:`~felupe.ThreeFieldVariation`, lead to pivoting, which would increase
    the fill-in and the time of the factorization by orders of magnitude.

    Without pivoting, the factorization is not guaranteed to be stable. Hence, the
    normwise backward error of the solution is evaluated, see Eq.
    :eq:`superlu-backward-error`, which requires one sparse matrix-vector product and
    which does not depend on the condition number of the matrix.

    ..  math::
        :label: superlu-backward-error

        \eta = \frac{
            \| \boldsymbol{b} - \boldsymbol{A}\ \boldsymbol{x} \|_\infty
        }{
            \| \boldsymbol{A} \|_\infty\ \| \boldsymbol{x} \|_\infty
            + \| \boldsymbol{b} \|_\infty
        }

    If the backward error exceeds the tolerance, or if the first factorization fails,
    the matrix is factorized again with the pivot threshold ``fallback_pivot_thresh``.
    Before the evaluation of the backward error, the solution is improved by one step
    of iterative refinement (``refine=True``), see Eq. :eq:`superlu-refinement`. It
    re-uses the factors of the matrix, i.e. it requires only a forward and a backward
    substitution.

    ..  math::
        :label: superlu-refinement

        \boldsymbol{x} \leftarrow \boldsymbol{x} + (\boldsymbol{L}\boldsymbol{U})^{-1}
            \left( \boldsymbol{b} - \boldsymbol{A}\ \boldsymbol{x} \right)

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
        SuperLUSolver(permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
        symmetric_mode=True, relax=20, panel_size=20, refine=True, tol=None,
        fallback_pivot_thresh=0.1)

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
        diag_pivot_thresh=0.0,
        symmetric_mode=True,
        relax=20,
        panel_size=20,
        options=None,
        refine=True,
        tol=None,
        fallback_pivot_thresh=0.1,
    ):
        self.permc_spec = permc_spec
        self.diag_pivot_thresh = diag_pivot_thresh
        self.symmetric_mode = symmetric_mode
        self.relax = relax
        self.panel_size = panel_size
        self.options = options
        self.refine = refine
        self.tol = tol
        self.fallback_pivot_thresh = fallback_pivot_thresh

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

        args += [
            f"refine={self.refine!r}",
            f"tol={self.tol!r}",
            f"fallback_pivot_thresh={self.fallback_pivot_thresh!r}",
        ]

        return f"SuperLUSolver({', '.join(args)})"

    def factorize(self, A, diag_pivot_thresh=None):
        """Return the LU decomposition of a sparse matrix, see
        :func:`scipy.sparse.linalg.splu`.

        Parameters
        ----------
        A : scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse square matrix (converted to CSC format).
        diag_pivot_thresh : float or None, optional
            The threshold for the partial pivoting. If None, the pivot threshold of the
            solver is used. Default is None.

        Returns
        -------
        scipy.sparse.linalg.SuperLU
            The LU decomposition with the method ``solve(b)``.
        """

        if diag_pivot_thresh is None:
            diag_pivot_thresh = self.diag_pivot_thresh

        options = {"SymmetricMode": self.symmetric_mode}

        if self.options is not None:
            options.update(self.options)

        return splu(
            A.tocsc(),
            permc_spec=self.permc_spec,
            diag_pivot_thresh=diag_pivot_thresh,
            relax=self.relax,
            panel_size=self.panel_size,
            options=options,
        )

    def backward_error(self, A, x, b, r=None):
        r"""Return the normwise backward error of a solution of the linear equation
        system ``A x = b``, i.e. the largest value of all right-hand sides.

        Parameters
        ----------
        A : scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse square matrix of the linear equation system.
        x : ndarray
            The solution(s) with the shape ``(n,)`` or ``(n, m)``.
        b : ndarray
            The right-hand side(s) with the same shape as the solution(s).
        r : ndarray or None, optional
            The residual(s) ``b - A x``. If None, the residuals are evaluated. Default
            is None.

        Returns
        -------
        float
            The normwise backward error, see Eq. :eq:`superlu-backward-error`. A NaN
            value of the solution leads to a NaN value of the backward error. For a
            right-hand side of zeros, the backward error of the zero solution is zero.
        """

        if r is None:
            r = b - A @ x

        # the maximum absolute row sum of the matrix
        A = A.tocsc()
        n = A.shape[0]
        rowsums = np.bincount(A.indices, weights=np.abs(A.data), minlength=n)
        norm_A = rowsums.max(initial=0.0)

        # one value for each right-hand side
        norm_r = np.abs(r.reshape(n, -1)).max(axis=0)
        norm_x = np.abs(x.reshape(n, -1)).max(axis=0)
        norm_b = np.abs(b.reshape(n, -1)).max(axis=0)

        denominator = norm_A * norm_x + norm_b

        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.where(denominator > 0, norm_r / denominator, norm_r)

        return errors.max()

    def _tolerance(self, A, b):
        "Return the tolerance of the backward error."

        if self.tol is not None:
            return self.tol

        # the lowest precision of the matrix and the right-hand side
        dtypes = [d for d in [A.dtype, b.dtype] if np.issubdtype(d, np.inexact)]
        eps = max([np.finfo(d).eps for d in dtypes], default=np.finfo(float).eps)

        return np.sqrt(eps)

    def _solve_sparse(self, A, b):
        "Solve a linear equation system with a sparse matrix."

        A = A.tocsc()

        thresholds = [self.diag_pivot_thresh]

        if self.fallback_pivot_thresh is not None:
            thresholds.append(self.fallback_pivot_thresh)

        for attempt, diag_pivot_thresh in enumerate(thresholds):
            last_attempt = attempt == len(thresholds) - 1

            try:
                lu = self.factorize(A, diag_pivot_thresh=diag_pivot_thresh)

            except RuntimeError as error:
                if "singular" not in str(error):
                    raise

                continue  # try the next factorization (with pivoting)

            x = lu.solve(b)

            if self.refine:
                x = x + lu.solve(b - A @ x)

            # the backward error is only required to decide about a next attempt
            if last_attempt or self.backward_error(A, x, b) <= self._tolerance(A, b):
                return x

        # like scipy.sparse.linalg.spsolve: warn and return NaN-values
        warn("Matrix is exactly singular", MatrixRankWarning, stacklevel=3)
        return np.full(b.shape, np.nan, dtype=np.result_type(A.dtype, b.dtype))

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
            x = self._solve_sparse(A, b)

        # a single right-hand side returns a 1d-array, like scipy.sparse.linalg.spsolve
        if b.ndim == 2 and b.shape[1] == 1:
            x = x.ravel()

        return x
