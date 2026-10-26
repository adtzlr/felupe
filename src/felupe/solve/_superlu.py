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
    diag_perturbation : float or None, optional
        The relative perturbation of the zero diagonal entries of the matrix, e.g. of
        the pressure field of a saddle point problem, before the factorization (static
        pivoting), see Eq. :eq:`superlu-perturbation`. If None, the square root of the
        machine epsilon of the precision of the matrix is used, e.g. about 1.5e-8 for
        double precision. With 0, the zero diagonal entries are not perturbed. The
        matrix is only perturbed with ``refine=True``. Default is None.
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
        A flag to improve the solution by iterative refinement, which re-uses the
        factors of the matrix. Default is True.
    tol : float or None, optional
        The tolerance of the normwise backward error of the solution. If None, the
        square root of the machine epsilon of the lowest precision of the matrix and
        the right-hand side is used, e.g. about 1.5e-8 for double precision. Default is
        None.
    fallback_pivot_thresh : float or None, optional
        The pivot threshold of a second factorization of the matrix, which is used if
        the backward error of the solution exceeds the tolerance or if the first
        factorization fails. If None, the matrix is factorized only once and the
        backward error is not evaluated (except for the iterative refinement of a
        perturbed matrix). Default is 0.1.

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
    i.e. the fill-reducing ordering of the columns is kept. The large off-diagonal
    entries of penalty-based items, e.g. of a :class:`~felupe.MultiPointConstraint` or
    of a :class:`~felupe.ContactRigidPlane`, do not lead to pivoting, which would
    increase the fill-in and the time of the factorization by orders of magnitude.

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

    However, SuperLU never takes a diagonal entry of zero as pivot, also not with
    ``diag_pivot_thresh=0.0``. Saddle point problems have zero diagonal entries, e.g.
    the pressure fields of :class:`~felupe.ThreeFieldVariation` and
    :class:`~felupe.NearlyIncompressible` (on :class:`~felupe.FieldsMixed`) or of a
    mixed weak form by :func:`~felupe.Form`. A zero diagonal entry, which is eliminated
    before it is filled by the elimination of one of its neighbours, leads to an
    exchange of rows and the fill-reducing ordering of the columns is lost. Hence, the
    zero diagonal entries :math:`A_{ii} = 0` are perturbed before the factorization,
    see Eq. :eq:`superlu-perturbation` (static pivoting [1]_, [2]_), by the relative
    perturbation :math:`\epsilon` (``diag_perturbation``) of an estimate of their
    pivots, i.e. of the diagonal entries of the Schur complement, where the inverse of
    the other block of the matrix is approximated by the inverse of its diagonal.

    ..  math::
        :label: superlu-perturbation

        \tilde{A}_{ii} = -\epsilon \sum_{j,\ A_{jj} \ne 0}
            \frac{A_{ji}^2}{|A_{jj}|} \qquad \text{if} \qquad A_{ii} = 0

    The perturbation is invariant to the scaling of the unknowns, e.g. by the units of
    the displacements and the pressure. Its negative sign is the sign of the pivots
    of a symmetric saddle point problem with a positive definite block, i.e. the
    perturbed matrix corresponds to a penalty formulation of the constraints. A zero
    diagonal entry without non-zero entries in the rows with non-zero diagonal entries
    of its column, e.g. of an empty row and column, is not perturbed. The perturbed
    matrix is factorized and the solution is improved by iterative refinement with the
    original matrix, see Eq. :eq:`superlu-refinement`, until the norm of the
    correction is below the tolerance times the norm of the solution (maximum norms,
    at most ten steps). The iterative refinement is stopped if the norm of the
    correction is not reduced by at least a factor of two (the norm of the first
    correction is compared to the norm of the solution of the perturbed matrix). If
    the iterative refinement does not converge or if the backward error of the refined
    solution exceeds the tolerance, the (unperturbed) matrix is factorized again with
    the pivot threshold ``fallback_pivot_thresh``.

    The solution has the same shape as :func:`scipy.sparse.linalg.spsolve`, i.e. a
    one-dimensional array for a right-hand side with the shape ``(n,)`` or ``(n, 1)``.
    The right-hand side may also be a sparse matrix (converted to a dense array) and
    multiple right-hand sides with the shape ``(n, m)`` are supported. A dense matrix is
    solved by :func:`numpy.linalg.solve`.

    Like :func:`scipy.sparse.linalg.spsolve`, an exactly singular matrix emits a
    :class:`~scipy.sparse.linalg.MatrixRankWarning` and returns NaN values, which
    raises a :class:`~felupe.NewtonConvergenceError` in :func:`~felupe.newtonraphson`.
    For an exactly singular matrix with perturbed zero diagonal entries, the iterative
    refinement does not converge (unless the right-hand side is in the range of the
    matrix), i.e. it is detected by the factorization of the fallback.

    References
    ----------
    .. [1] X. S. Li and J. W. Demmel, "Making sparse Gaussian elimination scalable by
       static pivoting", in Proceedings of the ACM/IEEE Conference on Supercomputing
       (SC'98), 1998, doi: `10.1109/SC.1998.10030
       <https://doi.org/10.1109/SC.1998.10030>`_.
    .. [2] O. Schenk and K. Gärtner, "On fast factorization pivoting methods for sparse
       symmetric indefinite systems", Electronic Transactions on Numerical Analysis,
       vol. 23, pp. 158–179, 2006.

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
        diag_perturbation=None, symmetric_mode=True, relax=20, panel_size=20,
        refine=True, tol=None, fallback_pivot_thresh=0.1)

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
        diag_perturbation=None,
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
        self.diag_perturbation = diag_perturbation
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
            f"diag_perturbation={self.diag_perturbation!r}",
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

    def perturb(self, A):
        r"""Return the matrix with perturbed zero diagonal entries, see Eq.
        :eq:`superlu-perturbation`.

        Parameters
        ----------
        A : scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse square matrix (converted to CSC format).

        Returns
        -------
        scipy.sparse.sparray or scipy.sparse.spmatrix
            The matrix (in CSC format) with the perturbed zero diagonal entries. If no
            diagonal entry is perturbed, the matrix in CSC format is returned (without
            a copy for a matrix in CSC format).
        """

        A = A.tocsc()
        diagonal = A.diagonal()
        zeros = np.flatnonzero(diagonal == 0)

        if len(zeros) == 0:
            return A

        # the columns of the zero diagonal entries
        columns = A[:, zeros]
        dtype = np.result_type(A.dtype, np.float32)
        absdiag = np.abs(diagonal[columns.indices])

        # the estimated (absolute) pivots of the zero diagonal entries
        weights = np.divide(
            np.abs(columns.data) ** 2,
            absdiag,
            out=np.zeros(len(absdiag)),
            where=absdiag > 0,
        )
        cols = np.repeat(np.arange(len(zeros)), np.diff(columns.indptr))
        pivots = np.bincount(cols, weights=weights, minlength=len(zeros))

        mask = pivots > 0

        if not np.any(mask):
            return A

        rows = zeros[mask]
        values = (-self._perturbation(A) * pivots[mask]).astype(dtype)

        return A + type(A)((values, (rows, rows)), shape=A.shape)

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

    def _perturbation(self, A):
        "Return the relative perturbation of the zero diagonal entries."

        if self.diag_perturbation is not None:
            return self.diag_perturbation

        return np.sqrt(np.finfo(np.result_type(A.dtype, np.float32)).eps)

    @staticmethod
    def _change(dx, x):
        "Return the largest relative change of the solution(s) (maximum norms)."

        n = len(x)
        norm_dx = np.abs(dx.reshape(n, -1)).max(axis=0)
        norm_x = np.abs(x.reshape(n, -1)).max(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            changes = np.where(norm_x > 0, norm_dx / norm_x, norm_dx)

        return changes.max()

    def _refine_perturbed(self, lu, A, x, b):
        """Improve the solution of a perturbed matrix by iterative refinement with the
        original matrix. Return the solution and a flag if the iterative refinement has
        converged with a backward error within the tolerance."""

        tol = self._tolerance(A, b)
        r = b - A @ x
        change = 1.0  # the relative change of the solution of the perturbed matrix

        for _ in range(10):
            dx = lu.solve(r)
            x = x + dx
            r = b - A @ x

            previous, change = change, self._change(dx, x)

            # converged, stagnated or diverged (also for NaN-values)
            if change <= tol or not change <= previous / 2:
                break

        converged = change <= tol

        return x, converged and self.backward_error(A, x, b, r=r) <= tol

    def _solve_sparse(self, A, b):
        "Solve a linear equation system with a sparse matrix."

        A = A.tocsc()

        # the first attempt with the perturbed matrix (without pivoting by default)
        perturb = self.refine and self.diag_perturbation != 0
        attempts = [(self.diag_pivot_thresh, perturb)]

        if self.fallback_pivot_thresh is not None:
            attempts.append((self.fallback_pivot_thresh, False))

        for attempt, (diag_pivot_thresh, perturb) in enumerate(attempts):
            last_attempt = attempt == len(attempts) - 1

            # the perturbed matrix is a new matrix, or the matrix itself
            M = self.perturb(A) if perturb else A

            try:
                lu = self.factorize(M, diag_pivot_thresh=diag_pivot_thresh)

            except RuntimeError as error:
                if "singular" not in str(error):
                    raise

                continue  # try the next factorization (with pivoting)

            x = lu.solve(b)

            if M is not A:
                x, accepted = self._refine_perturbed(lu, A, x, b)

                if accepted or last_attempt:
                    return x

                continue  # try the next factorization (with pivoting)

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
