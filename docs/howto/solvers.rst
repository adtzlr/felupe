Use external solvers
--------------------

FElupe uses SuperLU as direct sparse solver by default because it is shipped with SciPy
(and SciPy is already a dependency of FElupe). The default solver
:class:`~felupe.solve.SuperLUSolver` uses options of :func:`~scipy.sparse.linalg.splu`
which are tuned for the (structurally) symmetric matrices of the finite element method,
also with non-symmetric values, e.g. of frictional contacts. While it is definitely a
good choice for small to mid-sized problems, faster alternatives are easy to install and
use. This section demonstrates several possibilities, e.g. a fast direct solver from
`PyPardiso <https://github.com/haasad/PyPardisoProject>`_ (``pip install pypardiso``)
and the :func:`~scipy.sparse.linalg.minres` iterative solver from ``SciPy``. Custom
solvers may be passed to the evaluation of a :class:`~felupe.Job`.

..  code-block:: python

    import felupe as fem

    solver = fem.solve.SuperLUSolver()  # (default)

    job = fem.Job(steps)
    job.evaluate(solver=solver) # function `x = solver(A, b)`

Solvers from SciPy Sparse:

.. tab:: SciPy Sparse (direct)

   .. code-block:: python

      import felupe as fem

      # the default solver, based on scipy.sparse.linalg.splu
      solver = fem.solve.SuperLUSolver()

.. tab:: SciPy Sparse (direct, without tuned options)

   ..  code-block:: python

       # the default solver of FElupe before v12.0.0
       from scipy.sparse.linalg import spsolve

.. tab:: SciPy Sparse (iterative)

   ..  note::

       ``minres`` may be replaced by another iterative method, e.g. ``cg``.

   ..  code-block:: python
        
       import scipy.sparse.linalg as spla

       def solver(A, b):
           "Wrapper function for iterative solvers from scipy.sparse.linalg."
           
           return spla.minres(A, b, rtol=1e-4)[0]

Solvers from external packages:

.. tab:: PyPardiso (direct)

   Ensure to have `PyPardiso <https://github.com/haasad/PyPardisoProject>`_  installed.

   ..  code-block:: bash
      
       pip install pypardiso

   ..  code-block:: python
      
       from pypardiso import spsolve

.. tab:: PyPardiso (direct, optionally symmetric)

   Ensure to have `PyPardiso <https://github.com/haasad/PyPardisoProject>`_  installed.

   ..  code-block:: bash
      
       pip install pypardiso

   ..  code-block:: python
      
       from pypardiso import PyPardisoSolver
       from scipy.sparse import triu

       def spsolve(A, b):
           # mtype = 1: real and structurally symmetric, supernode pivoting
           # mtype = 2: real and symmetric positive definite
           # mtype =-2: real and symmetric indefinite, 
           #             diagonal or Bunch-Kaufman pivoting
           # mtype = 6: complex and symmetric
           # mtype = 11: default
           return PyPardisoSolver(mtype=-2).solve(triu(A, format="csr"), b).squeeze()
