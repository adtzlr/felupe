Use external solvers
--------------------

FElupe uses SuperLU as direct sparse solver by default because it is shipped with SciPy
(and SciPy is already a dependency of FElupe). While it is definitely a good choice for
small to mid-sized problems, faster alternatives are easy to install and use. This
section demonstrates several possibilities, e.g. a fast direct solver from
`PyPardiso <https://github.com/haasad/PyPardisoProject>`_ (``pip install pypardiso``)
and the :func:`~scipy.sparse.linalg.minres` iterative solver from ``SciPy``. Custom
solvers may be passed to the evaluation of a :class:`~felupe.Job`.

..  code-block:: python

    import felupe as fem
    
    from scipy.sparse.linalg import spsolve  # (default)

    job = fem.Job(steps)
    job.evaluate(solver=spsolve) # function `x = spsolve(A, b)`

Solvers from SciPy Sparse:

.. tab:: SciPy Sparse (direct)

   .. code-block:: python
      
      # the default solver
      from scipy.sparse.linalg import spsolve

.. tab:: SciPy Sparse (direct, symmetric)

   ..  code-block:: python

       from scipy.sparse.linalg import splu

       def solver(A, b):
       lu = splu(
           A.tocsc(),
           permc_spec="MMD_AT_PLUS_A",
           diag_pivot_thresh=0.0,
           options=dict(SymmetricMode=True),
       )
       return lu.solve(b)

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
