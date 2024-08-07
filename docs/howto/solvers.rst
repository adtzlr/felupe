Use external solvers
--------------------

FElupe uses SuperLU as direct sparse solver by default because it is shipped with SciPy (and SciPy is already a dependency of FElupe). While it is definitely a good choice for small to mid-sized problems, faster alternatives are easy to install and use. This section demonstrates several possibilities, e.g. a fast direct solver from `PyPardiso <https://github.com/haasad/PyPardisoProject>`_ (``pip install pypardiso``) and the ``minres`` iterative solver from ``SciPy``. Custom solvers may be passed to the evaluation of a job.

..  code-block:: python

    import felupe as fem

    job = fem.Job(steps)
    job.evaluate(solver=solver) # function `x = solver(A, b)`

Solvers from SciPy Sparse:

.. tab:: SciPy Sparse (direct)

   .. code-block:: python
      
      # the default solver
      from scipy.sparse.linalg import spsolve as solver

.. tab:: SciPy Sparse (iterative)

   ..  note::

       ``minres`` may be replaced by another iterative method.

   ..  code-block:: python
        
       import scipy.sparse.linalg as spla

       def solver(A, b):
           "Wrapper function for iterative solvers from scipy.sparse."
           
           return spla.minres(A, b)[0]

Solvers from external packages:

.. tab:: PyPardiso (direct)

   Ensure to have `PyPardiso <https://github.com/haasad/PyPardisoProject>`_  installed.

   ..  code-block:: bash
      
       pip install pypardiso

   ..  code-block:: python
      
       from pypardiso import spsolve as solver

.. tab:: PyPardiso (direct, symmetric)

   Ensure to have `PyPardiso <https://github.com/haasad/PyPardisoProject>`_  installed.

   ..  code-block:: bash
      
       pip install pypardiso

   ..  code-block:: python
      
       from pypardiso import PyPardisoSolver
       from scipy.sparse import triu

       def solver(A, b):
           # mtype = 1: real and structurally symmetric, supernode pivoting
           # mtype = 2: real and symmetric positive definite
           # mtype =-2: real and symmetric indefinite, 
           #             diagonal or Bunch-Kaufman pivoting
           # mtype = 6: complex and symmetric
           return PyPardisoSolver(mtype=-2).solve(triu(A).tocsr(), b).squeeze()
