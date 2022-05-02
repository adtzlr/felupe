Use external solvers
--------------------

FElupe uses SuperLU as direct sparse solver by default. Not because it is super fast - just because it is shipped with SciPy (and SciPy is already a dependancy of FElupe). While it is definitely a good choice for small to mid-sized problems, faster alternatives are easy to install and use. This section demonstrates several possibilities, e.g. a fast direct solver from `PyPardiso <https://github.com/haasad/PyPardisoProject>`_ (``pip install pypardiso``) and the ``minres`` iterative solver from `Krylov <https://github.com/nschloe/krylov>`_ (``pip install krylov``). Be aware to check the solution (residuals) for iterative solvers.

Solvers from SciPy Sparse:

.. tab:: SciPy Sparse (direct)

   .. code-block:: python
      
      import felupe as fe
      
      # ...
      
      system = fe.solve.partition(field, K, dof1, dof0)
      fe.solve.solve(*system)

.. tab:: SciPy Sparse (direct, symmetric)

   .. code-block:: python
      
      import felupe as fe
      import scipy.sparse.linalg as spla
      from scipy.sparse import tril
      
      # ...
      
      def solver(A, b):
          return spla.spsolve_triangular(tril(A).tocsr(), b).squeeze()
      
      system = fe.solve.partition(field, K, dof1, dof0)
      fe.solve.solve(*system)

.. tab:: SciPy Sparse (iterative)

   **Note**: ``minres`` may be replaced by another iterative method.

   ..  code-block:: python
        
       import felupe as fe
       import scipy.sparse.linalg as spla
       
       # ...
       
       def solver(A, b):
           "Wrapper function for iterative solvers from scipy.sparse."
           
           return spla.minres(A, b)[0]
       
       system = fe.solve.partition(field, K, dof1, dof0)
       fe.solve.solve(*system, solver=solver)

Solvers from external packages:

.. tab:: PyPardiso (direct)

   Ensure to have `PyPardiso <https://github.com/haasad/PyPardisoProject>`_  installed.

   ..  code-block:: bash
      
       pip install pypardiso

   ..  code-block:: python
      
       import felupe as fe
       from pypardiso import spsolve as solver
       
       # ...
       
       system = fe.solve.partition(field, K, dof1, dof0)
       fe.solve.solve(*system, solver=solver)

.. tab:: PyPardiso (direct, symmetric)

   Ensure to have `PyPardiso <https://github.com/haasad/PyPardisoProject>`_  installed.

   ..  code-block:: bash
      
       pip install pypardiso

   ..  code-block:: python
      
       import felupe as fe
       from pypardiso import PyPardisoSolver
       from scipy.sparse import triu
      
       # ...
       
       def solver(A, b):
           # mtype = 1: real and structurally symmetric, supernode pivoting
           # mtype = 2: real and symmetric positive definite
           # mtype =-2: real and symmetric indefinite, 
           #             diagonal or Bunch-Kaufman pivoting
           # mtype = 6: complex and symmetric
           return PyPardisoSolver(mtype=2).solve(triu(A).tocsr(), b).squeeze()
      
       system = fe.solve.partition(field, K, dof1, dof0)
       fe.solve.solve(*system, solver=solver)


    