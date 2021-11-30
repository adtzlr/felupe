Use external solvers
--------------------

FElupe uses SuperLU as direct sparse solver by default. Not because it is fact - just because it is shipped with scipy. While it is definitely a good choice for small to mid-sized problems, faster alternatives are easy to use. This guide shows two possibilities, a) a fast direct solver from `PyPardiso <https://github.com/haasad/PyPardisoProject>`_ (``pip install pypardiso``) and all iterative solvers from `Krylov <https://github.com/nschloe/krylov>`_ (``pip install krylov``).

.. tab:: SuperLU (default)

   .. code-block:: python
      
      import felupe as fe
      
      # ...
      
      system = fe.solve.partition(field, K, dof1, dof0)
      fe.solve.solve(*system)

.. tab:: PyPardiso

   Ensure to have `PyPardiso <https://github.com/haasad/PyPardisoProject>`_  installed:

   .. code-block:: bash
      
      pip install pypardiso
   

   .. code-block:: python
      
      import felupe as fe
      from pypardiso import spsolve as solver
      
      # ...
      
      system = fe.solve.partition(field, K, dof1, dof0)
      fe.solve.solve(*system, solver=solver)
        

.. tab:: Krylov

   Ensure to have `Krylov <https://github.com/nschloe/krylov>`_ installed:

   .. code-block:: bash
      
      pip install krylov

   ..  code-block:: python
        
       import felupe as fe
       import krylov
       
       # ...
       
       def solver(A, b):
           "Wrapper function for Krylov-solvers".
           
           return krylov.minres(A, b)[0]
       
       system = fe.solve.partition(field, K, dof1, dof0)
       fe.solve.solve(*system, solver=solver)
        


    