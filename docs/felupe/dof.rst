Degrees of Freedom
==================

This module contains the definition of a boundary condition as well as tools related to the partition of degrees of freedom and the application of boundary conditions on a field container.

**Core**

.. autosummary::

   felupe.Boundary


**Tools**

.. autosummary::

   felupe.dof.partition
   felupe.dof.apply
   felupe.dof.symmetry


**Load Cases**

.. autosummary::

   felupe.dof.uniaxial
   felupe.dof.biaxial


**Detailed API Reference**

.. autoclass:: felupe.Boundary
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.dof.partition

.. autofunction:: felupe.dof.apply

.. autofunction:: felupe.dof.symmetry

.. autofunction:: felupe.dof.uniaxial

.. autofunction:: felupe.dof.biaxial

.. autofunction:: felupe.dof.shear
