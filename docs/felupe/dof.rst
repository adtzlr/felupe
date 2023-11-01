Degrees of Freedom
==================

This module contains the definition of a boundary condition, tools related to the handling of degrees of freedom as well as boundary condition templates for simple load cases.

**Core**

.. currentmodule:: felupe

.. autosummary::

   Boundary


**Tools**

.. autosummary::

   dof.partition
   dof.apply
   dof.symmetry


**Load Cases**

.. autosummary::

   dof.uniaxial
   dof.biaxial
   dof.shear


**Detailed API Reference**

.. autoclass:: fem.Boundary
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: fem.dof.partition

.. autofunction:: fem.dof.apply

.. autofunction:: fem.dof.symmetry

.. autofunction:: fem.dof.uniaxial

.. autofunction:: fem.dof.biaxial

.. autofunction:: fem.dof.shear
