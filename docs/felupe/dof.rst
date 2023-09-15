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
