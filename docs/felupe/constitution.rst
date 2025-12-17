.. _felupe-api-constitution:

Constitution
~~~~~~~~~~~~

This module provides :class:`constitutive material <felupe.ConstitutiveMaterial>` formulations.

.. grid::

   .. grid-item-card:: 🏠 Models
      :link: constitution/core
      :link-type: ref

      Core frameworks and library with basic material models.

   .. grid-item-card:: 🧮 Automatic Differentiation
      :link: constitution/autodiff
      :link-type: ref

      Advanced frameworks and material models with support for automatic differentiation.

   .. grid-item-card:: 🔧 Tools & Helpers
      :link: constitution/tools
      :link-type: ref

      Utilities to speed up repetitive & complicated tasks.


There are many different pre-defined constitutive material formulations available, including definitions for linear-elasticity, small-strain plasticity, hyperelasticity or pseudo-elasticity. The generation of user materials may be simplified when using frameworks for user-defined functions, like hyperelasticity (with automatic differentiation) or a small-strain based framework with state variables. However, the most general case is given by a framework with functions for the evaluation of stress and elasticity tensors in terms of the deformation gradient.

**Constitutive Material Formulation**

.. currentmodule:: felupe

.. autosummary::

   ConstitutiveMaterial
   constitutive_material

**Deformation Gradient-based Materials**

.. autosummary::

   Material

**Detailed API Reference**

.. autoclass:: felupe.ConstitutiveMaterial
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitutive_material

.. autoclass:: felupe.Material
   :members:
   :undoc-members:
   :inherited-members:
