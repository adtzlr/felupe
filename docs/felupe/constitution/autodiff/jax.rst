.. _felupe-api-constitution-autodiff-jax:

Materials with Automatic Differentiation (JAX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page contains material model formulations with automatic differentiation using :mod:`jax`.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   constitution.jax.Hyperelastic
   constitution.jax.Material
   constitution.jax.total_lagrange
   constitution.jax.updated_lagrange

**Material Models for** :class:`felupe.constitution.jax.Hyperelastic`

These material model formulations are defined by a strain energy density function.

.. autosummary::

   felupe.constitution.jax.models.hyperelastic.mooney_rivlin
   felupe.constitution.jax.models.hyperelastic.third_order_deformation
   felupe.constitution.jax.models.hyperelastic.yeoh

**Material Models for** :class:`felupe.constitution.jax.Material`

.. autosummary::

   felupe.constitution.jax.models.lagrange.morph

**Tools**

.. autosummary::
   
   constitution.jax.vmap

**Detailed API Reference**

.. autoclass:: felupe.constitution.jax.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.constitution.jax.Material
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitution.jax.total_lagrange

.. autofunction:: felupe.constitution.jax.updated_lagrange

.. autofunction:: felupe.constitution.jax.models.hyperelastic.mooney_rivlin

.. autofunction:: felupe.constitution.jax.models.hyperelastic.third_order_deformation

.. autofunction:: felupe.constitution.jax.models.hyperelastic.yeoh

.. autofunction:: felupe.constitution.jax.models.lagrange.morph

.. autofunction:: felupe.constitution.jax.vmap
