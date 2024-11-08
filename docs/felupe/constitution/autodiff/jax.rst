.. _felupe-api-constitution-autodiff-jax:

Materials with Automatic Differentiation (JAX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page contains material model formulations with automatic differentiation using :mod:`jax`.

**Frameworks**

.. currentmodule:: felupe.constitution.jax

.. autosummary::
   
   Hyperelastic
   Material
   total_lagrange
   updated_lagrange

**Material Models for** :class:`felupe.constitution.jax.Hyperelastic`

These material model formulations are defined by a strain energy density function.

.. currentmodule:: felupe.constitution.jax.models.hyperelastic

.. autosummary::

   miehe_goektepe_lulei
   mooney_rivlin
   third_order_deformation
   yeoh

**Material Models for** :class:`felupe.constitution.jax.Material`

The material model formulations are defined by the first Piola-Kirchhoff stress tensor.
Function-decorators are available to use Total-Lagrange and Updated-Lagrange material
formulations in :class:`~felupe.constitution.jax.Material`.

.. currentmodule:: felupe.constitution.jax.models.lagrange

.. autosummary::

   morph
   morph_representative_directions

**Tools**

.. currentmodule:: felupe.constitution.jax

.. autosummary::
   
   vmap

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

.. autofunction:: felupe.constitution.jax.models.hyperelastic.miehe_goektepe_lulei

.. autofunction:: felupe.constitution.jax.models.hyperelastic.mooney_rivlin

.. autofunction:: felupe.constitution.jax.models.hyperelastic.third_order_deformation

.. autofunction:: felupe.constitution.jax.models.hyperelastic.yeoh

.. autofunction:: felupe.constitution.jax.models.lagrange.morph

.. autofunction:: felupe.constitution.jax.models.lagrange.morph_representative_directions

.. autofunction:: felupe.constitution.jax.vmap
