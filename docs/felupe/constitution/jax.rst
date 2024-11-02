.. _felupe-api-constitution-jax:

JAX-based Materials
~~~~~~~~~~~~~~~~~~~

This page contains material model formulations with automatic differentiation using :mod:`jax`.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   constitution.jax.Hyperelastic
   constitution.jax.Material

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

.. autofunction:: felupe.constitution.jax.models.lagrange.morph

.. autofunction:: felupe.constitution.jax.vmap
