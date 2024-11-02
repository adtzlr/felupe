.. _felupe-api-constitution-jax:

JAX-based Materials
~~~~~~~~~~~~~~~~~~~

This page contains material model formulations with automatic differentiation using :mod:`jax`.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   constitution.autodiff.jax.Hyperelastic
   constitution.autodiff.jax.Material

**Material Models for** :class:`felupe.constitution.autodiff.jax.Material`

.. autosummary::

   felupe.constitution.autodiff.jax.models.lagrange.morph

**Tools**

.. autosummary::
   
   constitution.autodiff.jax.vmap

**Detailed API Reference**

.. autoclass:: felupe.constitution.autodiff.jax.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.constitution.autodiff.jax.Material
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitution.autodiff.jax.models.lagrange.morph

.. autofunction:: felupe.constitution.autodiff.jax.vmap
