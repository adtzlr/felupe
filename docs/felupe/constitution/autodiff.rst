.. _felupe-api-constitution-autodiff:

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Automatic Differentiation:

   autodiff/tensortrax
   autodiff/jax

.. grid::

   .. grid-item-card:: NumPy
      :link: autodiff/tensortrax
      :link-type: doc

      Default NumPy-based backend ``tensortrax`` for automatic differentiation
      (bundled with FElupe).

   .. grid-item-card:: JAX
      :link: autodiff/jax
      :link-type: doc

      Use JAX as backend for automatic differentiation. Preferred option for
      computationally expensive material formulations.

FElupe supports multiple backends for constitutive material formulations with
automatic differentiation. It is straightforward to switch between these backends.

..  tab:: tensortrax (default)

    ..  code-block::
        
        import felupe as fem
        import felupe.constitution.tensortrax as mat
        import tensortrax.math as tm

        def neo_hooke(C, mu):
            "Strain energy function of the Neo-Hookean material formulation."
            return mu / 2 * (tm.linalg.det(C) ** (-1/3) * tm.trace(C) - 3)

        umat = mat.Hyperelastic(neo_hooke, mu=1.0)


..  tab:: JAX

    ..  code-block::
        
        import felupe as fem
        import felupe.constitution.jax as mat
        import jax.numpy as jnp

        def neo_hooke(C, mu):
            "Strain energy function of the Neo-Hookean material formulation."
            return mu / 2 * (jnp.linalg.det(C) ** (-1/3) * jnp.trace(C) - 3)

        umat = mat.Hyperelastic(neo_hooke, mu=1.0)


..  note::

    The default backend is available in the top-level package namespace, this includes
    all models from :mod:`felupe.constitution.tensortrax.models.hyperelastic` and
    :mod:`felupe.constitution.tensortrax.models.lagrange` as well as the material
    classes :class:`felupe.constitution.tensortrax.Material` and
    :class:`felupe.constitution.tensortrax.Hyperelastic`.
