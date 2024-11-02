.. _felupe-api-constitution-autodiff:

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: Constitution:

   constitution/autodiff/tensortrax
   constitution/autodiff/jax

FElupe supports multiple backends for constitutive material formulations with
automatic differentiation. The default backend is based on :mod:`tensortrax` which ships
with FElupe. For more computationally expensive material formulations, :mod:`jax` may
be the preferred option.

It is straightforward to switch between these backends.

..  tab:: tensortrax

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