.. _felupe-api-constitution-autodiff-jax:

Materials with Automatic Differentiation (JAX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page contains material model formulations with automatic differentiation using
:mod:`jax`.

..  note::

    JAX uses single-precision (32bit) data types by default. This requires to relax the
    tolerance of :func:`~felupe.newtonrhapson` to ``tol=1e-4``. If required, JAX may be
    enforced to use double-precision at startup with
    ``jax.config.update("jax_enable_x64", True)``.

..  note::

    The number of local XLA devices available must be greater or equal the number of the
    parallel-mapped axis, i.e. the number of quadrature points per cell when used in
    :class:`~felupe.constitution.jax.Material` and
    :class:`~felupe.constitution.jax.Hyperelastic` along with ``parallel=True``. To use
    the multiple cores of a CPU device as multiple local XLA devices, the XLA device
    count must be defined at startup.
    
    ..  code-block:: python
        
        import os

        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

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
   storakers
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

.. autofunction:: felupe.constitution.jax.models.hyperelastic.storakers

.. autofunction:: felupe.constitution.jax.models.hyperelastic.third_order_deformation

.. autofunction:: felupe.constitution.jax.models.hyperelastic.yeoh

.. autofunction:: felupe.constitution.jax.models.lagrange.morph

.. autofunction:: felupe.constitution.jax.models.lagrange.morph_representative_directions

.. autofunction:: felupe.constitution.jax.vmap
