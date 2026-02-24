.. _felupe-api-constitution-autodiff-tensortrax:

Materials with Automatic Differentiation (tensortrax)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page contains hyperelastic material model formulations with automatic differentiation using :mod:`tensortrax.math`.

.. figure:: /_static/logo_tensortrax.png
   :align: center

   Differentiable Tensors based on NumPy Arrays.

**Frameworks**

.. currentmodule:: felupe.constitution.tensortrax

.. autosummary::
   
   Hyperelastic
   Material
   total_lagrange
   updated_lagrange

**Material Models for** :class:`felupe.constitution.tensortrax.Hyperelastic`

These material model formulations are defined by a strain energy density function.

.. currentmodule:: felupe.constitution.tensortrax.models.hyperelastic

.. autosummary::

   alexander
   anssari_benam_bucchi
   arruda_boyce
   blatz_ko
   extended_tube
   finite_strain_viscoelastic
   lopez_pamies
   miehe_goektepe_lulei
   mooney_rivlin
   neo_hooke
   ogden
   ogden_roxburgh
   saint_venant_kirchhoff
   saint_venant_kirchhoff_orthotropic
   storakers
   third_order_deformation
   van_der_waals
   yeoh

**Material Models for** :class:`felupe.constitution.tensortrax.Material`

The material model formulations are defined by the first Piola-Kirchhoff stress tensor.
Function-decorators are available to use :func:`~felupe.constitution.tensortrax.total_lagrange`
and :func:`~felupe.constitution.tensortrax.updated_lagrange` material formulations in
:class:`~felupe.constitution.tensortrax.Material`.

.. currentmodule:: felupe.constitution.tensortrax.models.lagrange

.. autosummary::

   morph
   morph_representative_directions

**Detailed API Reference**

.. autoclass:: felupe.constitution.tensortrax.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.constitution.tensortrax.Material
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.MaterialAD
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitution.tensortrax.total_lagrange

.. autofunction:: felupe.constitution.tensortrax.updated_lagrange

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.alexander

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.anssari_benam_bucchi

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.arruda_boyce

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.blatz_ko

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.extended_tube

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.finite_strain_viscoelastic

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.lopez_pamies

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.miehe_goektepe_lulei

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.mooney_rivlin

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.neo_hooke

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.ogden

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.ogden_roxburgh

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.saint_venant_kirchhoff

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.saint_venant_kirchhoff_orthotropic

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.storakers

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.third_order_deformation

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.van_der_waals

.. autofunction:: felupe.constitution.tensortrax.models.hyperelastic.yeoh

.. autofunction:: felupe.constitution.tensortrax.models.lagrange.morph

.. autofunction:: felupe.constitution.tensortrax.models.lagrange.morph_representative_directions
