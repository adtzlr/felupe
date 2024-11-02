.. _felupe-api-constitution-autodiff-tensortrax:

Materials with Automatic Differentiation (tensortrax)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page contains hyperelastic material model formulations with automatic differentiation using :mod:`tensortrax.math`.

.. figure:: /_static/logo_tensortrax.png
   :align: center

   Differentiable Tensors based on NumPy Arrays.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   constitution.tensortrax.Hyperelastic
   constitution.tensortrax.Material
   constitution.tensortrax.total_lagrange
   constitution.tensortrax.updated_lagrange

**Material Models for** :class:`felupe.constitution.tensortrax.Hyperelastic`

These material model formulations are defined by a strain energy density function.

.. autosummary::

   alexander
   anssari_benam_bucchi
   arruda_boyce
   extended_tube
   finite_strain_viscoelastic
   lopez_pamies
   miehe_goektepe_lulei
   mooney_rivlin
   neo_hooke
   ogden
   ogden_roxburgh
   saint_venant_kirchhoff
   third_order_deformation
   van_der_waals
   yeoh

**Material Models for** :class:`~felupe.constitution.tensortrax.Material`

The material model formulations are defined by the first Piola-Kirchhoff stress tensor.
Function-decorators are available to use Total-Lagrange and Updated-Lagrange material
formulations in :class:`~felupe.constitution.tensortrax.Material`.

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

.. autofunction:: felupe.total_lagrange

.. autofunction:: felupe.updated_lagrange

.. autofunction:: felupe.alexander

.. autofunction:: felupe.anssari_benam_bucchi

.. autofunction:: felupe.arruda_boyce

.. autofunction:: felupe.extended_tube

.. autofunction:: felupe.finite_strain_viscoelastic

.. autofunction:: felupe.lopez_pamies

.. autofunction:: felupe.miehe_goektepe_lulei

.. autofunction:: felupe.mooney_rivlin

.. autofunction:: felupe.neo_hooke

.. autofunction:: felupe.ogden

.. autofunction:: felupe.ogden_roxburgh

.. autofunction:: felupe.saint_venant_kirchhoff

.. autofunction:: felupe.third_order_deformation

.. autofunction:: felupe.van_der_waals

.. autofunction:: felupe.yeoh

.. autofunction:: felupe.morph

.. autofunction:: felupe.morph_representative_directions