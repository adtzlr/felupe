.. _felupe-api-constitution-hyperelasticity:

Hyperelasticity
~~~~~~~~~~~~~~~

This page contains hyperelastic material model formulations with automatic differentiation using :mod:`tensortrax`. These material model formulations are defined by a strain energy density function.

.. figure:: /_static/logo_tensortrax.png
   :align: center

   Differentiable Tensors based on NumPy Arrays.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   Hyperelastic

**Material Models (Strain Energy Functions) for** :class:`~felupe.Hyperelastic`

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

**Detailed API Reference**

.. autoclass:: felupe.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

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
