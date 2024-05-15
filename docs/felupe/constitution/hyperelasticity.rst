.. _felupe-api-constitution-hyperelasticity:

Hyperelasticity
~~~~~~~~~~~~~~~

This page contains hyperelastic material model formulations with automatic differentiation using `tensortrax <https://github.com/adtzlr/tensortrax>`_. These material model formulations are defined by a strain energy density function.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   Hyperelastic

**Material Models (Strain Energy Functions) for** :class:`~felupe.Hyperelastic`

.. autosummary::

   alexander
   arruda_boyce
   extended_tube
   finite_strain_viscoelastic
   miehe_goektepe_lulei
   mooney_rivlin
   constitution.hyperelasticity.models.morph_representative_directions
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

.. autofunction:: felupe.arruda_boyce

.. autofunction:: felupe.extended_tube

.. autofunction:: felupe.finite_strain_viscoelastic

.. autofunction:: felupe.miehe_goektepe_lulei

.. autofunction:: felupe.mooney_rivlin

.. autofunction:: felupe.morph_representative_directions

.. autofunction:: felupe.neo_hooke

.. autofunction:: felupe.ogden

.. autofunction:: felupe.ogden_roxburgh

.. autofunction:: felupe.saint_venant_kirchhoff

.. autofunction:: felupe.third_order_deformation

.. autofunction:: felupe.van_der_waals

.. autofunction:: felupe.yeoh
