.. _felupe-api-constitution:

Constitution
~~~~~~~~~~~~

This module provides constitutive material formulations.

**Linear-Elasticity**

.. currentmodule:: felupe

.. autosummary::

   LinearElastic
   LinearElasticPlaneStrain
   LinearElasticPlaneStress
   constitution.LinearElasticTensorNotation
   LinearElasticLargeStrain

**Plasticity**

.. autosummary::

   LinearElasticPlasticIsotropicHardening

**Hyperelasticity**

.. autosummary::

   NeoHooke
   NeoHookeCompressible
   ThreeFieldVariation

**Pseudo-Elasticity (Isotropic Damage)**

.. autosummary::

   OgdenRoxburgh

**Hyperelastic User-Materials with Automatic Differentation**

.. autosummary::
   
   Hyperelastic
   MaterialAD
   saint_venant_kirchhoff
   neo_hooke
   mooney_rivlin
   yeoh
   third_order_deformation
   ogden
   arruda_boyce
   extended_tube
   van_der_waals
   finite_strain_viscoelastic

**(Small) Strain-based User Materials**

.. autosummary::

   MaterialStrain
   linear_elastic
   linear_elastic_plastic_isotropic_hardening

**Deformation-Gradient-based User Materials**

.. autosummary::

   Material

**Kinematics**

.. autosummary::

   LineChange
   AreaChange
   VolumeChange

**Detailed API Reference**

.. autoclass:: felupe.NeoHooke
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.NeoHookeCompressible
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.OgdenRoxburgh
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticLargeStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.constitution.LinearElasticTensorNotation
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlaneStress
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlaneStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlasticIsotropicHardening
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.ThreeFieldVariation
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Material
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.MaterialStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.linear_elastic

.. autofunction:: felupe.linear_elastic_plastic_isotropic_hardening

.. autoclass:: felupe.MaterialAD
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.saint_venant_kirchhoff

.. autofunction:: felupe.neo_hooke

.. autofunction:: felupe.mooney_rivlin

.. autofunction:: felupe.yeoh

.. autofunction:: felupe.third_order_deformation

.. autofunction:: felupe.ogden

.. autofunction:: felupe.arruda_boyce

.. autofunction:: felupe.extended_tube

.. autofunction:: felupe.van_der_waals

.. autofunction:: felupe.finite_strain_viscoelastic

.. autoclass:: felupe.LineChange
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.AreaChange
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.VolumeChange
   :members:
   :undoc-members:
   :inherited-members:
