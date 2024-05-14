.. _felupe-api-constitution:

Constitution
~~~~~~~~~~~~

This module provides :class:`constitutive material <felupe.ConstitutiveMaterial>` formulations.

There are many different pre-defined constitutive material formulations available, including definitions for linear-elasticity, small-strain plasticity, hyperelasticity or pseudo-elasticity. The generation of user materials may be simplified when using frameworks for user-defined functions, like hyperelasticity (with automatic differentiation) or a small-strain based framework with state variables. The most general case is given by a framework with functions for the evaluation of stress and elasticity tensors in terms of the deformation gradient.

**Base class (decorator) for constitutive material formulations**

.. currentmodule:: felupe

.. autosummary::

   ConstitutiveMaterial
   constitutive_material

**View Force-Stretch Curves on Elementary Deformations**

.. autosummary::

   ViewMaterial
   ViewMaterialIncompressible

**Merge Constitutive Materials**

.. autosummary::

   CompositeMaterial

**Linear-Elasticity**

.. autosummary::

   LinearElastic
   LinearElasticPlaneStrain
   LinearElasticPlaneStress
   constitution.LinearElasticTensorNotation
   LinearElasticLargeStrain

**Plasticity**

.. autosummary::

   LinearElasticPlasticIsotropicHardening

**Core Hyperelasticity (without Automatic Differentation)**

.. autosummary::

   NeoHooke
   NeoHookeCompressible
   OgdenRoxburgh
   Volumetric

**Hyperelastic Three-Field-Formulations** :math:`(\boldsymbol{u}, p, \bar{J})`

.. autosummary::

   ThreeFieldVariation
   NearlyIncompressible

**Hyperelasticity with Automatic Differentation**

.. autosummary::
   
   Hyperelastic
   total_lagrange

**Material Models (Strain Energy Functions) for** :class:`~felupe.Hyperelastic`

.. autosummary::

   alexander
   arruda_boyce
   extended_tube
   finite_strain_viscoelastic
   miehe_goektepe_lulei
   mooney_rivlin
   morph
   morph_representative_directions
   neo_hooke
   ogden
   ogden_roxburgh
   saint_venant_kirchhoff
   third_order_deformation
   van_der_waals
   yeoh

**Small Strain-based User Materials**

.. autosummary::

   MaterialStrain
   linear_elastic
   linear_elastic_plastic_isotropic_hardening

**Deformation Gradient-based User Materials**

.. autosummary::

   Material

**Kinematics**

.. autosummary::

   LineChange
   AreaChange
   VolumeChange

**Detailed API Reference**

.. autoclass:: felupe.ConstitutiveMaterial
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitutive_material

.. autoclass:: felupe.ViewMaterial
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.ViewMaterialIncompressible
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.CompositeMaterial
   :members:
   :undoc-members:
   :inherited-members:

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

.. autoclass:: felupe.Volumetric
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

.. autoclass:: felupe.NearlyIncompressible
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

.. autofunction:: felupe.alexander

.. autofunction:: felupe.arruda_boyce

.. autofunction:: felupe.extended_tube

.. autofunction:: felupe.finite_strain_viscoelastic

.. autofunction:: felupe.miehe_goektepe_lulei

.. autofunction:: felupe.mooney_rivlin

.. autofunction:: felupe.morph

.. autofunction:: felupe.morph_representative_directions

.. autofunction:: felupe.neo_hooke

.. autofunction:: felupe.ogden

.. autofunction:: felupe.ogden_roxburgh

.. autofunction:: felupe.saint_venant_kirchhoff

.. autofunction:: felupe.third_order_deformation

.. autofunction:: felupe.van_der_waals

.. autofunction:: felupe.yeoh

.. autofunction:: felupe.total_lagrange

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

.. autofunction:: felupe.constitution.lame_converter
