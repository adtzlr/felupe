.. _felupe-api-assembly:

Assembly
~~~~~~~~

This module contains classes for the integration and assembly of (weak) integral forms
into dense or sparse vectors and matrices. The integration algorithm switches 
automatically between general cartesion, plane strain or axisymmetric routines, 
dependent on the given fields.

..  hint::
    :class:`~felupe.IntegralForm` is used in the :ref:`felupe-api-mechanics` module
    (e.g. in a :class:`~felupe.SolidBody`) to integrate and/or assemble a
    :class:`constitutive material formulation <felupe.ConstitutiveMaterial>`
    and to provide an ``item`` for a :class:`~felupe.Step`
    or to use it in :func:`~felupe.newtonrhapson` directly.

**Core**

Take arrays for some pre-defined weak-forms and integrate them into dense or assembly
them into sparse vectors or matrices.

.. currentmodule:: felupe

.. autosummary::

   IntegralForm
   assembly.IntegralFormCartesian
   assembly.IntegralFormAxisymmetric


**Form Expressions**

Define weak-form expressions on-the-fly for flexible and general form expressions.

.. autosummary::

   Form

Create an item out of bilinear and linear weak-form expressions for a Step.

.. autosummary::

   FormItem

**Detailed API Reference**

.. autoclass:: felupe.IntegralForm
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.assembly.IntegralFormCartesian
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.assembly.IntegralFormAxisymmetric
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.Form

.. autoclass:: felupe.FormItem
   :members:
   :undoc-members:
   :inherited-members:
