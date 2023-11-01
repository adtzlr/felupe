Assembly
~~~~~~~~

This module contains classes for the integration and assembly of (weak) integral forms
into dense or sparse vectors and matrices. The integration algorithm switches 
automatically between general cartesion, plane strain or axisymmetric routines, 
dependent on the given fields.

**Core**

Take arrays for some pre-defined weak-forms and integrate them into dense or assembly
them into sparse vectors or matrices.

.. currentmodule:: felupe

.. autosummary::

   IntegralForm


**Form Expressions**

Define weak-form expressions on-the-fly for flexible and general form expressions.

.. autosummary::

   Form

**Detailed API Reference**

.. autoclass:: fem.IntegralForm
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: fem.Form