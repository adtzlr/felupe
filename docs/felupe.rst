.. _felupe-api:

API Reference
=============

FElupe consists of several (sub-) modules. Relevant functions and classes are available in the global namespace of FElupe where posssible. However, some classes or functions are only available in their respective submodule namespace due to naming conflicts, e.g.

- :class:`~felupe.Line` (:class:`element.Line <felupe.element.Line>`) and :class:`mesh.Line <felupe.mesh.Line>` or
- :class:`~felupe.Triangle` (:class:`element.Triangle <felupe.element.Triangle>`) and :class:`mesh.Triangle <felupe.mesh.Triangle>`.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   felupe/mesh
   felupe/element
   felupe/quadrature
   felupe/region
   felupe/field
   felupe/constitution
   felupe/assembly
   felupe/dof
   felupe/tools
   felupe/mechanics
   felupe/math
   felupe/view
