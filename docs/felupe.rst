FElupe API
==========

FElupe consists of several (sub-) modules. Relevant functions and classes are available in the global namespace of FElupe where posssible. However, some classes or functions are only available in their respective submodule namespace due to naming conflicts, e.g.

- :class:`Line` (:class:`element.Line`) and :class:`mesh.Line` or 
- :class:`Triangle` (:class:`element.Triangle`) and :class:`mesh.Triangle`.

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
