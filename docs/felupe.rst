FElupe API
==========

FElupe consists of several (sub-) modules. Relevant functions and classes are available in the global namespace of FElupe where posssible. However, some classes or functions are only available in their respective submodule namespace due to naming conflicts, e.g.

- :class:`felupe.Line` (:class:`felupe.element.Line`) and :class:`felupe.mesh.Line` or 
- :class:`felupe.Triangle` (:class:`felupe.element.Triangle`) and :class:`felupe.mesh.Triangle`.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   felupe/mesh
   felupe/element
   felupe/quadrature
   felupe/region
   felupe/field
   felupe/basis
   felupe/constitution
   felupe/assembly
   felupe/dof
   felupe/tools
   felupe/mechanics
