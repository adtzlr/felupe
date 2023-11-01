FElupe API
==========

FElupe consists of several (sub-) modules. Relevant functions and classes are available in the global namespace of FElupe where posssible. However, some classes or functions are only available in their respective submodule namespace due to naming conflicts, e.g.

- :class:`fem.Line` (:class:`fem.element.Line`) and :class:`fem.mesh.Line` or 
- :class:`fem.Triangle` (:class:`fem.element.Triangle`) and :class:`fem.mesh.Triangle`.

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
