Generate Meshes
~~~~~~~~~~~~~~~

FElupe provides a simple mesh generation module :mod:`felupe.mesh`. A Mesh instance contains essentially two arrays: one with ``points`` and another one containing the cell connectivities, called ``cells``. Only a single ``cell_type`` is supported per mesh. Optionally the ``cell_type`` is specified which is used if the mesh is saved as a VTK or a XDMF file. These cell types are identical to cell types used in meshio (`VTK types <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_): ``line``, ``quad`` and ``hexahedron`` for linear lagrange elements or ``triangle`` and  ``tetra`` for 2- and 3-simplices or ``VTK_LAGRANGE_HEXAHEDRON`` for 3d lagrange-cells with polynomial shape functions of arbitrary order.

..  code-block:: python

    import numpy as np
    import felupe as fem

    points = np.array([
        [ 0, 0], # point 1
        [ 1, 0], # point 2
        [ 0, 1], # point 3
        [ 1, 1], # point 4
    ], dtype=float)

    cells = np.array([
        [ 0, 1, 3, 2], # point-connectivity of first cell
    ])

    mesh = fem.Mesh(points, cells, cell_type="quad")

.. image:: images/quad.png
   :width: 400px


Generate a cube by hand
***********************

First let's start with the generation of a line from ``x=1`` to ``x=3`` with ``n=2`` points. Next, the line is expanded into a rectangle. The ``z`` argument of :func:`felupe.mesh.expand` represents the total expansion. Again, an expansion of our rectangle leads to a hexahedron. Several other useful functions are available beside :func:`felupe.mesh.expand`: :func:`felupe.mesh.rotate`, :func:`felupe.mesh.revolve` and :func:`felupe.mesh.sweep`. With these simple tools at hand, rectangles, cubes or cylinders may be constructed with ease.

..  code-block:: python

    line = fem.mesh.Line(a=1, b=3, n=7)
    rect = fem.mesh.expand(line, n=2, z=5)
    hexa = fem.mesh.expand(rect, n=2, z=3)

.. image:: images/cube.png
   :width: 400px


Lines, rectangles and cubes
***************************

Of course lines, rectangles, cubes and cylinders do not have to be constructed manually each time. Instead, some easier to use classes are povided by FElupe like :class:`felupe.mesh.Line`, :class:`felupe.Rectangle` or :class:`felupe.Cube`. For non equi-distant points per axis use :class:`felupe.Grid`.

Triangle and Tetrahedron meshes
*******************************

Any quad or tetrahedron mesh may be subdivided (triangulated) to meshes out of Triangle or Tetrahedrons by :func:`felupe.mesh.triangulate`.

..  code-block:: python

    rectangle_quad = fem.Rectangle(n=5)
    rectangle_tetra = fem.mesh.triangulate(rect_quad)

    cube_hexahedron = fem.Cube(n=5)
    cube_tetra = fem.mesh.triangulate(cube_hexahedron)

Meshes with midpoints
*********************

If a mesh with midpoints is required by a region, functions for edge, face and volume midpoint insertions are provided in :func:`felupe.mesh.add_midpoints_edges`, :func:`felupe.mesh.add_midpoints_faces` and :func:`felupe.mesh.add_midpoints_volumes`. A low-order mesh, e.g. a mesh with cell-type `quad`, can be converted to a quadratic mesh with :func:`felupe.mesh.convert`. By default, only midpoints on edges are inserted. Hence, the resulting cell-type is ``quad8``. If midpoints on faces are also calculated, the resulting cell-type is ``quad9``.

..  code-block:: python
    
    rectangle_quad4 = fem.Rectangle(n=6)
    rectangle_quad8 = fem.mesh.convert(rectangle_quad4, order=2)
    rectangle_quad9 = fem.mesh.convert(rectangle_quad4, order=2, calc_midfaces=True)

The same also applies on meshes with triangles.

..  code-block:: python

    rectangle_triangle3 = fem.mesh.triangulate(fem.Rectangle(n=6))
    rectangle_triangle6 = fem.mesh.add_midpoints_edges(rectangle_triangle3)