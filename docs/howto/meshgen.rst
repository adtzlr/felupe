Generate Meshes
~~~~~~~~~~~~~~~

FElupe provides a simple mesh generation module :mod:`felupe.mesh`. A Mesh instance contains essentially two arrays: one with ``points`` and another one containing the cell connectivities, called ``cells``. Only a single ``cell_type`` is supported per mesh. Optionally the ``cell_type`` is specified which is used if the mesh is saved as a VTK or a XDMF file. These cell types are identical to cell types used in meshio (`VTK types <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_): ``line``, ``quad`` and ``hexahedron`` for linear lagrange elements or ``triangle`` and  ``tetra`` for 2- and 3-simplices or ``VTK_LAGRANGE_HEXAHEDRON`` for 3d lagrange-cells with polynomial shape functions of arbitrary order.

..  code-block:: python

    import numpy as np
    import felupe as fe

    points = np.array([
        [ 0, 0], # point 1
        [ 1, 0], # point 2
        [ 0, 1], # point 3
        [ 1, 1], # point 4
    ], dtype=float)

    cells = np.array([
        [ 0, 1, 3, 2], # point-connectivity of first cell
    ])

    mesh = fe.Mesh(points, cells, cell_type="quad")

.. image:: images/quad.png
   :width: 400px


Generate a cube by hand
***********************

First let's start with the generation of a line from ``x=1`` to ``x=3`` with ``n=2`` points. Next, the line is expanded into a rectangle. The ``z`` argument of :func:`felupe.mesh.expand` represents the total expansion. Again, an expansion of our rectangle leads to a hexahedron. Several other useful functions are available beside :func:`felupe.mesh.expand`: :func:`felupe.mesh.rotate`, :func:`felupe.mesh.revolve` and :func:`felupe.mesh.sweep`. With these simple tools at hand, rectangles, cubes or cylinders may be constructed with ease.

..  code-block:: python

    line = fe.mesh._line_line(a=1, b=3, n=7)
    rect = fe.mesh.expand(line, n=2, z=5)
    hexa = fe.mesh.expand(rect, n=2, z=3)

    mesh = fe.Mesh(*hexa, cell_type="hexahedron")

.. image:: images/cube.png
   :width: 400px


Lines, rectangles and cubes
***************************

Of course lines, rectangles, cubes and cylinders do not have to be constructed manually each time. Instead, some easier to use classes are povided by FElupe like :class:`felupe.mesh.Line`, :class:`felupe.Rectangle` or :class:`felupe.Cube`.

Triangle and Tetrahedron meshes
*******************************

FElupe does not provide any tools for the creation of meshes consisting of triangles or tetrahedrons. For a similar interface and simple geometries it is recommended to use `meshzoo <https://github.com/nschloe/meshzoo>`_ instead (install with ``pip install meshzoo``).

..  code-block:: python

    import numpy as np
    import meshzoo

    cube = meshzoo.cube_tetra(
        np.linspace(0,1,11), np.linspace(0,1,11), np.linspace(0,1,11)
    )
    mesh = fe.Mesh(*cube, cell_type="tetra")

Meshes with midpoints
*********************

If a mesh with midpoints is required by a region, functions for edge, face and volume midpoint insertions are provided in :func:`felupe.mesh.add_midpoints_edges`, :func:`felupe.mesh.add_midpoints_faces` and :func:`felupe.mesh.add_midpoints_volumes`. A low-order mesh, e.g. a mesh with cell-type `quad`, can be converted to a quadratic mesh with :func:`felupe.mesh.convert`. By default, only midpoints on edges are inserted. Hence, the resulting cell-type is ``quad8``. If midpoints on faces are also calculated, the resulting cell-type is ``quad9``.

..  code-block:: python
    
    rectangle_quad4 = fe.Rectangle(n=6)
    rectangle_quad8 = fe.mesh.convert(rectangle, order=2)
    rectangle_quad9 = fe.mesh.convert(rectangle, order=2, calc_midfaces=True)

The same also applies on meshes with triangles. Meshzoo offers an equivalent midpoint insertion.

..  code-block:: python

    rectangle = meshzoo.rectangle_tri(np.linspace(0,1,6), np.linspace(0,1,6))
    rectangle_triangle6 = meshzoo.insert_midpoints_edges(*rectangle, "triangle")
    
    mesh = fe.Mesh(*rectangle_triangle6, cell_type="triangle6")
    # mesh = fe.Mesh(*fe.mesh.add_midpoints_edges(*rectangle, "triangle"))