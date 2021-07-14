# Simple mesh module of FElupe

FElupe provides a simple mesh generation module `felupe.mesh`. A Mesh instance contains essentially two arrays: one with **points** (*nodes*) and another one containing the cell connectivities, called **cells**. Only a single cell type is supported per mesh. Optionally the element type is specified which is used if the mesh is saved as VTK or XDMF file. These element types are identical to cell types used in `meshio`: `line`, `quad` and `hexahedron` for linear elements or `VTK_LAGRANGE_HEXAHEDRON` for elements with arbitrary order.

```python
import numpy as np
import felupe as fe

points = np.array([
    [ 0, 0], # point 1
    [ 1, 0], # point 2
    [ 0, 1], # point 3
    [ 1, 1], # point 4
])

cells = np.array([
    [ 0, 1, 3, 2], # connectivity of first cell
])

m = fe.mesh.Mesh(points, cells, cell_type="quad")
```

## Generate a cube by hand
First we start with the generation of a line from `1` to `3` with `n=2` points. Next, we expand the line to a rectangle. The `z` argument in expand represents the total expansion. Again, an expansion of our rectangle leads to a hexahedron. Several other useful tools are available beside `expand`: `rotate`, `revolve` and `sweep`. With these simple tools at hand, i.e. rectangles, cubes or cylinders may be constructed.

```python
line = fe.mesh.line_line(a=1, b=3, n=2)
rect = fe.mesh.expand(line, n=2, z=5)
hexa = fe.mesh.expand(rect, n=2, z=3)

m = fe.mesh.Mesh(*hexa, cell_type="hexahedron")
```

## Generate lines, rectangles and cubes
Of course lines, rectangles, cubes and cylinders do not have to be created by hand each time. Instead, some easier to use classes are povided by FElupe.

```python
mesh_line      = fe.mesh.Line(     a=0,         b=1,         n=2)
mesh_rectangle = fe.mesh.Rectangle(a=(0, 0),    b=(1, 1),    n=(2, 2))
mesh_cube      = fe.mesh.Cube(     a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2))

mesh_cylinder  = fe.mesh.Cylinder(D=2, H=1, n=(3, 9, 3), phi=360) # phi in degree
```