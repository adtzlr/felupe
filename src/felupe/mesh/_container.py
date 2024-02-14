# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

from copy import deepcopy

import numpy as np

from ._mesh import Mesh
from ._tools import merge_duplicate_points as sweep


class MeshContainer:
    """A container which operates on a list of meshes with identical dimensions.

    Parameters
    ----------
    meshes : list of Mesh
        A list of meshes which are organized by the mesh container.
    merge : bool, optional
        Flag to merge duplicate mesh points. This changes the cells arrays of the
        meshes. Default is False.
    decimals : float or None, optional
        Precision decimals for merging duplicated mesh points. Only relevant if
        merge=True. Default is None.

    Notes
    -----
    All meshes are modified to refer to the same points array. By default, the points
    arrays from the given list of meshes is concatenated and the cells arrays are
    modified accordingly. Optionally, the points array may be merged on duplicated
    points.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> cube = fem.Cube(n=3)
    >>> cylinder = fem.Circle().expand(n=2)
    >>> mesh = fem.MeshContainer([cube, cylinder])
    >>> mesh
    <felupe mesh container object>
      Number of points: 61
      Number of cells:
        hexahedron: 8
        hexahedron: 12

    The cells array of the second mesh starts with an offset

    >>> mesh.meshes[1].cells.min()
    27

    identical to the number of points from the first mesh.

    >>> cube.npoints
    27

    If the container is created with ``merge=True``, then the number of points is lower
    than before.

    >>> mesh = fem.MeshContainer([cube, cylinder], merge=True)
    >>> mesh
    <felupe mesh container object>
      Number of points: 51
      Number of cells:
        hexahedron: 8
        hexahedron: 12

    ..  image:: images/container.png
        :width: 400px

    """

    def __init__(self, meshes, merge=False, decimals=None):
        # obtain the dimension from the first mesh
        self.dim = meshes[0].dim

        # init points and list of meshes
        self.points = np.zeros((0, self.dim))
        self.meshes = []

        # append all meshes
        [self.append(mesh) for mesh in meshes]

        if merge:
            self.merge_duplicate_points(decimals=decimals)

    def append(self, mesh):
        "Append a :class:`~felupe.Mesh` to the list of meshes."

        # number of points
        points = np.vstack([self.points, mesh.points])
        self.meshes.append(Mesh(points, mesh.cells + len(self.points), mesh.cell_type))

        # ensure identical points-arrays
        for i, m in enumerate(self.meshes):
            self.meshes[i].points = self.points = points

    def pop(self, index):
        "Pop an item of the list of meshes."
        item = self.meshes.pop(index)
        return item

    def cells(self):
        "Return a list of tuples with cell-types and cell-connectivities."
        return [(mesh.cell_type, mesh.cells) for mesh in self.meshes]

    def merge_duplicate_points(self, decimals=None):
        "Merge duplicate points and update the meshes."

        # sweep points
        for i, mesh in enumerate(self.meshes):
            self.meshes[i] = sweep(mesh, decimals=decimals)

        # ensure identical points-arrays
        points = self.meshes[0].points
        for i, m in enumerate(self.meshes):
            self.meshes[i].points = self.points = points

    def as_meshio(self, combined=True, **kwargs):
        "Export a (combined) mesh object as :class:`meshio.Mesh`."

        import meshio

        if not combined:
            cells = [
                meshio.CellBlock(cell_type, data) for cell_type, data in self.cells()
            ]

        else:
            cells = {}
            for mesh in self.meshes:
                if mesh.cell_type not in cells.keys():
                    cells[mesh.cell_type] = mesh.cells
                else:
                    cells[mesh.cell_type] = np.vstack(
                        [cells[mesh.cell_type], mesh.cells]
                    )

        return meshio.Mesh(self.points, cells, **kwargs)

    def copy(self):
        "Return a deepcopy of the mesh container."
        return deepcopy(self)

    def plot(self, *args, colors=None, **kwargs):
        """Plot the meshes of the mesh container.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        """

        if colors is None:
            import matplotlib.colors as mcolors
            import pyvista

            colors = [
                pyvista.global_theme.color,
                *list(mcolors.TABLEAU_COLORS.values())[1:],
            ]

        plotter = None
        for mesh, color in zip(self.meshes, colors):
            plotter = mesh.view().plot(
                *args,
                show_undeformed=False,
                color=color,
                plotter=plotter,
                opacity=0.99,
                **kwargs,
            )

        return plotter

    def screenshot(
        self,
        *args,
        filename="mesh.png",
        transparent_background=None,
        scale=None,
        colors=None,
        **kwargs,
    ):
        """Take a screenshot of the meshes of the mesh container.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        if colors is None:
            import matplotlib.colors as mcolors
            import pyvista

            colors = [
                pyvista.global_theme.color,
                *list(mcolors.TABLEAU_COLORS.values())[1:],
            ]

        plotter = None
        for mesh, color in zip(self.meshes, colors):
            plotter = mesh.plot(
                *args,
                off_screen=True,
                color=color,
                plotter=plotter,
                opacity=0.99,
                **kwargs,
            )

        return plotter.screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

    def imshow(self, ax=None, *args, **kwargs):
        """Take a screenshot of the meshes of the mesh container, show the image data in
        a figure and return the ax.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

        ax.imshow(self.screenshot(*args, filename=None, **kwargs))
        ax.set_axis_off()

        return ax

    def __iadd__(self, mesh):
        self.append(mesh)
        return self

    def __getitem__(self, index):
        return self.meshes[index]

    def __repr__(self):
        header = "<felupe mesh container object>"
        points = f"  Number of points: {len(self.points)}"
        cells_header = "  Number of cells:"
        cells = []

        for cells_type, cells_data in self.cells():
            cells.append(f"    {cells_type}: {len(cells_data)}")

        return "\n".join([header, points, cells_header, *cells])

    def __str__(self):
        return self.__repr__()
