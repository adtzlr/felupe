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

import numpy as np

from ..mechanics._job import (
    deformation_gradient,
    displacement,
    log_strain,
    log_strain_principal,
)


class Scene:
    """Base class for plotting a static scene.

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    """

    def plot(
        self,
        name=None,
        component=0,
        label=None,
        factor=1.0,
        show_edges=True,
        show_undeformed=True,
        cmap="turbo",
        view="default",
        theme="default",
        scalar_bar_args=None,
        scalar_bar_vertical=False,
        add_axes=True,
        off_screen=False,
        plotter=None,
        **kwargs,
    ):
        """Plot scalars, selected by name and component.

        Parameters
        ----------
        name: str or None, optional
            Name of array of scalars to plot (default is None).
        component : int, optional
            Component of vector-valued scalars to plot (default is 0).
        label : str or None, optional
            A custom label which is shown in the scalar bar. If no label is given, it is
            created by the name including the component. For vector-valued scalars, the
            component number is replaced by ``"X"``, etc. For 3d-tensors in
            full or reduced vector storage (Voigt-Notation) the component number is
            replaced by ``"XY"``, etc. If ``"Principal Values of"`` is in the name, the
            component number is replaced by ``"\n (Max. Principal)"``, assuming that the
            principal values are sorted in descending order.
        factor : float, optional
            Factor for the scaling of the warped (deformed) mesh (default is 1.0).
        show_edges : bool, optional
            Show the edges of the cells (default is True).
        show_undeformed : bool, optional
            Show the undeformed model (default is True).
        cmap : str, optional
            The color map (default is "turbo").
        view : str or None, optional
            The camera position, e.g. "xy" or "iso" (default is "default"). If not
            specified, this is None for 3d-meshes and "xy" for 2d-meshes.
        theme : str, optional
            The theme used for plotting, e.g. "document" (default is "default").
        scalar_bar_vertical : bool, optional
            A flag to show the interactive scalar bar in vertical orientation on the
            right side (default is True).
        add_axes : bool, optional
            Add the axes, i.e. the coordinate system to the scene (default is True).
        off_screen : bool, optional
            Initialize the plotter off-screen and don't open a window on plotting. For
            screenshots, it is necessary to set ``off_screen=True`` (default is False).
        plotter : pyvista.Plotter or None, optional
            Use a given Plotter instead of creating a new instance (default is None).

        Returns
        -------
        plotter : pyvista.Plotter
            A Plotter object with methods ``plot()``, ``screenshot()``, etc.
        """

        import pyvista as pv

        pv.set_plot_theme(theme)

        if plotter is None:
            plotter = pv.Plotter(off_screen=off_screen)

        if scalar_bar_args is None:
            scalar_bar_args = {}

        if name is not None:

            if label is None:
                data_label = name

            if component is not None:

                if name in self.mesh.point_data.keys():
                    data = self.mesh.point_data[name]
                else:
                    data = self.mesh.cell_data[name]

                dim = 1

                if len(data.shape) == 2:
                    dim = data.shape[1]

                component_labels_dict = {
                    1: [""],
                    2: ["X", "Y"],
                    3: ["X", "Y", "Z"],
                    6: ["XX", "YY", "ZZ", "XY", "YZ", "XZ"],
                    9: [
                        "XX",
                        "XY",
                        "XZ",
                        "YX",
                        "YY",
                        "YZ",
                        "ZX",
                        "ZY",
                        "ZZ",
                    ],
                }

                if "Principal Values of " in name:
                    component_labels_dict[2] = [
                        "(Max. Principal)",
                        "(Min. Principal)",
                    ]
                    component_labels_dict[3] = [
                        "(Max. Principal)",
                        "(Int. Principal)",
                        "(Min. Principal)",
                    ]
                    data_label = data_label[20:]

                component_labels = np.arange(dim)
                if dim in component_labels_dict.keys():
                    component_labels = component_labels_dict[dim]

                component_label = component_labels[component]

            else:
                component_label = "Magnitude"

            label = f"{data_label} {component_label}"

        if show_undeformed:
            show_edges_undeformed = False
            opacity_undeformed = 0.2

            if name is None:
                show_edges_undeformed = show_edges
                opacity_undeformed = None

            plotter.add_mesh(
                self.mesh, show_edges=show_edges_undeformed, opacity=opacity_undeformed
            )

        if name is not None:
            plotter.add_mesh(
                mesh=self.mesh.warp_by_vector("Displacement", factor=factor),
                scalars=name,
                component=component,
                show_edges=show_edges,
                cmap=cmap,
                scalar_bar_args={
                    "title": label,
                    "interactive": True,
                    "vertical": scalar_bar_vertical,
                    **scalar_bar_args,
                },
                **kwargs,
            )

        if view == "default":

            if np.allclose(self.mesh.points[:, 2], 0):
                view = "xy"

            else:
                view = None
                plotter.camera.elevation = -15
                plotter.camera.azimuth = -100

        plotter.camera_position = view
        # pv.set_plot_theme(theme)

        if add_axes:
            plotter.add_axes()

        return plotter


class ViewXdmf(Scene):  # pragma: no cover
    """Provide Visualization methods for a XDMF file generated by
    `:meth:`Job.evaluate(filename="result.xdmf")`. The warped (deformed) mesh is created
    from the values of the point-data "Displacement".

    Parameters
    ----------
    filename : str
        The filename of the XDMF file (including the extension).
    time : float, optional
        The time value at which the data is extracted (default is 0).

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    """

    def __init__(
        self,
        filename,
        time=0,
    ):
        "XDMF Result file reader and plotter."

        self.filename = filename
        self.mesh = self._read(time)

    def _read(self, time):
        "Read the file and obtain the mesh."

        import pyvista as pv

        self.file = pv.XdmfReader(self.filename)
        self.file.set_active_time_value(time)

        mesh = self.file.read()[0]

        return mesh

    def set_active_time_value(self, time):
        "Set new active time value and re-read the mesh."

        self.mesh = self._read(time)


class ViewMesh(Scene):
    """Provide Visualization methods for :class:`felupe.Mesh`.

    Parameters
    ----------
    mesh : felupe.Mesh
        The mesh object.
    cell_type : pyvista.CellType or None, optional
        Cell-type of PyVista (default is None).

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    """

    def __init__(self, mesh, cell_type=None):

        import pyvista as pv

        points = np.pad(mesh.points, ((0, 0), (0, 3 - mesh.points.shape[1])))
        cells = np.pad(
            mesh.cells, ((0, 0), (1, 0)), constant_values=mesh.cells.shape[1]
        )

        if cell_type is None:
            meshio_to_pyvista_cell_types = {
                "line": pv.CellType.LINE,
                "triangle": pv.CellType.TRIANGLE,
                "triangle6": pv.CellType.QUADRATIC_TRIANGLE,
                "tetra": pv.CellType.TETRA,
                "tetra10": pv.CellType.QUADRATIC_TETRA,
                "quad": pv.CellType.QUAD,
                "quad8": pv.CellType.QUADRATIC_QUAD,
                "quad9": pv.CellType.BIQUADRATIC_QUAD,
                "hexahedron": pv.CellType.HEXAHEDRON,
                "hexahedron20": pv.CellType.QUADRATIC_HEXAHEDRON,
                "hexahedron27": pv.CellType.TRIQUADRATIC_HEXAHEDRON,
            }
            cell_type = meshio_to_pyvista_cell_types[mesh.cell_type]

        cell_types = cell_type * np.ones(mesh.ncells, dtype=int)

        self.mesh = pv.UnstructuredGrid(cells, cell_types, points)


class ViewField(ViewMesh):
    """Provide Visualization methods for :class:`felupe.Field`. The warped (deformed)
    mesh is created from the values of the first field (displacements).

    Parameters
    ----------
    field : felupe.FieldContainer
        The field-container.
    point_data : dict or None, optional
        Additional point-data dict (default is None).
    cell_data : dict or None, optional
        Additional cell-data dict (default is None).
    cell_type : pyvista.CellType or None, optional
        Cell-type of PyVista (default is None).

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    """

    def __init__(self, field, point_data=None, cell_data=None, cell_type=None):

        super().__init__(mesh=field.region.mesh, cell_type=cell_type)

        point_data_from_field = {}
        cell_data_from_field = {}

        point_data_from_field["Displacement"] = displacement(field)
        cell_data_from_field["Deformation Gradient"] = deformation_gradient(field)[0]
        cell_data_from_field["Logarithmic Strain"] = log_strain(field)[0]
        cell_data_from_field[
            "Principal Values of Logarithmic Strain"
        ] = log_strain_principal(field)[0]

        if point_data is None:
            point_data = {}

        if cell_data is None:
            cell_data = {}

        pdata = {**point_data_from_field, **point_data}
        cdata = {**cell_data_from_field, **cell_data}

        for label, data in pdata.items():
            self.mesh.point_data[label] = data

        for label, data in cdata.items():
            self.mesh.cell_data[label] = data

        self.mesh.set_active_scalars(None)
        self.mesh.set_active_vectors(None)
        self.mesh.set_active_tensors(None)
