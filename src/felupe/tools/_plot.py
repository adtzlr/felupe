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

from ..math import displacement, eigvalsh, equivalent_von_mises, tovoigt


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
        notebook=False,
        extract_surface=False,
        nonlinear_subdivision=1,
        smooth_shading=True,
        split_sharp_edges=True,
        edge_color="black",
        line_width=1.0,
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
        theme : str or None, optional
            The theme used for plotting, e.g. "default" or "document" (default is None).
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
        notebook : bool, optional
            When True, the resulting plot is placed inline a jupyter notebook. Assumes a
            jupyter console is active. Automatically enables off_screen (default is
            False).
        extract_surface : bool, optional
            Extract the surface mesh. Required to hide internal edges of quadratic
            cells (default is False). If True and ``show_edges=True``, the feature edges
            of a separated mesh are plotted.
        nonlinear_subdivision : int, optional
            Number of subdivisions to generate a smooth surface based on the mid-edge
            points (default is 1, no subdivision). If greater than 1, the surface of
            the mesh is extracted.
        smooth_shading : bool, optional
            A flag to enable smooth shading (default is True). Only considered if
            number of subdivisions is greater than 1.
        split_sharp_edges : bool, optional
            A flag to split sharp edges (default is True). Use this flag in combination
            with smooth shading. Only considered if number of subdivisions is greater
            than 1.
        edge_color : str, optional
            The color of the edge lines (default is "black").
        line_width : float, optional
            The line-width of the edge lines (default is 1.0).


        Returns
        -------
        plotter : pyvista.Plotter
            A Plotter object with methods ``plot()``, ``screenshot()``, etc.

        See Also
        --------
        pyvista.Plotter : Plotting object to display vtk meshes or numpy arrays.
        """

        import pyvista as pv

        if theme is not None:
            pv.set_plot_theme(theme)

        if plotter is None:
            plotter_kwargs = dict()
            if off_screen:
                plotter_kwargs["off_screen"] = off_screen
            if notebook:
                plotter_kwargs["notebook"] = notebook
            plotter = pv.Plotter(**plotter_kwargs)

        if scalar_bar_args is None:
            scalar_bar_args = {}

        if name is not None:
            if component is not None:
                if name in self.mesh.point_data.keys():
                    data = self.mesh.point_data[name]
                else:
                    data = self.mesh.cell_data[name]

        if name is not None and label is None:
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

                if "Principal Values of " in data_label:
                    data[:] = np.flip(np.sort(data, axis=-1), axis=-1)
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

                if "Equivalent of " in data_label:
                    component_labels_dict[1] = [""]

                component_labels = np.arange(dim)
                if dim in component_labels_dict.keys():
                    component_labels = component_labels_dict[dim]

                component_label = component_labels[component]

            else:
                component_label = "Magnitude"

            label = f"{data_label} {component_label}"

        if show_undeformed:
            plotter.add_mesh(
                self.mesh, show_edges=False, opacity=0.2, line_width=line_width
            )

        mesh = self.mesh
        if "Displacement" in self.mesh.point_data.keys():
            mesh = mesh.warp_by_vector("Displacement", factor=factor)

        surface = mesh
        show_edges_surface = show_edges
        kwargs_with_line_width = {**kwargs}

        if mesh.number_of_cells > 0:
            if extract_surface or nonlinear_subdivision > 1:
                surface = surface.extract_surface(
                    nonlinear_subdivision=nonlinear_subdivision
                )
                show_edges_surface = False
            else:
                kwargs_with_line_width["line_width"] = line_width

        # disable surface-related arguments if the mesh contains no cells
        if mesh.number_of_cells == 0 or nonlinear_subdivision == 1:
            smooth_shading = None
            split_sharp_edges = None

        # don't show edges for the base (surface) mesh to hide internal edges of
        # quadratic / Lagrange cell-types
        plotter.add_mesh(
            mesh=surface,
            scalars=name,
            component=component,
            show_edges=show_edges_surface,
            cmap=cmap,
            scalar_bar_args={
                "title": label,
                "interactive": True,
                "vertical": scalar_bar_vertical,
                **scalar_bar_args,
            },
            smooth_shading=smooth_shading,
            split_sharp_edges=split_sharp_edges,
            **kwargs_with_line_width,
        )

        # extract the feature edges (without cell-internal edges)
        if (
            mesh.number_of_cells > 0
            and show_edges
            and (extract_surface or nonlinear_subdivision > 1)
        ):
            edges = (
                mesh.separate_cells()
                .extract_surface(nonlinear_subdivision=nonlinear_subdivision)
                .extract_feature_edges()
            )
            actor = plotter.add_mesh(edges, color=edge_color, line_width=line_width)
            actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()

        if view == "default":
            if np.allclose(self.mesh.points[:, 2], 0):
                view = "xy"
                plotter.enable_parallel_projection()

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
    """Provide Visualization methods for :class:`felupe.Mesh` with optional given
    dicts of point- and cell-data items.

    Parameters
    ----------
    mesh : felupe.Mesh
        The mesh object.
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

    def __init__(self, mesh, point_data=None, cell_data=None, cell_type=None):
        self.mesh = mesh.as_pyvista(cell_type=cell_type)

        if point_data is None:
            point_data = {}

        if cell_data is None:
            cell_data = {}

        for label, data in point_data.items():
            self.mesh.point_data[label] = data

        for label, data in cell_data.items():
            self.mesh.cell_data[label] = data

        self.mesh.set_active_scalars(None)
        self.mesh.set_active_vectors(None)
        self.mesh.set_active_tensors(None)


class ViewField(ViewMesh):
    """Provide Visualization methods for :class:`felupe.FieldContainer`. The warped
    (deformed) mesh is created from the values of the first field (displacements). By
    default, the "Deformation Gradient" tensor, the "Logarithmic Strain" tensor and the
    "Principal Values of Logarithmic Strain" are evaluated as field-related items of the
    cell-data dict. Optional items of given point- and cell-data overwrite these default
    field-related cell-data items.

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
    project : callable or None, optional
        Callable to project internal cell-data at quadrature-points to mesh-points
        (default is None). Valid callables are :class:`~felupe.project` or
        :class:`~felupe.tools.extrapolate`.

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    See Also
    --------
    felupe.project: Project given values at quadrature-points to mesh-points.

    """

    def __init__(
        self, field, point_data=None, cell_data=None, cell_type=None, project=None
    ):
        point_data_from_field = {}
        cell_data_from_field = {}

        if project is None:
            cell_data_from_field = {
                "Deformation Gradient": field.evaluate.deformation_gradient()
                .mean(-2)
                .T,
                "Logarithmic Strain": field.evaluate.strain(tensor=True, asvoigt=True)
                .mean(-2)
                .T,
                "Principal Values of Logarithmic Strain": field.evaluate.strain(
                    tensor=False
                )
                .mean(-2)
                .T,
            }
        elif callable(project):
            point_data_from_field = {
                "Deformation Gradient": project(
                    field.evaluate.deformation_gradient(), field.region
                ),
                "Logarithmic Strain": project(
                    field.evaluate.strain(tensor=True, asvoigt=True), field.region
                ),
                "Principal Values of Logarithmic Strain": project(
                    field.evaluate.strain(tensor=False), field.region
                ),
            }
        else:
            raise TypeError("The project-argument must be callable or None.")

        point_data_from_field["Displacement"] = displacement(field)

        if point_data is None:
            point_data = {}

        if cell_data is None:
            cell_data = {}

        super().__init__(
            mesh=field.region.mesh,
            point_data={**point_data_from_field, **point_data},
            cell_data={**cell_data_from_field, **cell_data},
            cell_type=cell_type,
        )


class ViewSolid(ViewField):
    """Provide Visualization methods for :class:`felupe.Field` and `felupe.SolidBody`.
    The warped (deformed) mesh is created from the values of the first field
    (displacements). By default, the "Deformation Gradient" tensor, the
    "Logarithmic Strain" tensor and the "Principal Values of Logarithmic Strain" are
    evaluated as field-related items of the cell-data dict. Optional items of given
    point- and cell-data overwrite these default field-related cell-data items.

    Parameters
    ----------
    field : felupe.FieldContainer
        The field-container.
    solid : felupe.SolidBody or felupe.SolidBodyIncompressible or None, optional
        A solid body to evaluate the (Cauchy) stress (default is None).
    stress_type : str, optional
        The type of stress, either "Cauchy" or "Kirchhoff, which is exported (default is
        "Cauchy").
    point_data : dict or None, optional
        Additional point-data dict (default is None).
    cell_data : dict or None, optional
        Additional cell-data dict (default is None).
    cell_type : pyvista.CellType or None, optional
        Cell-type of PyVista (default is None).
    project : callable or None, optional
        Callable to project stress at quadrature-points to mesh-points (default is
        None). Valid callables are :class:`~felupe.project` or
        :class:`~felupe.tools.extrapolate`.

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    See Also
    --------
    felupe.project: Project given values at quadrature-points to mesh-points.
    """

    def __init__(
        self,
        field,
        solid=None,
        stress_type="Cauchy",
        point_data=None,
        cell_data=None,
        cell_type=None,
        project=None,
        **kwargs,
    ):
        if point_data is None:
            point_data = {}

        if cell_data is None:
            cell_data = {}

        point_data_from_solid = {}
        cell_data_from_solid = {}

        if solid is not None:
            stress_from_field = {
                "cauchy": solid.evaluate.cauchy_stress,
                "kirchhoff": solid.evaluate.kirchhoff_stress,
            }
            stress = stress_from_field[stress_type.lower()](field)
            stress_label = f"{stress_type.title()} Stress"

            if project is None:
                cell_data_from_solid[stress_label] = tovoigt(stress.mean(-2)).T
                cell_data_from_solid[f"Principal Values of {stress_label}"] = (
                    eigvalsh(stress).mean(-2).T
                )
                cell_data_from_solid[f"Equivalent of {stress_label}"] = (
                    equivalent_von_mises(stress).mean(-2).T
                )

            elif callable(project):
                point_data_from_solid[stress_label] = project(
                    tovoigt(stress), solid.field.region
                )
                point_data_from_solid[f"Principal Values of {stress_label}"] = project(
                    eigvalsh(stress), solid.field.region
                )
                point_data_from_solid[f"Equivalent of {stress_label}"] = project(
                    equivalent_von_mises(stress), solid.field.region
                )
            else:
                raise TypeError("The project-argument must be callable or None.")

        super().__init__(
            field=field,
            point_data={**point_data_from_solid, **point_data},
            cell_data={**cell_data_from_solid, **cell_data},
            cell_type=cell_type,
            project=project,
        )
