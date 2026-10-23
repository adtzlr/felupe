# SPDX-License-Identifier: GPL-3.0-or-later

from ..math import displacement
from ._mesh import ViewMesh


class ViewField(ViewMesh):
    r"""Provide Visualization methods for :class:`felupe.FieldContainer`. The warped
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
    points : ndarray
        Point coordinates of the mesh.
    cells : ndarray
        Point-connectivity of the cells of the mesh.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type of the mesh.
    point_data : dict
        The point-data dict.
    cell_data : dict
        The cell-data dict.
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`. It is created on first access, i.e.
        PyVista is only required for this attribute and for plotting.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> region = fem.RegionHexahedron(mesh)
        >>> u = np.sqrt(1 + np.arange(81)).reshape(27, 3) / 100
        >>> field = fem.FieldContainer([fem.Field(region, dim=3, values=u)])
        >>>
        >>> view = fem.ViewField(field, project=fem.project)
        >>> view.plot("Principal Values of Logarithmic Strain").show()

    See Also
    --------
    felupe.view.Scene : Base class for plotting a static scene.
    felupe.ViewMesh : Provide Visualization methods for a mesh with optional given
        dicts of point- and cell-data items.
    felupe.ViewSolid : Provide Visualization methods for a field container or a
        solid body.
    felupe.project: Project given values at quadrature-points to mesh-points.
    """

    def __init__(
        self, field, point_data=None, cell_data=None, cell_type=None, project=None
    ):
        point_data_from_field = {}
        cell_data_from_field = {}

        field_is_vector = field[0].dim == field.region.mesh.dim

        if hasattr(field.region, "dhdX") and field_is_vector:
            if project is None:
                cell_data_from_field = {
                    "Deformation Gradient": field.evaluate.deformation_gradient()
                    .mean(-2)
                    .T,
                    "Logarithmic Strain": field.evaluate.strain(
                        tensor=True, asvoigt=True
                    )
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

        point_data_from_field["Field"] = field[0].values

        if field_is_vector:
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
