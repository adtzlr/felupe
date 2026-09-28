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

from ..mesh import Mesh, MeshContainer
from ..region import RegionVertex


def _is_shared(mesh, other_mesh):
    """Return True if the cells of ``mesh`` are the leading columns of the cells of
    ``other_mesh``, a constant offset is allowed.
    """

    ncells, ppc = mesh.cells.shape

    if ncells != len(other_mesh.cells) or ppc > other_mesh.cells.shape[1]:
        return False

    difference = mesh.cells - other_mesh.cells[:, :ppc]
    offset = difference[0, 0]

    return (difference == offset).all()


def _is_disconnected(mesh):
    "Return True if every used point of the mesh belongs to exactly one cell."
    counts = np.unique(mesh.cells, return_counts=True)[1]
    return counts.max() == 1


def _map_values(new_values, new_cells, old_values, old_cells):
    "Map values from old to new point-numbering via the cell-connectivities."
    new_values[new_cells.ravel()] = old_values[old_cells.ravel()]


def _merge_primary(field_containers, decimals, **kwargs):
    "Merge the meshes of the first (primary) fields of the field containers."

    meshes = [c.fields[0].region.mesh for c in field_containers]
    mesh_container = MeshContainer(meshes, merge=True, decimals=decimals, **kwargs)

    npoints = len(mesh_container.points)
    dim = field_containers[0].fields[0].dim
    values = np.zeros((npoints, dim), dtype=field_containers[0].fields[0].values.dtype)

    for c, mesh in zip(field_containers, mesh_container.meshes):
        old = c.fields[0]
        _map_values(values, mesh.cells, old.values, old.region.mesh.cells)

    return mesh_container, [m.cells for m in mesh_container.meshes], values


def _merge_secondary(field_containers, k, primary_cells, primary_points, decimals, **kw):
    """Merge the k-th (secondary) fields of the field containers. The global point
    numbering is compacted to the used points and ordered by shared, connected and
    disconnected points."""

    fields = [c.fields[k] for c in field_containers]
    meshes = [f.region.mesh for f in fields]

    modes = []
    for c, mesh in zip(field_containers, meshes):
        if _is_shared(mesh, c.fields[0].region.mesh):
            modes.append("shared")
        elif _is_disconnected(mesh):
            modes.append("disconnected")
        else:
            modes.append("connected")

    # global (not yet compacted) cells, with id-ranges:
    # [0, n0) shared, [n0, n0 + n1) connected, [n0 + n1, ...) disconnected
    n0 = len(primary_points)
    global_cells = [None] * len(fields)

    # shared: take the leading columns of the merged primary cells
    for i, mode in enumerate(modes):
        if mode == "shared":
            global_cells[i] = primary_cells[i][:, : meshes[i].cells.shape[1]]

    # connected: merge duplicate points by coordinates
    connected = [i for i, mode in enumerate(modes) if mode == "connected"]
    connected_points = np.zeros((0, primary_points.shape[1]))

    if connected:
        compacted = []
        for i in connected:
            used, inverse = np.unique(meshes[i].cells, return_inverse=True)
            points = meshes[i].points[used]

            if len(used) > 1 and np.all(np.ptp(points, axis=0) == 0):
                raise ValueError(
                    f"The mesh of field {k} of field container {i} has no point "
                    "coordinates (all points are equal) and is neither disconnected "
                    "nor shares the topology of the first field. Duplicate points "
                    "can't be merged. Create the dual field with calc_points=True."
                )

            cells = inverse.reshape(meshes[i].cells.shape)
            compacted.append(Mesh(points, cells, meshes[i].cell_type))

        container = MeshContainer(compacted, merge=True, decimals=decimals, **kw)
        connected_points = container.points

        for i, mesh in zip(connected, container.meshes):
            global_cells[i] = n0 + mesh.cells

    # disconnected: stack the used points (without merging)
    offset = n0 + len(connected_points)
    disconnected_points = []

    for i, mode in enumerate(modes):
        if mode == "disconnected":
            used, inverse = np.unique(meshes[i].cells, return_inverse=True)
            global_cells[i] = offset + inverse.reshape(meshes[i].cells.shape)
            disconnected_points.append(meshes[i].points[used])
            offset += len(used)

    # all global point coordinates (before compaction)
    all_points = np.vstack(
        [
            primary_points,
            connected_points,
            *disconnected_points,
        ]
    )

    # compact the global point numbering to the used points
    used, inverse = np.unique(
        np.concatenate([c.ravel() for c in global_cells]), return_inverse=True
    )
    points = all_points[used]

    new_cells = []
    start = 0
    for cells in global_cells:
        new_cells.append(inverse[start : start + cells.size].reshape(cells.shape))
        start += cells.size

    # map the values of the fields to the new point numbering
    dim = fields[0].dim
    values = np.zeros((len(points), dim), dtype=fields[0].values.dtype)

    for f, cells in zip(fields, new_cells):
        _map_values(values, cells, f.values, f.region.mesh.cells)

    return points, new_cells, values


def merge(fields, decimals=None, **kwargs):
    """Merge a list of field containers into a single top-level field container and
    modify the field containers and the underlying fields in-place.

    Parameters
    ----------
    fields : list of FieldContainer
        The list of field containers to be merged.
    decimals : int or None, optional
        Precision decimals for merging duplicated mesh points. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments for :class:`~felupe.MeshContainer`.

    Returns
    -------
    FieldContainer
        The top-level field container, to be used as the ``x0``-argument in
        :meth:`~felupe.Job.evaluate` and for the creation of boundary conditions. The
        given field containers are modified & reloaded in-place, along with a new
        attribute ``x0`` that points to this top-level field container.

    Notes
    -----
    All field containers must have the same number of fields and the fields at the same
    position must have the same dimension. The first fields are merged on duplicated
    mesh points. The additional (e.g. dual) fields are merged per position, depending on
    their meshes:

    * a mesh which shares the topology of the first field (e.g. continuous pressure for
      Taylor-Hood or MINI elements) is merged along with the first field,
    * a disconnected mesh (e.g. a cell-wise constant pressure) is stacked without
      merging any points and
    * any other mesh is merged on duplicated point coordinates.

    For each additional field, only the used points of the meshes are kept. The field
    values are mapped to the new point numbering and the fields of the given field
    containers are linked to the fields of the top-level field container.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh1 = fem.Rectangle(n=3)
        >>> displacement1 = fem.FieldAxisymmetric(fem.RegionQuad(mesh1), dim=2)
        >>> field1 = fem.FieldContainer([displacement1])
        >>>
        >>> mesh2 = fem.Rectangle(a=(1, 0), b=(2, 1), n=3)
        >>> displacement2 = fem.FieldAxisymmetric(fem.RegionQuad(mesh2), dim=2)
        >>> field2 = fem.FieldContainer([displacement2])
        >>>
        >>> field = fem.field.merge([field1, field2])
        >>>
        >>> umat = fem.NeoHookeCompressible(mu=1, lmbda=2)
        >>> solid1 = fem.SolidBody(umat, field1)
        >>> solid2 = fem.SolidBody(umat, field2)
        >>>
        >>> boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        >>>
        >>> step = fem.Step(items=[solid1, solid2], boundaries=boundaries)
        >>> job = fem.Job(steps=[step]).evaluate()

    Mixed-field containers are merged in the same way.

    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh1 = fem.Rectangle(n=3)
        >>> field1 = fem.FieldsMixed(fem.RegionQuad(mesh1), n=3, planestrain=True)
        >>>
        >>> mesh2 = fem.Rectangle(a=(1, 0), b=(2, 1), n=3)
        >>> field2 = fem.FieldsMixed(fem.RegionQuad(mesh2), n=3, planestrain=True)
        >>>
        >>> field = fem.field.merge([field1, field2])
        >>>
        >>> umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
        >>> solid1 = fem.SolidBody(umat, field1)
        >>> solid2 = fem.SolidBody(umat, field2)
        >>>
        >>> boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        >>>
        >>> step = fem.Step(items=[solid1, solid2], boundaries=boundaries)
        >>> job = fem.Job(steps=[step]).evaluate()

    """

    if len(fields) < 1:
        raise ValueError("The list of field containers to be merged is empty.")

    for field in fields:
        if not hasattr(field, "is_container"):
            raise TypeError(
                "The given fields are not field containers. Please use a list of "
                "field containers as input for the merge function."
            )

    nfields = len(fields[0].fields)
    for field in fields:
        if len(field.fields) != nfields:
            raise TypeError(
                "All field containers must have the same number of fields. Got "
                f"{[len(field.fields) for field in fields]}."
            )

    for k in range(nfields):
        dims = [field.fields[k].dim for field in fields]
        if len(set(dims)) > 1:
            raise ValueError(
                f"The fields at position {k} must have the same dimension. Got {dims}."
            )

    # merge the first fields
    container, primary_cells, primary_values = _merge_primary(
        fields, decimals=decimals, **kwargs
    )

    # merge the additional fields (all data is evaluated before any reload)
    secondary = {}
    for k in range(1, nfields):
        # a field which shares the region of the first field is merged with it
        shares_region = [f.fields[k].region is f.fields[0].region for f in fields]

        if all(shares_region):
            continue

        if any(shares_region):
            raise TypeError(
                f"The fields at position {k} must either all or none share the region "
                "with the first field of their field container."
            )

        secondary[k] = _merge_secondary(
            fields, k, primary_cells, container.points, decimals=decimals, **kwargs
        )

    # create a new top-level (global) vertex field container
    Field = fields[0][0].__field__
    x0_fields = [
        Field.from_mesh_container(
            container, dim=fields[0][0].dim, values=primary_values
        )
    ]

    for k in range(1, nfields):
        if k in secondary:
            points, cells, values = secondary[k]
        else:
            points, cells, values = (
                container.points,
                primary_cells,
                np.zeros((len(container.points), fields[0][k].dim)),
            )
            for field, c in zip(fields, cells):
                _map_values(values, c, field[k].values, field[k].region.mesh.cells)

        used = np.unique(np.concatenate([c.ravel() for c in cells]))
        vertex_mesh = Mesh(points, used.reshape(-1, 1), cell_type="vertex")
        x0_fields.append(
            fields[0][k].__field__(
                RegionVertex(vertex_mesh), dim=fields[0][k].dim, values=values
            )
        )

    x0 = x0_fields[0].as_container(mesh_container=container)
    x0.reload(x0_fields)

    # reload regions of field containers in-place
    for i, (field, new_mesh) in enumerate(zip(fields, container.meshes)):
        primary_region = field.fields[0].region

        # reload the region of the first field with the new mesh
        primary_region.reload(mesh=new_mesh)

        for k, f in enumerate(field.fields):
            if k in secondary and f.region is not primary_region:
                points, cells, values = secondary[k]
                mesh = Mesh(points, cells[i], cell_type=f.region.mesh.cell_type)
                f.region.reload(mesh=mesh)

            # reload the underlying field with the (reloaded) region
            f.reload(region=f.region)

        # reload the field container (indices and offsets)
        field.reload()

        # link the field values to the values of the top-level field container
        field.link(x0)

        # add the top-level field container as attribute x0
        field.x0 = x0

    return x0
