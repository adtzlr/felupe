# SPDX-License-Identifier: GPL-3.0-or-later

from copy import copy

import numpy as np

from ..mesh import Mesh, MeshContainer
from ..region import RegionVertex


def _vertex_field(values):
    """Return a field on a vertex region with one point per row of the given values,
    which only provides the values and the degrees of freedom of a top-level field."""

    from ._base import Field

    npoints, dim = values.shape
    mesh = Mesh(
        points=np.zeros((npoints, 1)),
        cells=np.arange(npoints).reshape(-1, 1),
        cell_type="vertex",
    )

    return Field(RegionVertex(mesh), dim=dim, values=values)


def _is_continuous(container, k):
    """A field is continuous across the bodies if it is defined on the mesh of the first
    field of its container. Otherwise, it is a body-local (e.g. dual) field."""

    return container[k].region.mesh is container[0].region.mesh


def _keys(container):
    """Return a key for each field of a field container, which identifies the field of
    the top-level field container. The first fields are merged. The other fields are
    grouped by their kind (continuous or body-local), their dimension and their
    occurrence within the field container."""

    keys = [("first",)]
    occurrences = {}

    for k in range(1, len(container.fields)):
        group = (_is_continuous(container, k), container[k].dim)
        occurrences[group] = occurrences.get(group, -1) + 1
        keys.append((*group, occurrences[group]))

    return keys


def merge(fields, decimals=None, **kwargs):
    r"""Merge a list of field containers into a single top-level field container and
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
    The field containers may hold different numbers and kinds of fields, e.g. a
    displacement field only, mixed fields like :class:`~felupe.FieldsMixed` or the
    displacement and the tensor-valued field of :class:`~felupe.ThirdMediumContactMixed`.
    The first fields (e.g. the displacements) of all field containers are merged. The
    other fields are grouped by their kind (continuous or body-local, see below), their
    dimension and their order of appearance in a field container. Each group is a field
    of the top-level field container, ordered by the first appearance of the groups.

    *   The first field and all fields which are defined on the mesh of the first
        field are **continuous**: the points of their meshes are merged. A continuous
        field, which is not defined in all field containers (e.g. the tensor-valued
        field of :class:`~felupe.ThirdMediumContactMixed`), only has the merged points
        of the cells of its field containers.
    *   All other fields, e.g. the dual fields of :class:`~felupe.FieldsMixed`, are
        **local** to their field container (body): their points are stacked, i.e.
        there is no coupling of these fields across the bodies (e.g. pressure jumps
        between materials are allowed).

    Each field of a field container is reloaded with the global numbering of the
    points of its field of the top-level field container, i.e. they have the same
    degrees of freedom. If the fields of a field container are not the first fields of
    the top-level field container in the same order, the indices of their fields of the
    top-level field container are stored in the attribute ``slots`` of the field
    container. These are used to link the fields (see
    :meth:`~felupe.FieldContainer.link`) and to assemble the items (see
    :func:`~felupe.tools.embed`). The values of the fields are linked to (shared with)
    the top-level field container. The initial values of the top-level field
    container are taken from the given fields (e.g. a volume ratio of one for the dual
    fields of :class:`~felupe.FieldsMixed`).

    ..  note::
        Items with an internal state of the given fields, e.g.
        :class:`~felupe.SolidBodyNearlyIncompressible`, must be created after the merge.

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

    Mixed field containers are merged in the same way, e.g. a displacement field for
    a linear-elastic body and mixed fields for a nearly-incompressible body.

    ..  pyvista-plot::

        >>> mesh3 = fem.Rectangle(a=(2, 0), b=(3, 1), n=3)
        >>> field3 = fem.FieldsMixed(fem.RegionQuad(mesh3), n=3, axisymmetric=True)
        >>>
        >>> mesh4 = fem.Rectangle(a=(3, 0), b=(4, 1), n=3)
        >>> displacement4 = fem.FieldAxisymmetric(fem.RegionQuad(mesh4), dim=2)
        >>> field4 = fem.FieldContainer([displacement4])
        >>>
        >>> field = fem.field.merge([field3, field4])
        >>>
        >>> umat3 = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
        >>> umat4 = fem.LinearElasticLargeStrain(E=10, nu=0.3)
        >>> solids = [fem.SolidBody(umat3, field3), fem.SolidBody(umat4, field4)]
        >>>
        >>> boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        >>>
        >>> step = fem.Step(items=solids, boundaries=boundaries)
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

    from ._container import FieldContainer

    dims = {field[0].dim for field in fields}
    if len(dims) > 1:
        raise ValueError(
            f"The first fields of the field containers have different dimensions "
            f"{sorted(dims)}."
        )

    # assign the fields of the field containers to the fields of the top-level field
    # container (slots), ordered by their first appearance
    keys = [_keys(field) for field in fields]
    top_level_keys = list(dict.fromkeys(key for k in keys for key in k))
    slots = [[top_level_keys.index(key) for key in k] for k in keys]
    nslots = len(top_level_keys)

    # the (field container, field) indices of each field of the top-level container
    members = [
        [(i, k) for i, s in enumerate(slots) for k, slot in enumerate(s) if slot == j]
        for j in range(nslots)
    ]
    continuous = [True] + [key[0] for key in top_level_keys[1:]]

    # the meshes of the body-local fields, before the meshes of the first fields are
    # reloaded (a body-local field may share its region with another field)
    local_meshes = [[f.region.mesh for f in field.fields] for field in fields]
    original_cells = [field.region.mesh.cells for field in fields]

    # merge the points of the meshes of the first fields
    container = MeshContainer(
        [field.region.mesh for field in fields], merge=True, decimals=decimals, **kwargs
    )

    # a continuous field, which is not defined in all field containers (e.g. the
    # tensor-valued field of a third medium), only has the merged points of the cells of
    # its field containers (compact numbering). The first field uses all merged points.
    compact = [None] * nslots

    for j in range(1, nslots):
        if continuous[j]:
            used = np.unique(
                np.concatenate(
                    [container.meshes[i].cells.ravel() for i, k in members[j]]
                )
            )

            if len(used) < len(container.points):
                index = np.full(len(container.points), -1)
                index[used] = np.arange(len(used))
                compact[j] = (used, index)

    # the values of the top-level fields
    top_level_values = []

    for j in range(nslots):
        i0, k0 = members[j][0]
        dim = fields[i0][k0].dim
        dtype = np.result_type(*[fields[i][k].values.dtype for i, k in members[j]])

        if continuous[j]:
            # scatter the point values of the fields to the merged points
            values = np.zeros((len(container.points), dim), dtype=dtype)

            for i, k in members[j]:
                new_cells = container.meshes[i].cells.ravel()
                old_cells = original_cells[i].ravel()
                values[new_cells] = fields[i][k].values[old_cells]

            if compact[j] is not None:
                used, index = compact[j]
                values = values[used]

        else:
            # stack the point values of the body-local fields
            values = np.concatenate(
                [fields[i][k].values for i, k in members[j]]
            ).astype(dtype)

        top_level_values.append(values)

    # reload the regions of the first fields with the merged meshes
    for field, new_mesh in zip(fields, container.meshes):
        field.region.reload(mesh=new_mesh)

    # reload the body-local fields on regions with stacked meshes
    reloaded = {}

    for j in range(nslots):
        if continuous[j]:
            continue

        npoints = sum(local_meshes[i][k].npoints for i, k in members[j])
        offset = 0

        for i, k in members[j]:
            mesh = local_meshes[i][k]
            region = fields[i][k].region
            key = (id(region), offset, npoints)

            if key not in reloaded:
                # keep the points of the body-local mesh (e.g. for gradients)
                points = np.zeros((npoints, mesh.dim), dtype=mesh.points.dtype)
                points[offset : offset + mesh.npoints] = mesh.points
                new_mesh = Mesh(
                    points=points,
                    cells=mesh.cells + offset,
                    cell_type=mesh.cell_type,
                )
                # don't modify a region which may be shared with other fields
                new_region = copy(region)
                new_region.reload(mesh=new_mesh)
                reloaded[key] = new_region

            fields[i][k].reload(region=reloaded[key])
            offset += mesh.npoints

    # reload the continuous fields (and their own regions) with the merged meshes
    for j in range(nslots):
        if not continuous[j]:
            continue

        for i, k in members[j]:
            f = fields[i][k]
            new_mesh = container.meshes[i]

            if compact[j] is not None:
                used, index = compact[j]
                compact_mesh = Mesh(
                    points=container.points[used],
                    cells=index[new_mesh.cells],
                    cell_type=new_mesh.cell_type,
                )
                # don't modify a region which may be shared with other fields
                region = copy(f.region)
                region.reload(mesh=compact_mesh)
                f.reload(region=region)
                continue

            if f.region.mesh is not new_mesh:
                f.region.reload(mesh=new_mesh)

            f.reload(region=f.region)

    # create a new top-level (global) vertex field container
    top_level_fields = []

    for j, values in enumerate(top_level_values):
        if continuous[j] and compact[j] is None:
            # take the type of the first field of this group
            i0, k0 = members[j][0]
            Field = fields[i0][k0].__field__
            fk = Field.from_mesh_container(
                container, dim=values.shape[1], values=values
            )
        else:
            fk = _vertex_field(values)

        top_level_fields.append(fk)

    x0 = FieldContainer(top_level_fields, mesh_container=container)

    for field, s in zip(fields, slots):
        # reload the field container (indices and offsets)
        field.reload()

        # the indices of the fields of the top-level field container, only if they
        # differ from the first fields of the top-level field container
        field.slots = None if s == list(range(len(s))) else s

        # share the values of the fields with the top-level field container
        field.link(x0)

        # add the top-level field container as attribute x0
        field.x0 = x0

    return x0
