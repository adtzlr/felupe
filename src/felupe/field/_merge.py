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
    The field containers may hold different numbers of fields, e.g. a displacement
    field only or mixed fields like :class:`~felupe.FieldsMixed`. The fields are
    grouped by their position in the field containers, i.e. the ``k``-th field of the
    top-level field container holds the ``k``-th fields of all field containers which
    have at least ``k + 1`` fields. Hence, the fields must be ordered consistently,
    e.g. displacement, pressure and volume ratio.

    *   The first field and all fields which are defined on the mesh of the first
        field are **continuous**: the points of their meshes are merged.
    *   All other fields, e.g. the dual fields of :class:`~felupe.FieldsMixed`, are
        **local** to their field container (body): their points are stacked, i.e.
        there is no coupling of these fields across the bodies (e.g. pressure jumps
        between materials are allowed).

    Each field of a field container is reloaded with the global numbering of the
    points of the top-level field container, i.e. a field container and the top-level
    field container share the offsets of their fields. The values of the fields are
    linked to (shared with) the top-level field container. The initial values of the
    top-level field container are taken from the given fields (e.g. a volume ratio of
    one for the dual fields of :class:`~felupe.FieldsMixed`).

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

    nfields = max(len(field.fields) for field in fields)

    # classify the fields by their position in the field containers
    continuous = [True]
    for k in range(1, nfields):
        flags = {_is_continuous(f, k) for f in fields if len(f.fields) > k}

        if len(flags) > 1:
            raise TypeError(
                f"The fields at position {k} of the field containers are defined on "
                "the mesh of the first field in some, but not in all field containers. "
                "Please use consistently ordered fields."
            )

        continuous.append(flags.pop())

    for k in range(nfields):
        dims = {f[k].dim for f in fields if len(f.fields) > k}
        if len(dims) > 1:
            raise ValueError(
                f"The fields at position {k} of the field containers have different "
                f"dimensions {sorted(dims)}."
            )

    # the meshes of the body-local fields, before the meshes of the first fields are
    # reloaded (a body-local field may share its region with another field)
    local_meshes = [[f[k].region.mesh for k in range(len(f.fields))] for f in fields]
    original_cells = [f.region.mesh.cells for f in fields]

    # merge the points of the meshes of the first fields
    container = MeshContainer(
        [f.region.mesh for f in fields], merge=True, decimals=decimals, **kwargs
    )

    # the values of the top-level fields
    top_level_values = []

    for k in range(nfields):
        members = [i for i, f in enumerate(fields) if len(f.fields) > k]
        dim = fields[members[0]][k].dim
        dtype = np.result_type(*[fields[i][k].values.dtype for i in members])

        if continuous[k]:
            # scatter the point values of the fields to the merged points
            values = np.zeros((len(container.points), dim), dtype=dtype)

            for i in members:
                new_cells = container.meshes[i].cells.ravel()
                old_cells = original_cells[i].ravel()
                values[new_cells] = fields[i][k].values[old_cells]

        else:
            # stack the point values of the body-local fields
            values = np.concatenate([fields[i][k].values for i in members]).astype(
                dtype
            )

        top_level_values.append(values)

    # reload the regions of the first fields with the merged meshes
    for field, new_mesh in zip(fields, container.meshes):
        field.region.reload(mesh=new_mesh)

    # reload the body-local fields on regions with stacked meshes
    reloaded = {}

    for k in range(nfields):
        if continuous[k]:
            continue

        members = [i for i, f in enumerate(fields) if len(f.fields) > k]
        npoints = sum(local_meshes[i][k].npoints for i in members)
        offset = 0

        for i in members:
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
    for field, new_mesh in zip(fields, container.meshes):
        for k, f in enumerate(field.fields):
            if continuous[k]:
                if f.region.mesh is not new_mesh:
                    f.region.reload(mesh=new_mesh)

                f.reload(region=f.region)

    # create a new top-level (global) vertex field container
    top_level_fields = []

    for k, values in enumerate(top_level_values):
        if continuous[k]:
            # take the type of the k-th field of the first field container with k-th
            # field
            Field = next(f[k] for f in fields if len(f.fields) > k).__field__
            fk = Field.from_mesh_container(
                container, dim=values.shape[1], values=values
            )
        else:
            fk = _vertex_field(values)

        top_level_fields.append(fk)

    x0 = FieldContainer(top_level_fields, mesh_container=container)

    for field in fields:
        # reload the field container (indices and offsets)
        field.reload()

        # share the values of the fields with the top-level field container
        field.link(x0)

        # add the top-level field container as attribute x0
        field.x0 = x0

    return x0
