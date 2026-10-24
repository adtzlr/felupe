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

    *   The first fields (e.g. the displacements) of all field containers are
        **merged**, i.e. the points of their meshes are merged. They must be of the
        same type (e.g. :class:`~felupe.FieldAxisymmetric`) and of the same dimension.
        The top-level field container starts with the merged first field.
    *   All other fields belong to their field container (body), e.g. the dual fields of
        :class:`~felupe.FieldsMixed` or the tensor-valued field of
        :class:`~felupe.ThirdMediumContactMixed`. They are **appended** to the
        top-level field container, i.e. they are not coupled across the bodies (e.g.
        pressure jumps between materials are allowed). Their regions and meshes are not
        modified. A field which should be continuous across the bodies (other than the
        first field) must be defined in a single field container.

    The indices of the fields of a field container in the top-level field container
    are stored in the attribute ``slots`` of the field container, if they differ from
    the first fields of the top-level field container. These are used to link the
    fields (see :meth:`~felupe.FieldContainer.link`) and to assemble the items (see
    :func:`~felupe.tools.embed`). The values of the fields are linked to (shared with)
    the top-level field container, i.e. the initial values are taken from the given
    fields (e.g. a volume ratio of one for the dual fields of
    :class:`~felupe.FieldsMixed`).

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

    types = {type(field[0]).__name__ for field in fields}
    if len(types) > 1:
        raise TypeError(
            f"The first fields of the field containers have different types "
            f"{sorted(types)}, e.g. axisymmetric and plane strain fields can't be "
            "merged."
        )

    dims = {field[0].dim for field in fields}
    if len(dims) > 1:
        raise ValueError(
            f"The first fields of the field containers have different dimensions "
            f"{sorted(dims)}."
        )

    # a field which shares the region with the first field keeps the mesh of its body
    for field in fields:
        for f in field.fields[1:]:
            if f.region is field.region:
                f.reload(region=copy(field.region))

    # merge the points of the meshes of the first fields
    cells = [field.region.mesh.cells for field in fields]
    container = MeshContainer(
        [field.region.mesh for field in fields], merge=True, decimals=decimals, **kwargs
    )

    # take the values of the first fields at the merged points
    dtype = np.result_type(*[field[0].values.dtype for field in fields])
    values = np.zeros((len(container.points), fields[0][0].dim), dtype=dtype)

    for field, new_mesh, old_cells in zip(fields, container.meshes, cells):
        values[new_mesh.cells.ravel()] = field[0].values[old_cells.ravel()]

        # reload the region and the first field with the merged mesh
        field.region.reload(mesh=new_mesh)
        field[0].reload(region=field.region)

    # the top-level field container: the merged first field, followed by all other
    # fields of all field containers
    top_level_fields = [
        fields[0][0].__field__.from_mesh_container(
            container, dim=values.shape[1], values=values
        )
    ]

    for field in fields:
        slots = [0]

        for f in field.fields[1:]:
            slots.append(len(top_level_fields))
            top_level_fields.append(_vertex_field(f.values))

        # reload the field container (indices and offsets)
        field.reload()

        # the indices of the fields in the top-level field container, only if they
        # differ from the first fields of the top-level field container
        field.slots = None if slots == list(range(len(slots))) else slots

    x0 = FieldContainer(top_level_fields, mesh_container=container)

    for field in fields:
        # share the values of the fields with the top-level field container
        field.link(x0)

        # add the top-level field container as attribute x0
        field.x0 = x0

    return x0
