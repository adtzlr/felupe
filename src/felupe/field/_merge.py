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

from ..mesh import MeshContainer


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
    Field containers with dual fields are not supported.

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

    """

    if len(fields) < 1:
        raise ValueError("The list of field containers to be merged is empty.")

    for field in fields:
        if not hasattr(field, "is_container"):
            raise TypeError(
                "The given fields are not field containers. Please use a list of "
                "field containers as input for the merge function."
            )

    for field in fields:
        for f in field.fields:
            if "Dual" in type(f).__name__:
                raise TypeError(
                    "Dual fields can't be merged. "
                    "Please use a list of field containers without dual fields as "
                    "input for the merge function."
                )

    regions = [field.region for field in fields]
    meshes = [region.mesh for region in regions]

    container = MeshContainer(meshes, merge=True, decimals=decimals, **kwargs)

    # take the type and the dimension
    # of the first sub-field of the first field container
    Field = fields[0][0].__field__
    dim = fields[0][0].dim

    # create a new top-level (global) vertex field container
    x0 = Field.from_mesh_container(container, dim=dim).as_container(
        mesh_container=container
    )

    # reload regions of field containers in-place
    for field, new_mesh in zip(fields, container.meshes):

        # reload the region of the field container with the new mesh
        field.region.reload(mesh=new_mesh)

        # reload the underlying fields of the current field container
        for f in field.fields:
            f.reload(region=field.region)

        # reload the field container (indices and offsets)
        field.reload()

        # add the top-level field container as attribute x0
        field.x0 = x0

    return x0
