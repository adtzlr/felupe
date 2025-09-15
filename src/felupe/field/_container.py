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

from ..math import rotate_points
from ..mesh import MeshContainer
from ..region import (
    RegionBiQuadraticQuad,
    RegionHexahedron,
    RegionQuad,
    RegionTriQuadraticHexahedron,
    RegionVertex,
)
from ..view import ViewField
from ._evaluate import EvaluateFieldContainer


class FieldContainer:
    """A container for fields which holds a list or a tuple of :class:`Field`
    instances.

    Parameters
    ----------
    fields : list or tuple of :class:`~felupe.Field`, :class:``~felupe.FieldAxisymmetric`, :class:``~felupe.FieldPlaneStrain` or :class:`~felupe.FieldContainer`
        List with fields. The region is linked to the first field.
    **kwargs : dict, optional
        Extra class attributes for the field container.

    Attributes
    ----------
    evaluate : field.EvaluateFieldContainer
        Methods to evaluate the deformation gradient and strain measures, see
        :class:`~felupe.field.EvaluateFieldContainer` for details on the provided
        methods.

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> region = fem.RegionHexahedron(mesh)
        >>> region_dual = fem.RegionConstantHexahedron(mesh.dual(points_per_cell=1))
        >>> displacement = fem.Field(region, dim=3)
        >>> pressure = fem.Field(region_dual)
        >>> field = fem.FieldContainer([displacement, pressure])
        >>> field
        <felupe FieldContainer object>
          Number of fields: 2
          Dimension of fields:
            Field: 3
            Field: 1

    A new :class:`~felupe.FieldContainer` is also created by one of the logical-and
    combinations of a :class:`~felupe.Field`, :class:`~felupe.FieldAxisymmetric`,
    :class:`~felupe.FieldPlaneStrain` or :class:`~felupe.FieldContainer`.

    ..  pyvista-plot::
        :context:

        >>> displacement & pressure
        <felupe FieldContainer object>
          Number of fields: 2
          Dimension of fields:
            Field: 3
            Field: 1

        >>> volume_ratio = fem.Field(region_dual)
        >>> field & volume_ratio  # displacement & pressure & volume_ratio
        <felupe FieldContainer object>
          Number of fields: 3
          Dimension of fields:
            Field: 3
            Field: 1
            Field: 1

    See Also
    --------
    felupe.Field : Field on points of a :class:`~felupe.Region` with dimension ``dim``
        and initial point ``values``.
    felupe.FieldAxisymmetric : An axisymmetric :class:`~felupe.Field` on points of a
        two dimensional :class:`~felupe.Region` with dimension ``dim`` (default is 2)
        and initial point ``values`` (default is 0).
    felupe.FieldPlaneStrain : A plane strain :class:`~felupe.Field` on points of a
        two dimensional :class:`~felupe.Region` with dimension ``dim`` (default is 2)
        and initial point ``values`` (default is 0).

    """

    def __init__(self, fields, **kwargs):

        # flatten the given list of fields (unpack field containers)
        self.fields = []

        for field in fields:
            if isinstance(field, FieldContainer):
                self.fields.extend(field.fields)
            else:
                self.fields.append(field)

        # set optional user-defined attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.region = self.fields[0].region

        # get sizes of fields and calculate offsets
        self.fieldsizes = [f.indices.dof.size for f in self.fields]
        self.offsets = np.cumsum(self.fieldsizes)[:-1]

        self.evaluate = EvaluateFieldContainer(self)

    def __repr__(self):
        header = "<felupe FieldContainer object>"
        size = f"  Number of fields: {len(self.fields)}"
        fields_header = "  Dimension of fields:"
        fields = [
            f"    {type(field).__name__}: {field.dim}"
            for a, field in enumerate(self.fields)
        ]

        return "\n".join([header, size, fields_header, *fields])

    def extract(
        self, grad=True, sym=False, add_identity=True, dtype=None, out=None, order="C"
    ):
        r"""Generalized extraction method which evaluates either the gradient or the
        field values at the integration points of all cells in the region. Optionally,
        the symmetric part of the gradient is evaluated and/or the identity matrix is
        added to the gradient.

        Parameters
        ----------
        grad : bool or list of bool, optional
            Flag(s) for gradient evaluation(s). A boolean value is appplied on the first
            field only and all other fields are extracted with ``grad=False``. To
            enable or disable gradients per-field, use a list of boolean values instead
            (default is True).
        sym : bool, optional
            Flag for symmetric part if the gradient is evaluated (default is False).
        add_identity : bool, optional
            Flag for the addition of the identity matrix if the gradient is evaluated
            (default is True).
        dtype : data-type or None, optional
            If provided, forces the calculation to use the data type specified. Default
            is None.
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).
        orders : str or list of str, optional
            Controls the memory layout of the outputs. 'C' means it should be C
            contiguous. 'F' means it should be Fortran contiguous, 'A' means it should
            be 'F' if the inputs are all 'F', 'C' otherwise. 'K' means it should be as
            close to the layout as the inputs as is possible, including arbitrarily
            permuted axes. Default is 'C'.

        Returns
        -------
        tuple of ndarray
            (Symmetric) gradient or interpolated field values evaluated at
            the integration points of each cell in the region.

        Notes
        -----
        If the gradient is not requested, the interpolation method returns the field
        values evaluated at the numeric integration  points ``q`` for each cell ``c`` in
        the region (so-called *trailing axes*).

        ..  math::

            u_{i(qc)} = \hat{u}_{ai}\ h_{a(qc)}

        On the other hand, the gradient method returns the gradient of the field values
        w.r.t. the undeformed mesh point coordinates, evaluated at the integration
        points of all cells in the region.

        ..  math::

            \left( \frac{\partial u_i}{\partial X_J} \right)_{(qc)} =
                \hat{u}_{ai} \left( \frac{\partial h_a}{\partial X_J} \right)_{(qc)}

        See Also
        --------
        felupe.Field.interpolate : Interpolate field values located at mesh-points to
            the quadrature points in the region.
        felupe.Field.grad : Gradient as partial derivative of field values w.r.t.
            undeformed coordinates.

        """

        if isinstance(grad, bool):
            grad = (grad,)

        if isinstance(order, str):
            order = (order,)

        if out is None:
            out = [None] * len(self.fields)

        grads = np.pad(grad, (0, len(self.fields) - 1))
        orders = order * len(self.fields)

        return tuple(
            f.extract(g, sym, add_identity=add_identity, dtype=dtype, out=res, order=od)
            for g, f, res, od in zip(grads, self.fields, out, orders)
        )

    def values(self):
        "Return the field values."
        return tuple(f.values for f in self.fields)

    def copy(self):
        "Return a copy of the field."
        return deepcopy(self)

    def link(self, other_field=None):
        "Link value array of other field."
        if other_field is None:
            for u in self.fields[1:]:  # link n-to-1 (all-to-first)
                u.values = self.fields[0].values
        else:
            for field, newfield in zip(self.fields, other_field.fields):  # link 1-to-1
                field.values = newfield.values

    def checkpoint(self):
        """Return a checkpoint of the field container.

        Returns
        -------
        dict
            A dict with the checkpoint array.

        See Also
        --------
        felupe.FieldContainer.restore : Restore a checkpoint of a field container
            inplace.
        """
        return {"field": self.copy()}

    def restore(self, checkpoint):
        """Restore a checkpoint inplace.

        Parameters
        ----------
        checkpoint : dict
            A dict with checkpoint arrays / objects.

        See Also
        --------
        felupe.FieldContainer.checkpoint : Return a checkpoint of the field container.
        """

        for field, newfield in zip(self.fields, checkpoint["field"].fields):
            field.values[:] = newfield.values

    def revolve(self, n=11, phi=180):
        """Return a revolved field container.

        Parameters
        ----------
        n : int, optional
            Number of n-point revolutions (or (n-1) cell revolutions), default is 11.
        phi : float or ndarray, optional
            Revolution angle in degree (default is 180).

        Returns
        -------
        FieldContainer
            The revolved field container.

        Examples
        --------
        First, create an axisymmetric field.

        ..  pyvista-plot::
            :context:
            :force_static:

            >>> import felupe as fem
            >>>
            >>> region = fem.RegionQuad(mesh=fem.Rectangle(n=6))
            >>> field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
            >>> field.plot().show()

        The first field of the field container is now revolved around the x-axis.

        ..  pyvista-plot::
            :context:
            :force_static:

            >>> new_field = field.revolve(n=11, phi=180)
            >>> new_field.plot().show()

        See Also
        --------
        SolidBody.revolve : Return a revolved solid body
        SolidBodyNearlyIncompressible.revolve : Return a revolved solid body

        """

        if len(self.fields) > 1:
            raise ValueError("Revolve is not supported for more than one field.")

        # revolve the mesh around the x-axis and create new region and new field
        new_mesh = self.region.mesh.revolve(n=n, phi=phi, axis=0, expand_dim=True)

        new_region = {
            RegionBiQuadraticQuad: RegionTriQuadraticHexahedron,
            RegionQuad: RegionHexahedron,
            RegionVertex: RegionVertex,
        }[type(self.region)](new_mesh)

        if type(new_region) is RegionTriQuadraticHexahedron:
            n = 2 * n - 1

        if np.isscalar(phi):
            rotation_angles = np.linspace(0, phi, n)
        else:
            rotation_angles = phi
            n = len(rotation_angles)

        if rotation_angles[-1] == 360:
            rotation_angles = rotation_angles[:-1]

        new_values = []
        for angle_deg in rotation_angles:
            new_values.append(
                rotate_points(
                    points=np.pad(self.fields[0].values, ((0, 0), (0, 1))),
                    angle_deg=angle_deg,
                    axis=0,
                )
            )

        dim = new_values[-1].shape[1]
        Field = self.fields[0].__field__

        return FieldContainer(
            [Field(new_region, dim=dim, values=np.array(new_values).reshape(-1, dim))]
        )

    def merge(self, decimals=None):
        """Merge all fields and return a list of field containers as well as the
        top-level field container.

        Parameters
        ----------
        decimals : int or None, optional
            Precision decimals for merging duplicated mesh points. Default is None.

        Returns
        -------
        list of FieldContainer
            A list with field containers to be used in different items (solid bodies).
        FieldContainer
            The top-level field container, to be used as the ``x0``-argument in
            `meth:`~felupe.Job.evaluate and for the creation of boundary conditions.

        Notes
        -----
        ..  note::

            This works only if all regions are template regions, like
            :class:`~felupe.RegionQuad` or :class:`~felupe.RegionHexahedron`, which are
            supported by :class:`~felupe.FieldDual`.

        Examples
        --------
        ..  pyvista-plot::

            >>> import felupe as fem
            >>>
            >>> mesh1 = fem.Rectangle(n=3)
            >>> field1 = fem.FieldAxisymmetric(fem.RegionQuad(mesh1), dim=2)
            >>>
            >>> mesh2 = fem.Rectangle(a=(1, 0), b=(2, 1), n=3)
            >>> field2 = fem.FieldAxisymmetric(fem.RegionQuad(mesh2), dim=2)
            >>>
            >>> fields, x0 = (field1 & field2).merge()
            >>>
            >>> umat = fem.NeoHookeCompressible(mu=1, lmbda=2)
            >>> solid1 = fem.SolidBody(umat, fields[0])
            >>> solid2 = fem.SolidBody(umat, fields[1])
            >>>
            >>> boundaries, loadcase = fem.dof.uniaxial(x0, clamped=True)
            >>>
            >>> step = fem.Step(items=[solid1, solid2], boundaries=boundaries)
            >>> job = fem.Job(steps=[step]).evaluate(x0=x0)

        """

        regions = [field.region for field in self.fields]
        meshes = [region.mesh for region in regions]

        container = MeshContainer(meshes, merge=True, decimals=decimals)

        new_fields = []

        # only take meshes of non-dual fields
        current_mesh = container.meshes[0]

        for field, mesh in zip(self.fields, container.meshes):

            if "Dual" in type(field).__name__:
                RegionOriginal = type(field.__args__[0])
                region = RegionOriginal(current_mesh)
                new_field = type(field)(region, *field.__args__[1:], **field.__kwargs__)

            else:

                # update the current mesh
                current_mesh = mesh

                RegionOriginal = type(field.region)
                region = RegionOriginal(current_mesh)
                new_field = type(field)(region, dim=field.dim, dtype=field.values.dtype)

            new_fields.append(new_field)

        def group_dual_fields(fields):

            # init lists for new fields and subfields per container
            new_fields = []
            subfields = []

            # loop over fields
            for field in fields:

                # check if type of field is not FieldDual (without importing FieldDual)
                if "Dual" not in type(field).__name__:

                    # finalize the subfields to the list of new fields
                    # if it is not empty
                    if len(subfields) > 0:
                        new_fields.append(subfields)

                    # start a new list of subfields
                    subfields = [field]

                # append the dual field to the list of subfields
                else:
                    subfields.append(field)

            # finally add the last subfields to the list of new fields
            if len(subfields) > 0:
                new_fields.append(subfields)

            return new_fields

        new_fields_grouped = group_dual_fields(new_fields)
        Field = self.fields[0].__field__

        vertex_field = Field.from_mesh_container(container).as_container(
            mesh_container=container
        )

        return [FieldContainer(f) for f in new_fields_grouped], vertex_field

    def view(self, point_data=None, cell_data=None, cell_type=None, project=None):
        """View the field with optional given dicts of point- and cell-data items.

        Parameters
        ----------
        point_data : dict or None, optional
            Additional point-data dict (default is None).
        cell_data : dict or None, optional
            Additional cell-data dict (default is None).
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).
        project : callable or None, optional
            Project internal cell-data at quadrature-points to mesh-points (default is
            None).

        Returns
        -------
        felupe.ViewField
            A object which provides visualization methods for
            :class:`felupe.FieldContainer`.

        See Also
        --------
        felupe.ViewField : Visualization methods for :class:`felupe.FieldContainer`.
        felupe.project: Project given values at quadrature-points to mesh-points.
        felupe.topoints: Shift given values at quadrature-points to mesh-points.
        """

        return ViewField(
            self,
            point_data=point_data,
            cell_data=cell_data,
            cell_type=cell_type,
            project=project,
        )

    def plot(self, *args, project=None, **kwargs):
        """Plot the first field of the container.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        felupe.project: Project given values at quadrature-points to mesh-points.
        felupe.topoints: Shift given values at quadrature-points to mesh-points.
        """
        return self.view(project=project).plot(*args, **kwargs)

    def screenshot(
        self,
        *args,
        filename="field.png",
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the first field of the container.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

    def imshow(self, *args, ax=None, dpi=None, **kwargs):
        """Take a screenshot of the first field of the container, show the image data in
        a figure and return the ax.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(dpi=dpi)

        ax.imshow(self.screenshot(*args, filename=None, **kwargs))
        ax.set_axis_off()

        return ax

    def __add__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field += dfield

        return fields

    def __sub__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field -= dfield

        return fields

    def __mul__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field *= dfield

        return fields

    def __truediv__(self, newvalues):
        fields = deepcopy(self)
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)

        for field, dfield in zip(fields, newvalues):
            field /= dfield

        return fields

    def __iadd__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field += dfield
        return self

    def __isub__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field -= dfield
        return self

    def __imul__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field *= dfield
        return self

    def __itruediv__(self, newvalues):
        if len(newvalues) != len(self.fields):
            newvalues = np.split(newvalues, self.offsets)
        for field, dfield in zip(self.fields, newvalues):
            field /= dfield
        return self

    def __getitem__(self, idx):
        "Slice-based access to underlying fields."

        return self.fields[idx]

    def __len__(self):
        "Number of fields inside the container."

        return len(self.fields)

    def __and__(self, field):
        fields = [self]

        if field is not None:
            fields.append(field)

        return FieldContainer(fields)
