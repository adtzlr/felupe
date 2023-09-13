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


class Boundary:
    """A Boundary as a collection of prescribed degrees of freedom (numbered
    coordinate components of a field at points of a mesh).

    Parameters
    ----------
    field : felupe.Field
        Field on wich the boundary is created.
    name : str, optional (default is "default")
        Name of the boundary.
    fx : float or callable, optional
        Mask-function for x-component of mesh-points which returns `True` at points
        on which the boundary will be applied (default is ``np.isnan``). If a float is
        passed, this is transformed to ``lambda x: np.isclose(x, fx)``.
    fy : float or callable, optional
        Mask-function for y-component of mesh-points which returns `True` at points
        on which the boundary will be applied (default is ``np.isnan``). If a float is
        passed, this is transformed to ``lambda y: np.isclose(y, fy)``.
    fz : float or callable, optional
        Mask-function for z-component of mesh-points which returns `True` at points
        on which the boundary will be applied (default is ``np.isnan``). If a float is
        passed, this is transformed to ``lambda z: np.isclose(z, fz)``.
    value : ndarray or float, optional
        Value(s) of the selected (prescribed) degrees of freedom (default is 0.0).
    skip : tuple of bool or int, optional
        A tuple to define which axes of the selected points should be skipped, i.e.
        not prescribed (default is ``(False, False, False)``).
    mask : ndarray
        Boolean mask for the prescribed degrees of freedom. If a mask is passed, ``fx``,
        ``fy`` and ``fz`` are ignored. However, ``skip`` is still applied on the mask.
    mode : string, optional
        A string which defines the logical operation for the selected points per
        axis (default is `or`).

    Attributes
    ----------
    mask : ndarray
        1d- or 2d-boolean mask array for the prescribed degrees of freedom.
    dof : ndarray
        1d-array of ints which contains the prescribed degrees of freedom.
    points : ndarray
        1d-array of ints which contains the point ids on which one or more degrees of
        freedom are prescribed.
    value : ndarray or float
        Value of the selected (prescribed) degrees of freedom.

    Examples
    --------
    A boundary condition prescribes values for chosen degrees of freedom of a given
    field (**not** a field container). This is demonstrated for a plane-strain
    vector field on a quad-mesh of a circle.

    >>> import felupe as fem

    >>> mesh = fem.Circle(radius=1, n=6)
    >>> x, y = mesh.points.T
    >>> region = fem.RegionQuad(mesh)
    >>> displacement = fem.FieldPlaneStrain(region, dim=2)
    >>> field = fem.FieldContainer([displacement])

    A boundary on the displacement field which prescribes all components of the field
    on the outermost left point of the circle is created. The easiest way is to pass the
    desired value to ``fx``. The same result is obtained if a callable function is
    passed to ``fx``.

    >>> left = fem.Boundary(displacement, fx=x.min())
    >>> left = fem.Boundary(displacement, fx=lambda x: np.isclose(x, x.min()))

    >>> plotter = mesh.plot(off_screen=True)
    >>> plotter.add_points(
    >>>     np.pad(mesh.points[left.points], ((0, 0), (0, 1))),
    >>>     point_size=20,
    >>>     color="red",
    >>> )
    >>> img = plotter.screenshot("boundary_left.png", transparent_background=True)

    ..  image:: images/boundary_left.png

    If ``fx`` and ``fy`` are given, the masks are combined by *logical-or*.

    >>> axes = fem.Boundary(displacement, fx=0, fy=0, mode="or")

    ..  image:: images/boundary_axes.png

    This may be changed to *logical-and* if desired.

    >>> center = fem.Boundary(displacement, fx=0, fy=0, mode="and")

    ..  image:: images/boundary_center.png

    For the most-general case, a user-defined boolean mask for the selection of the
    mesh-points is provided. While the two upper methods are useful to select
    points separated per point-coordinates, providing a mask is more flexible as
    it may involve all three coordinates (or any other quantities of interest).

    >>> mask = np.logical_and(np.isclose(x**2 + y**2, 1), x <= 0)
    >>> surface = fem.Boundary(displacement, mask=mask)

    ..  image:: images/boundary_surface.png

    A boundary condition may be skipped on given axes, i.e. if only the x-components
    of a field should be prescribed on the selected points, then the y-axis must
    be skipped.

    >>> axes_x = fem.Boundary(displacement, fx=0, fy=0, skip=(False, True))

    Values for the prescribed degress of freedom are either applied during creation
    or by the update-method.

    >>> left = fem.Boundary(displacement, fx=x.min(), value=-0.2)
    >>> left.update(-0.3)

    Sometimes it is useful to create a boundary with all axes skipped. This
    boundary has no prescribed degrees of freedom and hence, is without effect.
    However, it may still be used in a characteristic job for the boundary to be
    tracked.

    See Also
    --------
    felupe.CharacteristicCurve : A job with a boundary to be tracked.
    felupe.dof.partition : Partition degrees of freedom into prescribed and active dof.
    felupe.dof.apply : Apply prescribed values for a list of boundaries.

    """

    def __init__(
        self,
        field,
        name="default",
        fx=np.isnan,
        fy=np.isnan,
        fz=np.isnan,
        value=0.0,
        skip=(False, False, False),
        mask=None,
        mode="or",
    ):
        mesh = field.region.mesh
        dof = field.indices.dof

        self.field = field
        self.dim = field.dim  # mesh.dim
        self.name = name
        self.value = value
        self.skip = np.array(skip).astype(int)[: mesh.dim]  # self.dim

        self.mode = mode

        # check if callable
        _fx = fx if callable(fx) else lambda x: np.isclose(x, fx)
        _fy = fy if callable(fy) else lambda y: np.isclose(y, fy)
        _fz = fz if callable(fz) else lambda z: np.isclose(z, fz)

        self.fun = [_fx, _fy, _fz][: mesh.dim]

        if mask is None:
            # apply functions on the points per coordinate
            # fx(x), fy(y), fz(z) and create a mask for each coordinate
            mask = [f(x) for f, x in zip(self.fun, mesh.points.T)]

            # select the logical combination function "or" or "and"
            combine = {"or": np.logical_or, "and": np.logical_and}[self.mode]

            # combine the masks with "logical_or" if dim > 1
            if mesh.dim == 1:
                mask = mask[0]

            elif mesh.dim == 2:
                mask = combine(mask[0], mask[1])

            elif mesh.dim == 3:  # and mesh.points.shape[1] == 3:
                tmp = np.logical_or(mask[0], mask[1])
                mask = combine(tmp, mask[2])

        # tile the mask
        self.mask = np.tile(mask.reshape(-1, 1), self.dim)

        # check if some axes should be skipped
        if True not in skip:
            pass
        else:
            # exclude mask from axes which should be skipped
            self.mask[:, np.where(self.skip)[0]] = False

        self.dof = dof[self.mask]
        self.points = np.arange(mesh.npoints)[mask]

    def update(self, value):
        "Update the value of the boundary in-place."

        self.value = value
