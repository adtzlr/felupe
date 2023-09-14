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

from ._boundary import Boundary
from ._tools import apply, partition


def _get_first_field(field):
    "Get first field of FieldContainer."

    return field.fields[0]


def symmetry(field, axes=(True, True, True), x=0.0, y=0.0, z=0.0, bounds=None):
    """Return a dict of boundaries for the symmetry axes on the x-, y- and
    z-coordinates.

    Parameters
    ----------
    field : felupe.Field
        Field on wich the symmetry boundaries are created.
    axes : tuple of bool or int
        Flags to invoke symmetries on the x-, y- and z-axis.
    x : float, optional
        Center of the x-symmetry (default is 0.0).
    y : float, optional
        Center of the y-symmetry (default is 0.0).
    z : float, optional
        Center of the z-symmetry (default is 0.0).
    bounds : dict of felupe.Boundary, optional
        Extend a given dict of boundaries by the symmetry boundaries (default is None).

    Returns
    -------
    dict of felupe.Boundary
        New or extended dict of boundaries including symmetry boundaries.

    Notes
    -----
    The symmetry boundaries are labeled as ``"symx"``, ``"symy"`` and ``"symz"``.

    +---------------+-------------------------+--------------------------+
    | Symmetry Axis | Prescribed (Fixed) Axes |      Skip-Argument       |
    +===============+=========================+==========================+
    |       x       |           y, z          | ``(True, False, False)`` |
    +---------------+-------------------------+--------------------------+
    |       y       |           x, z          | ``(False, True, False)`` |
    +---------------+-------------------------+--------------------------+
    |       z       |           x, y          | ``(False, False, True)`` |
    +---------------+-------------------------+--------------------------+

    Examples
    --------
    The x-symmetry boundary for a symmetry on the x-axis contains all points at the
    given x-coordinate. The degrees of freedom are prescribed except for the symmetry
    x-axis.

    >>> import numpy as np
    >>> import felupe as fem

    >>> mesh = fem.Circle(radius=1, n=6, sections=[0, 270])
    >>> x, y = mesh.points.T
    >>> region = fem.RegionQuad(mesh)
    >>> displacement = fem.FieldPlaneStrain(region, dim=2)

    >>> boundaries = fem.dof.symmetry(displacement, axes=(True, False), x=0.0)

    >>> plotter = mesh.plot(off_screen=True)
    >>> plotter.add_points(
    >>>     np.pad(mesh.points[boundaries["symx"].points], ((0, 0), (0, 1))),
    >>>     point_size=20,
    >>>     color="red",
    >>> )
    >>> img = plotter.screenshot("boundary_symx.png", transparent_background=True)

    ..  image:: images/boundary_symx.png

    See Also
    --------
    felupe.Boundary : A collection of prescribed degrees of freedom.
    """

    # convert axes to array and slice by mesh dimension
    enforce = np.array(axes).astype(bool)[: field.dim]

    # invert boolean identity matrix and use its rows
    # for the skip argument (a symmetry condition on
    # axis "z" fixes all displacements u_z=0 but keeps
    # in-plane displacements active)
    skipax = ~np.eye(3).astype(bool)
    kwarglist = [
        {"fx": x, "skip": skipax[0][: field.dim]},
        {"fy": y, "skip": skipax[1][: field.dim]},
        {"fz": z, "skip": skipax[2][: field.dim]},
    ]

    if bounds is None:
        bounds = {}
    labels = ["symx", "symy", "symz"]

    # loop over symmetry conditions and add them to a new dict
    for a, (symaxis, kwargs) in enumerate(zip(enforce, kwarglist[: field.dim])):
        if symaxis:
            bounds[labels[a]] = Boundary(field, **kwargs)

    return bounds


def uniaxial(field, left=None, right=None, move=0.2, axis=0, clamped=False, sym=True):
    """Return a dict of boundaries for uniaxial loading between a left (fixed or
    symmetry face) and a right (applied) end face along a given axis with optional
    selective symmetries at the origin. Optionally, the right end face is assumed to be
    rigid (clamped) in the transversal directions perpendicular to the longitudinal
    direction.

    Parameters
    ----------
    field : felupe.FieldContainer
        FieldContainer on wich the symmetry boundaries are created.
    right : float or None, optional
        The position of the right end face where the longitudinal movement is applied
        along the given axis (default is None). If None, the outermost right position
        of the mesh-points is taken, i.e.
        ``right=field.region.mesh.points[:, axis].max()``.
    move : float, optional
        The value of the longitudinal displacement applied at the right end face
        (default is 0.2).
    axis : int, optional
        The longitudinal axis (default is 0).
    clamped : bool, optional
        A flag to assume the right end face to be rigid, i.e. zero
        displacements in the direction of the transversal axes are enforced (default is
        True).
    left : float or None, optional
        The position of the left end face along the given axis (default is None). If
        None, the outermost left position of the mesh-points is taken, i.e.
        ``left=field.region.mesh.points[:, axis].min()``.
    sym : bool or tuple of bool, optional
        A flag to invoke all (bool) or individual (tuple) symmetry boundaries at the
        left end face in the direction of the longitudinal axis as well as in the
        directions of the transversal axes.

    Returns
    -------
    dict of felupe.Boundary
        Dict of boundaries for a uniaxial loadcase.
    dict of ndarray
        Loadcase-related partitioned prescribed ``dof0`` and active ``dof1`` degrees of
        freedom as well as the external displacement values ``ext0`` for the prescribed
        degrees of freedom.

    Examples
    --------
    A quarter of a solid hyperelastic cube is subjected to uniaxial displacement-
    controlled compression on a rigid end face.

    >>> import felupe as fem

    >>> region = fem.RegionHexahedron(fem.Cube(a=(0, 0, 0), b=(2, 3, 1), n=(11, 16, 6)))
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])

    >>> boundaries = fem.dof.uniaxial(field, axis=2, clamped=True)[0]

    The longitudinal displacement is applied incrementally.

    >>> solid = fem.SolidBodyNearlyIncompressible(fem.NeoHooke(mu=1), field, bulk=5000)
    >>> step = fem.Step(
    >>>     items=[solid],
    >>>     ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=5)},
    >>>     boundaries=boundaries
    >>> )

    >>> fem.Job(steps=[step]).evaluate()
    >>> img = field.screenshot("Principal Values of Logarithmic Strain")

    ..  image:: images/loadcase_ux.png

    See Also
    --------
    felupe.Boundary : A collection of prescribed degrees of freedom.
    felupe.dof.partition : Partition degrees of freedom into prescribed and active dof.
    felupe.dof.apply : Apply prescribed values for a list of boundaries.
    felupe.dof.symmetry : Return a dict of boundaries for the symmetry axes.

    """

    f = _get_first_field(field)

    fx = ["fx", "fy", "fz"][axis]

    mask = np.ones(3, dtype=bool)
    mask[axis] = False

    active = tuple(mask.astype(int))
    inactive = tuple((~mask).astype(int))

    if not hasattr(sym, "__len__"):
        sym = (sym, sym, sym)

    if right is None:
        right = f.region.mesh.points[:, axis].max()

    bounds = symmetry(f, axes=sym)

    if not sym[axis]:
        if left is None:
            left = f.region.mesh.points[:, axis].min()

        bounds["left-x"] = Boundary(f, skip=active, **{fx: left})

    if clamped:
        bounds["right"] = Boundary(f, skip=inactive, **{fx: right})

        if not sym[axis]:
            bounds["left-yz"] = Boundary(f, skip=inactive, **{fx: left})

    bounds["move"] = Boundary(f, skip=active, value=move, **{fx: right})

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)


def biaxial(
    field,
    lefts=(None, None),
    rights=(None, None),
    moves=(0.2, 0.2),
    axes=(0, 1),
    clamped=False,
    sym=True,
):
    """Define boundaries for biaxial loading on a quarter model (x > 0, y > 0,
    z > 0) with symmetries at x=0, y=0 and z=0.

    Note that `clamped=True` is not a valid loadcase for a cube. Use a cross-
    like shape instead where the clamped faces at fx=1 and fy=1 do not share
    mesh-points."""

    f = _get_first_field(field)

    fxyz = ["fx", "fy", "fz"]

    lefts = np.asarray(lefts)
    rights = np.asarray(rights)

    masks = [np.ones(3, dtype=bool), np.ones(3, dtype=bool)]
    for i, axis in enumerate(axes):
        masks[i][axis] = False

    actives = [tuple(mask.astype(int)) for mask in masks]
    inactives = [tuple((~mask).astype(int)) for mask in masks]

    if not hasattr(sym, "__len__"):
        sym = (sym, sym, sym)

    for i, (right, axis) in enumerate(zip(rights, axes)):
        if right is None:
            rights[i] = f.region.mesh.points[:, axis].max()

    bounds = symmetry(f, axes=sym)

    for i, (left, axis, active) in enumerate(zip(lefts, axes, actives)):
        if not sym[axis]:
            if left is None:
                left = f.region.mesh.points[:, axis].min()

            fx = fxyz[axis]
            bounds[f"left-{axis}"] = Boundary(f, skip=active, **{fx: left})

    for i, (right, axis, active, inactive, move) in enumerate(
        zip(rights, axes, actives, inactives, moves)
    ):
        fx = fxyz[axis]
        if clamped:
            bounds[f"right-{axis}"] = Boundary(f, skip=inactive, **{fx: right})

            if not sym[axis]:
                bounds["left-{axis}-z"] = Boundary(f, skip=inactive, **{fx: left})

        bounds[f"move-{axis}"] = Boundary(f, skip=active, value=move, **{fx: right})

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)


def planar(field, right=None, move=0.2, clamped=False):
    """Define boundaries for biaxial loading on a quarter model (x > 0, y > 0,
    z > 0) with symmetries at x=0, y=0 and z=0."""

    f = _get_first_field(field)

    if right is None:
        right = f.region.mesh.points[:, 0].max()

    bounds = symmetry(f)

    if clamped:
        bounds["right"] = Boundary(f, fx=right, skip=(1, 0, 0))

    bounds["move"] = Boundary(f, fx=right, skip=(0, 1, 1), value=move)
    bounds["fix-y"] = Boundary(f, fy=right, skip=(1, 0, 1))

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)


def shear(
    field,
    bottom=None,
    top=None,
    move=0.2,
    axis_shear=0,
    axis_compression=1,
    compression=(0, 0),
    sym=True,
):
    """Define boundaries for shear loading between two clamped plates. The
    bottom plate remains fixed while the shear is applied at the top plate."""

    f = _get_first_field(field)

    if bottom is None:
        bottom = f.region.mesh.points[:, axis_compression].min()

    if top is None:
        top = f.region.mesh.points[:, axis_compression].max()

    if sym:
        axes = [True, True, True]
        axes[axis_shear] = False
        axes[axis_compression] = False

        bounds = symmetry(f, axes=axes)
    else:
        bounds = {}

    fy = ["fx", "fy", "fz"][axis_compression]

    skip_compression = [0, 0, 0]
    skip_compression[axis_compression] = 1

    not_skip_compression = [1, 1, 1]
    not_skip_compression[axis_compression] = 0

    not_skip_thickness = [0, 0, 0]
    not_skip_thickness[axis_compression] = 1
    not_skip_thickness[axis_shear] = 1

    not_skip_shear = [1, 1, 1]
    not_skip_shear[axis_shear] = 0

    bounds["bottom"] = Boundary(f, **{fy: bottom}, skip=skip_compression)
    bounds["top"] = Boundary(f, **{fy: top}, skip=not_skip_thickness)
    bounds["compression_bottom"] = Boundary(
        f, **{fy: bottom}, skip=not_skip_compression, value=compression[0]
    )
    bounds["compression_top"] = Boundary(
        f, **{fy: top}, skip=not_skip_compression, value=-compression[1]
    )
    bounds["move"] = Boundary(f, **{fy: top}, skip=not_skip_shear, value=move)

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)
