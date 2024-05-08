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

    ..  pyvista-plot::

        >>> import numpy as np
        >>> import felupe as fem
        >>> import pyvista as pv
        >>>
        >>> mesh = fem.Circle(radius=1, n=6, sections=[0, 270])
        >>> x, y = mesh.points.T
        >>> region = fem.RegionQuad(mesh)
        >>> displacement = fem.FieldPlaneStrain(region, dim=2)
        >>>
        >>> boundaries = fem.dof.symmetry(displacement, axes=(True, False), x=0.0)
        >>>
        >>> plotter = pv.Plotter()
        >>> actor = plotter.add_points(
        ...     np.pad(mesh.points[boundaries["symx"].points], ((0, 0), (0, 1))),
        ...     point_size=20,
        ...     color="red",
        ... )
        >>> mesh.plot(plotter=plotter, opacity=0.7).show()

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
    loading direction.

    Parameters
    ----------
    field : felupe.FieldContainer
        FieldContainer on wich the symmetry boundaries are created.
    left : float or None, optional
        The position of the left end face along the given axis (default is None). If
        None, the outermost left position of the mesh-points is taken, i.e.
        ``left=field.region.mesh.points[:, axis].min()``.
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

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> region = fem.RegionHexahedron(fem.Cube(a=(0, 0, 0), b=(2, 3, 1), n=(6, 11, 5)))
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> boundaries = fem.dof.uniaxial(field, axis=2, clamped=True)[0]

    The longitudinal displacement is applied incrementally.

    ..  pyvista-plot::
        :context:

        >>> solid = fem.SolidBodyNearlyIncompressible(fem.NeoHooke(mu=1), field, bulk=5000)
        >>> step = fem.Step(
        ...     items=[solid],
        ...     ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=5)},
        ...     boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> field.plot("Principal Values of Logarithmic Strain").show()

    See Also
    --------
    felupe.Boundary : A collection of prescribed degrees of freedom.
    felupe.dof.partition : Partition degrees of freedom into prescribed and active dof.
    felupe.dof.apply : Apply prescribed values for a list of boundaries.
    felupe.dof.symmetry : Return a dict of boundaries for the symmetry axes.

    """

    f = field[0]

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
    clampes=(False, False),
    sym=True,
):
    """Return a dict of boundaries for biaxial loading between a left (applied or
    symmetry face) and a right (applied) end face along a given pair of axes with
    optional selective symmetries at the origin. Optionally, the applied end faces are
    assumed to be rigid (clamped) in the transversal directions perpendicular to the
    longitudinal loading direction.

    Parameters
    ----------
    field : felupe.FieldContainer
        FieldContainer on wich the symmetry boundaries are created.
    lefts : tuple of float or None, optional
        The position of the left end faces where the longitudinal movement is applied
        along the given axes (default is (None, None)). If an item of the tuple is None,
        the outermost left position of the mesh-points is taken, i.e.
        ``lefts=[field.region.mesh.points[:, axis].min() for axis in axes]``.
    rights : tuple of float or None, optional
        The position of the right end faces where the longitudinal movement is applied
        along the given axes (default is (None, None)). If an item of the tuple is None,
        the outermost right position of the mesh-points is taken, i.e.
        ``rights=[field.region.mesh.points[:, axis].max() for axis in axes]``.
    moves : tuple of float, optional
        The values of the longitudinal displacements applied each one half of the value
        at the left and right end faces (default is (0.2, 0.2)).
    axes : tuple of int, optional
        The pair of longitudinal axes (default is (0, 1)).
    clampes : tuple of bool, optional
        Flags to assume the applied end faces to be rigid, i.e. zero
        displacements in the direction of the transversal axes are enforced (default is
        True).
    sym : bool or tuple of bool, optional
        A flag to invoke all (bool) or individual (tuple) symmetry boundaries at the
        left end face in the directions of the longitudinal axes as well as in the
        direction of the transversal axis.

    Returns
    -------
    dict of felupe.Boundary
        Dict of boundaries for a biaxial loadcase.
    dict of ndarray
        Loadcase-related partitioned prescribed ``dof0`` and active ``dof1`` degrees of
        freedom as well as the external displacement values ``ext0`` for the prescribed
        degrees of freedom.

    Notes
    -----
    ..  warning:: Note that `clampes=(True, True)` is not a valid loadcase for a cube.
        Instead, use a shape where the clamped end faces do not share mesh-points.

    Examples
    --------
    A cross-like planar specimen of a hyperelastic solid is subjected to biaxial
    displacement-controlled tension on rigid end faces.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(a=(0, 0), b=(1, 1), n=(21, 21))
        >>> x, y = mesh.points.T
        >>> points = np.arange(mesh.npoints)[np.logical_or.reduce([x <= 0.6, y <= 0.6])]
        >>> mesh.update(cells=mesh.cells[np.all(np.isin(mesh.cells, points), axis=1)])
        >>>
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        >>>
        >>> boundaries = fem.dof.biaxial(field, clampes=(True, True))[0]

    The longitudinal displacements are applied incrementally.

    ..  pyvista-plot::
        :context:

        >>> solid = fem.SolidBodyNearlyIncompressible(fem.NeoHooke(mu=1), field, bulk=5000)
        >>> step = fem.Step(
        ...     items=[solid],
        ...     ramp={
        ...         boundaries["move-right-0"]: fem.math.linsteps([0, 0.1], num=5),
        ...         boundaries["move-right-1"]: fem.math.linsteps([0, 0.1], num=5),
        ...     },
        ...     boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> field.plot("Principal Values of Logarithmic Strain").show()

    Repeating the above example with ``fem.dof.biaxial(field, clampes=(False, False)``
    results in a different deformation at the end faces.

    ..  pyvista-plot::

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(a=(0, 0), b=(1, 1), n=(21, 21))
        >>> x, y = mesh.points.T
        >>> points = np.arange(mesh.npoints)[np.logical_or.reduce([x <= 0.6, y <= 0.6])]
        >>> mesh.update(cells=mesh.cells[np.all(np.isin(mesh.cells, points), axis=1)])
        >>>
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        >>>
        >>> boundaries = fem.dof.biaxial(field, clampes=(False, False))[0]
        >>>
        >>> solid = fem.SolidBodyNearlyIncompressible(fem.NeoHooke(mu=1), field, bulk=5000)
        >>> step = fem.Step(
        ...     items=[solid],
        ...     ramp={
        ...         boundaries["move-right-0"]: fem.math.linsteps([0, 0.1], num=5),
        ...         boundaries["move-right-1"]: fem.math.linsteps([0, 0.1], num=5),
        ...     },
        ...     boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> field.plot("Principal Values of Logarithmic Strain").show()

    The biaxial load case may also invoke a planar loading, where one of the
    longitudinal axes is fixed with no displacements at the end plates. The clampling
    must at least be deactivated on the fixed longitudinal axis.

    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=5)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> boundaries = fem.dof.biaxial(
        ...     field, clampes=(True, False), moves=(0, 0), sym=False, axes=(0, 1)
        ... )[0]
        >>> solid = fem.SolidBodyNearlyIncompressible(fem.NeoHooke(mu=1), field, bulk=5000)
        >>> step = fem.Step(
        ...     items=[solid],
        ...     ramp={boundaries["move-right-0"]: fem.math.linsteps([0, 0.3], num=5),},
        ...     boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> field.plot("Principal Values of Logarithmic Strain").show()

    See Also
    --------
    felupe.Boundary : A collection of prescribed degrees of freedom.
    felupe.dof.partition : Partition degrees of freedom into prescribed and active dof.
    felupe.dof.apply : Apply prescribed values for a list of boundaries.
    felupe.dof.symmetry : Return a dict of boundaries for the symmetry axes.

    """

    f = field[0]

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

    for i, (left, axis, active, move) in enumerate(zip(lefts, axes, actives, moves)):
        if not sym[axis]:
            if left is None:
                lefts[i] = f.region.mesh.points[:, axis].min()

            fx = fxyz[axis]
            bounds[f"move-left-{axis}"] = Boundary(
                f, skip=active, value=-move, **{fx: lefts[i]}
            )

    for i, (left, right, axis, active, inactive, move, clamped) in enumerate(
        zip(lefts, rights, axes, actives, inactives, moves, clampes)
    ):
        fx = fxyz[axis]
        if clamped:
            bounds[f"right-{axis}"] = Boundary(f, skip=inactive, **{fx: right})

            if not sym[axis]:
                bounds[f"left-{axis}"] = Boundary(f, skip=inactive, **{fx: left})

        bounds[f"move-right-{axis}"] = Boundary(
            f, skip=active, value=move, **{fx: right}
        )

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)


def shear(
    field,
    bottom=None,
    top=None,
    moves=(0.2, 0.0, 0.0),
    axes=(0, 1),
    sym=True,
):
    """Return a dict of boundaries for shear loading with optional combined compression
    between a rigid bottom and a rigid top end face along a given pair of axes. The
    first axis is the direction of shear and the second axis the direction of
    compression. The bottom face remains fixed while the shear is applied at the top
    face. Optionally, a symmetry boundary condition in the thickness direction at
    the origin may be added.

    Parameters
    ----------
    field : felupe.FieldContainer
        FieldContainer on wich the symmetry boundaries are created.
    bottom : float or None, optional
        The position of the bottom end face (default is None). If None, the outermost
        bottom position of the mesh-points is taken, i.e.
        ``bottom=[field.region.mesh.points[:, axis].min() for axis in axes]``.
    top : float or None, optional
        The position of the top end face (default is None). If None, the outermost
        top position of the mesh-points is taken, i.e.
        ``top=[field.region.mesh.points[:, axis].min() for axis in axes]``.
    moves : tuple of float, optional
        The values of the displacements applied on the end faces (default is
        (0.2, 0.0, 0.0)). The first item is the shear displacement applied on the top
        end face. The second and third items refer to the tension/compression
        displacements. The second item is applied on the bottom and the third item on
        the top end face.
    axes : tuple of int, optional
        The pair of axes: the first item is the axis of shear and the second item is the
        axis of compression (default is (0, 1)).
    sym : bool, optional
        A flag to invoke a symmetry boundary in the direction of the thickness axis.

    Returns
    -------
    dict of felupe.Boundary
        Dict of boundaries for a biaxial loadcase.
    dict of ndarray
        Loadcase-related partitioned prescribed ``dof0`` and active ``dof1`` degrees of
        freedom as well as the external displacement values ``ext0`` for the prescribed
        degrees of freedom.

    Examples
    --------
    A rectangular planar specimen of a hyperelastic solid is subjected to a
    displacement-controlled combined shear-compression loading on rigid end faces.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(a=(0, 0), b=(4, 1), n=(41, 11))
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

    The top edge is moved by ``-0.1`` to add a 10% constant compressive loading.

    ..  pyvista-plot::
        :context:

        >>> boundaries = fem.dof.shear(field, moves=(0, 0, -0.1))[0]

    The shear displacement is applied incrementally.

    ..  pyvista-plot::
        :context:

        >>> solid = fem.SolidBodyNearlyIncompressible(fem.NeoHooke(mu=1), field, bulk=5000)
        >>> step = fem.Step(
        ...     items=[solid],
        ...     ramp={boundaries["move"]: fem.math.linsteps([0, 1], num=5)},
        ...     boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> field.plot("Principal Values of Logarithmic Strain").show()

    See Also
    --------
    felupe.Boundary : A collection of prescribed degrees of freedom.
    felupe.dof.partition : Partition degrees of freedom into prescribed and active dof.
    felupe.dof.apply : Apply prescribed values for a list of boundaries.
    felupe.dof.symmetry : Return a dict of boundaries for the symmetry axes.

    """

    f = field[0]

    if bottom is None:
        bottom = f.region.mesh.points[:, axes[1]].min()

    if top is None:
        top = f.region.mesh.points[:, axes[1]].max()

    if sym:
        sym = np.ones(3, dtype=bool)
        sym[axes,] = False

        bounds = symmetry(f, axes=sym)
    else:
        bounds = {}

    fy = ["fx", "fy", "fz"][axes[1]]

    skip_compression = [0, 0, 0]
    skip_compression[axes[1]] = 1

    not_skip_compression = [1, 1, 1]
    not_skip_compression[axes[1]] = 0

    not_skip_thickness = [0, 0, 0]
    not_skip_thickness[axes[1]] = 1
    not_skip_thickness[axes[0]] = 1

    not_skip_shear = [1, 1, 1]
    not_skip_shear[axes[0]] = 0

    bounds["bottom"] = Boundary(f, **{fy: bottom}, skip=skip_compression)
    bounds["top"] = Boundary(f, **{fy: top}, skip=not_skip_thickness)
    bounds["compression_bottom"] = Boundary(
        f, **{fy: bottom}, skip=not_skip_compression, value=moves[1]
    )
    bounds["compression_top"] = Boundary(
        f, **{fy: top}, skip=not_skip_compression, value=moves[2]
    )
    bounds["move"] = Boundary(f, **{fy: top}, skip=not_skip_shear, value=moves[0])

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)
