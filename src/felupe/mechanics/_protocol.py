# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Protocol, runtime_checkable

from ..field import FieldContainer
from ._helpers import Assemble, Results


@runtime_checkable
class Item(Protocol):
    r"""The interface of an item, which is used in a :class:`~felupe.Step`, in a
    :class:`~felupe.Job` and in :func:`~felupe.newtonraphson`.

    An item contributes a sparse vector and a sparse matrix to the system of equations
    of the Newton-Raphson method, e.g. a solid body, a load or a constraint. Every
    object with the attributes ``field``, ``assemble`` and ``results`` is an item, i.e.
    an item doesn't have to be derived from this class. The attributes are checked by
    ``isinstance(obj, Item)``.

    Attributes
    ----------
    field : FieldContainer
        The field container of the item. Before each evaluation of the vector, the
        fields of the item are linked to the unknowns of the Newton-Raphson method by
        :meth:`FieldContainer.link() <felupe.FieldContainer.link>`.
    assemble : Assemble
        The methods to assemble the vector and the matrix of the item, see
        :class:`~felupe.mechanics.Assemble`. Both may be smaller than the system of
        equations, their shapes are increased to the shape of the system.

        * ``assemble.vector(field, parallel=False)`` returns the sparse vector (the
          residuals) for the given (linked) field container.
        * ``assemble.matrix(parallel=False)`` returns the sparse matrix (the
          linearization of the vector). In each iteration, the matrix is evaluated
          after the vector for the same field container, i.e. the results of the
          evaluation of the vector may be re-used.
        * ``assemble.multiplier`` is None or a scale factor for the vector and the
          matrix.

    results : Results
        The results of the item, see :class:`~felupe.mechanics.Results`. An item with
        state variables stores the trial state variables of the last evaluation in
        ``results._statevars``. After a converged iteration, the Newton-Raphson method
        calls ``results.update_statevars()``, which accepts the trial state variables
        as the state variables ``results.statevars`` of the converged solution.

    Notes
    -----
    An item may provide optional methods, which are only required for some features.

    * ``update(value)``: the item is ramped in a :class:`~felupe.Step`, e.g. the
      values of a :class:`~felupe.PointLoad`. :class:`~felupe.mechanics.UpdateItem`
      ramps further parameters of an item.
    * ``checkpoint()`` and ``restore(checkpoint)``: the state of the item, which is
      modified by the iterations of the Newton-Raphson method, is restored by the
      :class:`~felupe.CutbackPlugin`, e.g. the internal fields of a
      :class:`~felupe.SolidBodyNearlyIncompressible`.
    * ``time_step``: the time increment, which is updated by a
      :class:`~felupe.TimeStep`, e.g. of a solid body with a time-dependent material.
    * ``results.force``: the assembled vector, which is used by the
      :class:`~felupe.CharacteristicCurve` to evaluate the reaction forces.
    * ``assemble.mass()``: the mass matrix for a :class:`~felupe.FreeVibration`.
    * ``plot()``: a plotter of the item for the :class:`~felupe.AnimationWriterPlugin`.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube(n=3)
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=2), field=field)
    >>>
    >>> isinstance(solid, fem.mechanics.Item)
    True

    A boundary condition is not an item.

    >>> isinstance(fem.Boundary(field[0]), fem.mechanics.Item)
    False

    See Also
    --------
    felupe.mechanics.Assemble : A class with methods for assembling vectors and
        matrices of an Item.
    felupe.mechanics.Results : A class with intermediate results of an Item.
    felupe.Step : A Step with multiple substeps, subsequently depending on the solution
        of the previous substep.
    felupe.newtonraphson : Find a root of a real function using the Newton-Raphson
        method.
    """

    field: FieldContainer
    assemble: Assemble
    results: Results
