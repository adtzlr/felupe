# SPDX-License-Identifier: GPL-3.0-or-later

from ._helpers import Assemble, Results


class TimeStep:
    r"""A time step item, which updates the time increment of other items.

    Parameters
    ----------
    items : list of Item
        List of items with an attribute ``time_step``, which is updated with the time
        increment, e.g. solid bodies with time-dependent materials
        (:class:`~felupe.SolidBody`, :class:`~felupe.SolidBodyNearlyIncompressible` or
        :class:`~felupe.TrussBody`) or thermal items
        (:class:`~felupe.thermal.SolidBodyThermal`,
        :class:`~felupe.thermal.SolidBodyHeatFlux`,
        :class:`~felupe.thermal.SolidBodySurfaceHeatTransfer` or
        :class:`~felupe.thermal.SolidBodySurfaceRadiation`).
    time_old : float, optional
        Initial time (default is 0.0).
    time_new : float or None, optional
        New time (default is None).

    Notes
    -----
    The time :math:`t` is ramped in a :class:`~felupe.Step`, i.e. the time step item is
    a key of the ramp with the values of the time for each substep. For a new time
    :math:`t_{n+1}`, the time increment :math:`\Delta t` is evaluated as the difference
    between the new time and the old time, see Eq. :eq:`time-step`.

    ..  math::
        :label: time-step

        \Delta t = t_{n+1} - t_n

    The time increment is assigned to the attribute ``time_step`` of each item in the
    list and the new time becomes the old time. A time increment of zero is valid, e.g.
    for the first value of a ramp which starts at the initial time. A new time which is
    smaller than the old time raises a :class:`ValueError`.

    All ramped items of a :class:`~felupe.Step` are updated before the Newton-Raphson
    iterations of a substep. Hence, the time increment is assigned to the items before
    they are evaluated, independent of the position of the time step item in the list
    of items of the step.

    The old time is part of the :meth:`checkpoint` of the time step item. If a substep
    is subdivided into increments by a :class:`~felupe.CutbackPlugin`, the time is
    interpolated and the time increment of each increment is the difference to the
    time of the last converged increment.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=11).triangulate()
        >>> region = fem.RegionTriangle(mesh)
        >>> temperature = fem.Field(region, dim=1)
        >>> field = fem.FieldContainer([temperature])
        >>>
        >>> solid = fem.thermal.SolidBodyThermal(
        ...     field,
        ...     mass_density=1.0,
        ...     specific_heat_capacity=1.0,
        ...     thermal_conductivity=1.0,
        ... )
        >>>
        >>> time = fem.TimeStep(items=[solid])
        >>> table = fem.math.linsteps([0, 1], num=100)
        >>>
        >>> step = fem.Step(items=[time, solid], ramp={time: table})

    See Also
    --------
    felupe.Step : A Step with multiple substeps, subsequently depending on the solution
        of the previous substep.
    felupe.CutbackPlugin : A cutback of the increment of failed substeps.
    felupe.thermal.SolidBodyThermal : A thermal solid body.

    """

    def __init__(
        self,
        items,
        time_old=0.0,
        time_new=None,
    ):
        self.field = items[0].field  # get dummy field from first item
        self.assemble = Assemble()
        self.results = Results()
        self.time_old = time_old
        self.time_new = time_new
        self.items = items

    def update(self, time_new):
        time_step = time_new - self.time_old

        if time_step < 0:
            raise ValueError(
                f"New time {time_new} is smaller than old time {self.time_old}."
            )

        for item in self.items:
            item.time_step = time_step

        self.time_old = time_new

    def checkpoint(self):
        """Return a checkpoint of the time step item.

        Returns
        -------
        dict
            A dict with the old time.

        See Also
        --------
        felupe.TimeStep.restore : Restore a checkpoint of the time step item.
        felupe.CutbackPlugin : A cutback of the increment of failed substeps.
        """

        return {"time_old": self.time_old}

    def restore(self, checkpoint):
        """Restore a checkpoint inplace.

        Parameters
        ----------
        checkpoint : dict
            A dict with the old time.

        See Also
        --------
        felupe.TimeStep.checkpoint : Return a checkpoint of the time step item.
        """

        self.time_old = checkpoint["time_old"]
