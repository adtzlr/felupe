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

HOOKS = (
    "before_job",
    "before_step",
    "before_substep",
    "before_newton",
    "before_iteration",
    "before_linear_solve",
    "after_linear_solve",
    "after_iteration",
    "after_newton",
    "after_failed_substep",
    "after_substep",
    "after_step",
    "after_job",
)


class Context:
    """A class to keep track of the context of a Job during evaluation.

    Parameters
    ----------
    job : felupe.Job or None, optional
        The job object.
    step : felupe.Step or None, optional
        The step object.
    substep : felupe.tools.NewtonResult or None, optional
        The result of the completed substep. The field container of the substep is
        available as ``substep.x``.
    items : list or None, optional
        The list of items of the Newton-Raphson method or of a step (default is None).
        Only available in the hooks of :func:`~felupe.newtonraphson` and in the hooks
        of a substep, see :meth:`~felupe.Step.generate_states`.
    dof1 : ndarray or None, optional
        The active degrees of freedom of the Newton-Raphson method (default is None).
        Only available in the hooks of :func:`~felupe.newtonraphson`.
    dof0 : ndarray or None, optional
        The prescribed degrees of freedom of the Newton-Raphson method (default is
        None). Only available in the hooks of :func:`~felupe.newtonraphson`.
    ext0 : ndarray or None, optional
        The external values of the prescribed degrees of freedom of the Newton-Raphson
        method (default is None). Only available in the hooks of
        :func:`~felupe.newtonraphson`.
    fun : callable or None, optional
        A callable ``f = fun(x)`` which evaluates the objective function (the
        residuals) for given unknowns ``x``. For a list of items, the vectors of all
        items are assembled and the fields of the items are linked to ``x``.
        Additional arguments of the Newton-Raphson method are already bound. Default is
        None. Only available in the hooks of :func:`~felupe.newtonraphson`.
    update : callable or None, optional
        The callable ``x = update(x, dx)`` of the Newton-Raphson method which updates
        the unknowns (default is None). Only available in the hooks of
        :func:`~felupe.newtonraphson`.
    check : callable or None, optional
        A callable ``xnorm, fnorm, success = check(dx, x, f)`` which checks the
        convergence of the Newton-Raphson method for given unknowns ``x`` with the
        values of the objective function ``f`` (default is None). In contrast to the
        ``check``-argument of :func:`~felupe.newtonraphson`, the state variables of
        the items are not updated. Only available in the hooks of
        :func:`~felupe.newtonraphson`.
    x0 : felupe.FieldContainer or None, optional
        The field container with the unknowns of a step, which is the starting point
        of the Newton-Raphson method of a substep (default is None). Only available in
        the hooks of a substep, see :meth:`~felupe.Step.generate_states`.
    solve : callable or None, optional
        A callable ``res = solve(values)`` which updates the ramped items (and
        boundaries) of a step with a dict of ``values``, updates the load case and
        evaluates the Newton-Raphson method, starting from the unknowns ``x0``. It
        returns a :class:`~felupe.tools.NewtonResult` and errors of the Newton-Raphson
        method are raised. On success, the unknowns ``x0`` are linked to the result,
        i.e. the result is the starting point of the next evaluation. Default is None.
        Only available in the hooks of a substep, see
        :meth:`~felupe.Step.generate_states`.

    See Also
    --------
    felupe.Plugin : Base class for plugins.
    felupe.IterationState : A class to keep track of the state of an iteration.
    felupe.SubstepState : A class to keep track of the state of a substep.

    """

    def __init__(
        self,
        job=None,
        step=None,
        substep=None,
        items=None,
        dof1=None,
        dof0=None,
        ext0=None,
        fun=None,
        update=None,
        check=None,
        x0=None,
        solve=None,
    ):
        self.job = job
        self.step = step
        self.substep = substep
        self.items = items
        self.dof1 = dof1
        self.dof0 = dof0
        self.ext0 = ext0
        self.fun = fun
        self.update = update
        self.check = check
        self.x0 = x0
        self.solve = solve


class EventDispatcher:
    """A class to dispatch events to plugins during evaluation.

    Parameters
    ----------
    plugins : list or None, optional
        The list of plugins.

    """

    def __init__(self, plugins=None):
        self.plugins = list(plugins) if plugins is not None else []
        self.dispatcher = self.configure(self.plugins)

    def configure(self, plugins):
        """Configure the dispatcher with the given plugins and return a dict with hooks
        as keys.
        """

        self.plugins = plugins
        dispatcher = {hook: [] for hook in HOOKS}

        # check if methods are available for the hooks in the plugins
        # add them to the dispatcher if they are available
        for plugin in self.plugins:

            # simple callable plugin
            if callable(plugin):
                dispatcher["after_substep"].append(plugin)
                continue

            # hook-based plugin
            for hook in HOOKS:
                method = getattr(plugin, hook, None)
                if method is not None:
                    dispatcher[hook].append(method)

        return dispatcher

    def add_plugin(self, plugin):
        "Add a plugin to the dispatcher and reconfigure it."

        self.plugins.append(plugin)
        self.dispatcher = self.configure(self.plugins)

    def trigger(self, hook, context, state):
        "Trigger a hook with context and current state to all registered functions."

        for fun in self.dispatcher[hook]:
            fun(context, state)
