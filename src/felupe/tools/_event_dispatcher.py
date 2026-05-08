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
    "before_newton",
    "before_iteration",
    "before_linear_solve",
    "after_linear_solve",
    "after_iteration",
    "after_newton",
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
    substep : felupe.FieldContainer or None, optional
        The field container object.

    """

    def __init__(self, job=None, step=None, substep=None):
        self.job = job
        self.step = step
        self.substep = substep


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
