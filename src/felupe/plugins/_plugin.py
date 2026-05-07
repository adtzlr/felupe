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


class Plugin:
    """Base class for plugins.

    Notes
    -----
    All methods (hooks) are optional and will be called at the appropriate time during
    the simulation. The context and state objects are passed to each method, allowing
    a plugin to access and / or modify a simulation as needed.

    The :class:`~felupe.Context` object holds information about the current job,
    step or substep. `state` depends on the method and can be used to access the
    current state, i.e. :class:`~felupe.JobState` or :class:`~felupe.IterationState`.

    ..  note::

        All methods are optional.

    See Also
    --------
    felupe.Job : A job with a list of steps and a method to evaluate them.
    felupe.EventDispatcher : A class to dispatch events to plugins during evaluation.
    felupe.Context : A class to keep track of the context of a Job during evaluation.
    felupe.JobState : A class to keep track of the state of a Job during evaluation.
    felupe.IterationState : A class to keep track of the state of an iteration.
    """

    def before_job(self, context, state):
        """This method is called before the evaluation of a job.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.JobState
            The state of the job.

        """
        pass

    def before_step(self, context, state):
        """This method is called before a step.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.JobState
            The state of the job.

        """
        pass

    def before_newton(self, context, state):
        """This method is called before the Newton-Raphson solver.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.IterationState
            The state of the iteration.

        """
        pass

    def before_iteration(self, context, state):
        """This method is called before a Newton-Raphson iteration.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.IterationState
            The state of the iteration.

        """
        pass

    def before_linear_solve(self, context, state):
        """This method is called before the linear solver inside a Newton-Raphson
        iteration.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.IterationState
            The state of the iteration.

        """
        pass

    def after_linear_solve(self, context, state):
        """This method is called after the linear solver inside a Newton-Raphson
        iteration.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.IterationState
            The state of the iteration.

        """
        pass

    def after_iteration(self, context, state):
        """This method is called after a Newton-Raphson iteration.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.IterationState
            The state of the iteration.

        """
        pass

    def after_newton(self, context, state):
        """This method is called after the Newton-Raphson solver.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.IterationState
            The state of the iteration.

        """
        pass

    def after_substep(self, context, state):
        """This method is called after a substep.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.IterationState
            The state of the iteration.

        """
        pass

    def after_step(self, context, state):
        """This method is called after a step.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.JobState
            The state of the job.

        """
        pass

    def after_job(self, context, state):
        """This method is called after the evaluation of a job.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.JobState
            The state of the job.

        """
        pass
