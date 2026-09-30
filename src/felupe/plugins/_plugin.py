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
    step or substep. In the hooks of the Newton-Raphson method, it holds the items, the
    degrees of freedom and the callables of :func:`~felupe.newtonraphson`. In the hooks
    of a substep, it holds the items, the unknowns and a callable to solve the substep
    for given values of the ramp, see :meth:`~felupe.Step.generate`. `state` depends on
    the method and can be used to access the current state, i.e.
    :class:`~felupe.JobState`, :class:`~felupe.SubstepState` or
    :class:`~felupe.IterationState`.

    The hooks are called in the following order during the evaluation of a
    :class:`~felupe.Job`. The hook ``after_failed_substep`` is only called if the
    Newton-Raphson method of a substep raised an error.

    ..  code-block:: text

        before_job
            before_step
                before_substep
                    before_newton
                        before_iteration
                            before_linear_solve
                            after_linear_solve
                        after_iteration
                    after_newton
                    after_failed_substep
                after_substep
            after_step
        after_job

    ..  note::

        All methods are optional.

    See Also
    --------
    felupe.Job : A job with a list of steps and a method to evaluate them.
    felupe.EventDispatcher : A class to dispatch events to plugins during evaluation.
    felupe.Context : A class to keep track of the context of a Job during evaluation.
    felupe.JobState : A class to keep track of the state of a Job during evaluation.
    felupe.SubstepState : A class to keep track of the state of a substep.
    felupe.IterationState : A class to keep track of the state of an iteration.
    felupe.LinesearchPlugin : A backtracking line search for the Newton-Raphson method.
    felupe.CutbackPlugin : A cutback of the increment of failed substeps.
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

    def before_substep(self, context, state):
        """This method is called before a substep, i.e. before the ramped items of the
        step are updated and before the Newton-Raphson method is evaluated.

        Parameters
        ----------
        context : felupe.Context
            The context object with the step, the items, the unknowns ``x0`` and the
            callable ``solve``.
        state : felupe.SubstepState
            The state of the substep.

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
        iteration. A plugin may modify the update of the unknowns in this hook, see
        :class:`~felupe.IterationState`.

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

    def after_failed_substep(self, context, state):
        """This method is called after the Newton-Raphson method of a substep raised
        an error. A plugin may recover the substep in this hook, see
        :class:`~felupe.SubstepState`.

        Parameters
        ----------
        context : felupe.Context
            The context object with the step, the items, the unknowns ``x0`` and the
            callable ``solve``.
        state : felupe.SubstepState
            The state of the substep with the raised error.

        """
        pass

    def after_substep(self, context, state):
        """This method is called after a substep.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.JobState
            The state of the job.

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
