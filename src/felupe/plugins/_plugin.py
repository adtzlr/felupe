# SPDX-License-Identifier: GPL-3.0-or-later


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
    for given values of the ramp, see :meth:`~felupe.Step.generate_states`. `state`
    depends on the method and can be used to access the current state, i.e.
    :class:`~felupe.JobState`, :class:`~felupe.SubstepState` (in the hooks of a
    substep, including ``after_substep``) or :class:`~felupe.IterationState`.

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

    A plugin may also be a callable ``plugin(context, state)``, e.g. a function, which
    is only called after each completed substep (in the hook ``after_substep``). It
    replaces the ``callback`` of a :class:`~felupe.Job` (removed in 12.0.0), see the
    notes of :class:`~felupe.Job`.

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
        """This method is called after a completed substep.

        Parameters
        ----------
        context : felupe.Context
            The context object with the job, the step and the result of the substep
            ``context.substep``.
        state : felupe.SubstepState
            The state of the substep (the same object as in the hooks
            ``before_substep`` and ``after_failed_substep``), which is also a
            :class:`~felupe.JobState` with the step number, the substep number and the
            time.

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
        """This method is called after the evaluation of a job, also if the evaluation
        failed (e.g. to close result files). Then, the error is available as
        ``state.error`` and it is raised after all plugins are called.

        Parameters
        ----------
        context : felupe.Context
            The context object.
        state : felupe.JobState
            The state of the job.

        """
        pass
