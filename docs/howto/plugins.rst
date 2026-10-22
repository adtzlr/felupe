Use Plugins
~~~~~~~~~~~

Plugins extend the evaluation of a :class:`~felupe.Job`. A plugin provides methods
with the names of events, called hooks, e.g. before or after each step, each substep or
each iteration of the Newton-Raphson method. Plugins record results, write files, show
the progress or even change the evaluation, e.g. by a line search or by a cutback of
failed substeps. The plugins are passed as a list to the job.

..  code-block:: python

    import felupe as fem

    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=50), field=field)

    move = fem.math.linsteps([0, -0.4], num=4)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    job = fem.Job(steps=[step], plugins=[fem.CutbackPlugin()])
    job.evaluate()

Built-in Plugins
****************

FElupe ships with the following plugins. The progress and the XDMF result file are
enabled by the arguments of :meth:`~felupe.Job.evaluate`, all other plugins are passed
to the job.

..  list-table::
    :header-rows: 1

    * - Plugin
      - Description
      - Optional dependencies
    * - :class:`~felupe.ProgressPlugin`
      - Shows a progress bar or a text-based output of the Newton-Raphson iterations.
      - ``felupe[progress]``
    * - :class:`~felupe.XDMFWriterPlugin`
      - Writes the results of all substeps to a XDMF time series file.
      - ``felupe[io]``
    * - :class:`~felupe.CharacteristicCurvePlugin`
      - Records a force-displacement curve of a boundary condition.
      - ``felupe[plot]`` (for plotting)
    * - :class:`~felupe.AnimationWriterPlugin`
      - Writes an animation (GIF or MP4) of the deformed items.
      - ``felupe[view,io]``
    * - :class:`~felupe.DashboardWidgetPlugin`
      - Shows a live dashboard of the job in a Jupyter widget or in a live HTML file.
      - ``felupe[dashboard]`` (for the widget)
    * - :class:`~felupe.LinesearchPlugin`
      - Scales the Newton increment by a step length (backtracking line search).
      -
    * - :class:`~felupe.CutbackPlugin`
      - Subdivides failed substeps into smaller increments.
      -

..  tab:: Progress

    The progress is shown by default. Its verbosity is controlled by the argument
    ``verbose`` of :meth:`~felupe.Job.evaluate`: a progress bar for ``True`` (default),
    a text-based output of the Newton-Raphson iterations for ``2`` and no output for
    ``False``. The progress bar requires ``tqdm``, otherwise the text-based output is
    used.

    ..  code-block:: python

        job.evaluate(verbose=2)

..  tab:: XDMF result file

    A XDMF time series result file is written if a filename is given. Additional
    point- and cell-data may be passed by the arguments ``point_data`` and
    ``cell_data``. The result file may be opened in ParaView.

    ..  code-block:: python

        job.evaluate(filename="result.xdmf")

..  tab:: Characteristic curve

    The displacement of the first point and the sum of the reaction forces of a
    :class:`~felupe.Boundary` are recorded after each completed substep. A
    :class:`~felupe.CharacteristicCurve` is a job with a built-in characteristic
    curve plugin.

    ..  code-block:: python

        curve = fem.CharacteristicCurvePlugin(boundary=boundaries["move"])
        job = fem.Job(steps=[step], plugins=[curve]).evaluate()

        x, y = curve.to_arrays()
        fig, ax = curve.plot(xlabel="Displacement", ylabel="Force", marker="o")

..  tab:: Animation

    A frame of the deformed items is written after each completed substep. Additional
    keyword-arguments are passed to the plot-methods of the items, e.g.
    ``solid.plot(name, off_screen=True)``.

    ..  code-block:: python

        animation = fem.AnimationWriterPlugin(
            items=[solid],
            filename="animation.gif",  # or "animation.mp4"
            name="Principal Values of Cauchy Stress",
            off_screen=True,
        )
        job = fem.Job(steps=[step], plugins=[animation]).evaluate()

..  tab:: Dashboard

    The dashboard is shown in the output of the cell which evaluates the job, e.g. in
    JupyterLab or in the interactive window of VS Code. It shows the deformed model of
    the latest converged substep, the load path including cutbacks, the convergence of
    the Newton-Raphson method and performance statistics. With a filename, a live HTML
    file is written, which updates itself in the web browser. This also works for a
    script in a terminal (without a Jupyter frontend).

    ..  code-block:: python

        dashboard = fem.DashboardWidgetPlugin(title="Compression of a cube")
        job = fem.Job(steps=[step], plugins=[dashboard]).evaluate()

        # a live HTML file, e.g. for a script in a terminal
        dashboard = fem.DashboardWidgetPlugin(filename="job.html")

..  tab:: Line search

    The step length of the Newton increment is halved until the deformation gradients
    are admissible and the residuals decrease sufficiently. The line search is also
    available for :func:`~felupe.newtonraphson`. For a
    :class:`~felupe.SolidBodyNearlyIncompressible`, see the notes of
    :class:`~felupe.LinesearchPlugin`.

    ..  code-block:: python

        linesearch = fem.LinesearchPlugin()
        job = fem.Job(steps=[step], plugins=[linesearch]).evaluate()

        linesearch.alphas  # the accepted step lengths, one list for each substep

..  tab:: Cutback

    If the Newton-Raphson method of a substep fails, the unknowns and the state of the
    items are restored and the substep is subdivided into smaller increments. Custom
    items with a state, which is modified during the Newton iterations, have to provide
    the methods ``checkpoint()`` and ``restore(checkpoint)``, see
    :class:`~felupe.mechanics.Item`.

    ..  code-block:: python

        cutback = fem.CutbackPlugin(factor=0.5, max_cutbacks=10)
        job = fem.Job(steps=[step], plugins=[cutback]).evaluate()

        cutback.load_factors  # the load factors of the increments of each substep
        cutback.cutbacks  # the number of cutbacks of each substep

Combine Plugins
***************

Plugins may be combined. The hooks of the plugins are called in the order of the list,
followed by the plugins of :meth:`~felupe.Job.evaluate` (the progress and the result
file). A failed line search leads to a cutback. The increments of a subdivided substep
are internal to the substep, i.e. plugins which record or write results after a
completed substep, like the characteristic curve, are called only once per substep.

..  code-block:: python

    curve = fem.CharacteristicCurvePlugin(boundary=boundaries["move"])
    dashboard = fem.DashboardWidgetPlugin()

    job = fem.Job(
        steps=[step],
        plugins=[dashboard, fem.LinesearchPlugin(), fem.CutbackPlugin(), curve],
    )
    job.evaluate(filename="result.xdmf")

..  note::

    List a :class:`~felupe.DashboardWidgetPlugin` before a
    :class:`~felupe.CutbackPlugin` to show the error messages of failed attempts.

Custom Plugins
**************

A custom plugin is a subclass of :class:`~felupe.Plugin`. All hooks are optional, i.e.
only the required hooks have to be implemented. Each hook takes the context, e.g. the
job, the step or the result of a substep, and the current state, e.g. the step number
or the norm of the residuals of an iteration, see :class:`~felupe.Context`,
:class:`~felupe.JobState`, :class:`~felupe.SubstepState` and
:class:`~felupe.IterationState`. The hooks are called in the following order.

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

This plugin counts the Newton-Raphson iterations, including the iterations of failed
attempts of a :class:`~felupe.CutbackPlugin`. The hook ``after_job`` is also called if
the evaluation of the job fails.

..  code-block:: python

    class IterationCounter(fem.Plugin):
        "Count the Newton-Raphson iterations."

        def __init__(self):
            self.iterations = []

        def before_newton(self, context, state):
            self.iterations.append(0)

        def after_iteration(self, context, state):
            self.iterations[-1] += 1

        def after_job(self, context, state):
            print(f"Newton-Raphson iterations: {sum(self.iterations)}")

    counter = IterationCounter()
    job = fem.Job(steps=[step], plugins=[counter]).evaluate()

The hooks of the Newton-Raphson method are also called by
:func:`~felupe.newtonraphson`, i.e. the same plugin may be passed to
``newtonraphson(items, plugins=[counter], **loadcase)``. The hooks of the job, the
steps and the substeps are only called by a job.

A callable, e.g. a function ``plugin(context, state)``, is a simple plugin which is
called after each completed substep.

..  code-block:: python

    umax = []

    def save_umax(context, state):
        umax.append(abs(context.substep.x[0].values).max())

    job = fem.Job(steps=[step], plugins=[save_umax]).evaluate()

..  note::

    Plugins may also change the evaluation. In the hook ``after_linear_solve``, the
    update of the unknowns may be modified, see :class:`~felupe.IterationState`. In the
    hook ``after_failed_substep``, a failed substep may be recovered, see
    :class:`~felupe.SubstepState`. The :class:`~felupe.LinesearchPlugin` and the
    :class:`~felupe.CutbackPlugin` are implemented by these hooks.

..  note::

    Plugins replace the callbacks of a :class:`~felupe.Job` and of
    :func:`~felupe.newtonraphson` (removed in v12.0.0). The migration is shown in the
    notes of :class:`~felupe.Job` and of :func:`~felupe.newtonraphson`.
