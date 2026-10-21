# SPDX-License-Identifier: GPL-3.0-or-later

"""A compact live dashboard of a job in a Jupyter widget."""

import math
import sys
import warnings
from time import perf_counter

import numpy as np

from .._plugin import Plugin
from ._surface import surface

# the codes of the status of a substep in the load path
PENDING, RUNNING, CONVERGED, CUTBACK, FAILED = ".", "r", "c", "b", "f"


def _num(value):
    "Return a JSON-compatible float (non-finite values are converted to None)."
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _peak_memory():
    "Return the peak resident memory of the process in bytes (or None)."
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else 1024 * peak
    except Exception:  # pragma: no cover
        return None


def _fields(container):
    "Return the fields of a field container (or of a single field)."
    return list(getattr(container, "fields", [container]))


def _float32(array):
    "Return the bytes of an array as contiguous 32-bit floats."
    return np.ascontiguousarray(np.asarray(array), dtype=np.float32).tobytes()


def _first_line(error):
    "Return the first line of the message of an error (or its type)."
    lines = str(error).strip().splitlines()
    return lines[0][:200] if lines else type(error).__name__


def _kernel():
    "Return True if the code runs in an IPython kernel, e.g. of Jupyter or VS Code."
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover
        return False
    return getattr(get_ipython(), "kernel", None) is not None


def _widget_class():
    "Return the class of the widget (requires the optional dependency anywidget)."
    try:
        from ._widget import DashboardWidget
    except ImportError as error:
        raise ImportError(
            "The DashboardWidgetPlugin requires the optional dependency anywidget. "
            "Install it by `pip install felupe[dashboard]`."
        ) from error
    return DashboardWidget


def _display(widget):  # pragma: no cover
    "Display a widget in the output of the current cell."
    from IPython.display import display

    display(widget)


class DashboardWidgetPlugin(Plugin):
    r"""A compact live dashboard of a :class:`~felupe.Job` in a Jupyter widget.

    The dashboard is shown in the output of the cell which evaluates the job, e.g. in
    JupyterLab, Jupyter Notebook or the interactive window of VS Code. It shows the
    deformed model of the latest converged substep, coloured by the norm of the
    values of a field (displacements, temperature, ...), the progress of the job
    including cutbacks, the properties of the model, the convergence of the
    Newton-Raphson method and performance statistics.

    Parameters
    ----------
    title : str or None, optional
        The title of the dashboard (default is None).
    interval : float, optional
        The minimum time in seconds between two updates of the solver figures during
        a substep (default is 0.1). The end of each substep is always shown.

    Attributes
    ----------
    widget : felupe.plugins._dashboard._widget.DashboardWidget or None
        The widget of the last evaluated job (or None). It may be displayed again,
        e.g. ``display(plugin.widget)`` in another cell.
    overhead : float
        The time in seconds spent in the hooks of the dashboard during the last job.

    Notes
    -----
    The dashboard requires the optional dependency `anywidget
    <https://anywidget.dev>`_, which is installed by ``pip install felupe[dashboard]``.
    The widget is created and displayed in the hook ``before_job``. Each evaluation of
    a job creates a new widget, i.e. the outputs of previous cells keep the state of
    their job. The state of the widget is updated while the cell is running, because
    the messages from the kernel to the frontend are sent immediately. All controls
    of the dashboard (the field, the scale of the deformation, the view) are evaluated
    in the frontend, i.e. they also work while the job is running.

    Outside of an IPython kernel, e.g. in a script which is run in the terminal, a
    warning is raised once and the dashboard is disabled without any overhead. Hence,
    the same script with ``# %%``-cells may be run in the interactive window of VS
    Code (with the dashboard) and in the terminal (without the dashboard). A frontend
    without support for widgets, e.g. a QtConsole, shows a short text instead.

    ..  list-table:: What the dashboard shows.
        :header-rows: 1

        * - Section
          - Content
        * - Viewport
          - The deformed model (by the first field with ``mesh.dim`` components)
            with banded contours of the norm of a field with values at the points
            of the mesh for the latest converged substep. Cells of a higher order
            are drawn by their corner points.
        * - Load path
          - One cell for each substep, converged, converged with cutbacks (with
            marks at the load factors of its increments) or failed.
        * - Solver
          - Residual norm, tolerance, attempt and iteration, the residual norms of
            the attempts of the current substep, iterations, cutbacks and failed
            attempts.
        * - Model
          - The points, cells and the cell type of the mesh, the finite element and
            the quadrature of the region and the degrees of freedom (total and
            active).
        * - Performance
          - Wall time of the assembly, the linear solve, the residual and update
            and the rest, time per iteration and peak memory of the process.

    The times are measured between the hooks of :func:`~felupe.newtonraphson`: the
    assembly between ``before_iteration`` and ``before_linear_solve``, the linear
    solve between ``before_linear_solve`` and ``after_linear_solve`` and the
    residual and update (including a line search) between ``after_linear_solve`` and
    ``after_iteration``.

    The plugin is not callable on purpose. List it before a
    :class:`~felupe.CutbackPlugin` to show the error messages of failed attempts. The
    attempts of a cutback are observed in any order, because they trigger the hooks of
    :func:`~felupe.newtonraphson` again.

    Examples
    --------
    ..  code-block:: python

        import felupe as fem

        mesh = fem.Cube(n=6)
        region = fem.RegionHexahedron(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        boundaries = fem.dof.uniaxial(field, clamped=True)
        solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=50), field=field)

        move = fem.math.linsteps([0, -0.5], num=3)
        step = fem.Step(
            items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
        )

        dashboard = fem.DashboardWidgetPlugin(title="Compression of a cube")
        job = fem.Job(steps=[step], plugins=[dashboard, fem.CutbackPlugin()])
        job.evaluate()

    See Also
    --------
    felupe.Job : A job with a list of steps and a method to evaluate them.
    felupe.Plugin : Base class for plugins.
    felupe.CutbackPlugin : A cutback of the increment of failed substeps.
    felupe.ProgressPlugin : A job plugin to output the progress of a job evaluation in
        the terminal.
    """

    def __init__(self, title=None, interval=0.1):
        self.title = title
        self.interval = float(interval)

        self.widget = None
        self.overhead = 0.0

        self._active = False
        self._warned = False

    # ------------------------------------------------------------------ utilities

    def _elapsed(self):
        return perf_counter() - self._t0

    def _sync(self, force=False):
        "Update the traits of the widget (at most every ``interval`` seconds)."
        now = perf_counter()
        if not force and now - self._last_sync < self.interval:
            return

        self._last_sync = now
        widget = self.widget

        with widget.hold_sync():
            widget.progress = self._progress()
            widget.newton = dict(
                tol=self._tol,
                attempts=[
                    dict(failed=attempt["failed"], fnorms=list(attempt["fnorms"]))
                    for attempt in self._attempts
                ],
            )
            if self._path_changed:
                self._path_changed = False
                widget.path = dict(
                    status="".join(self._path),
                    load_factors={
                        str(k): list(v) for k, v in self._load_factors.items()
                    },
                )

    def _progress(self):
        "Return the solver figures and performance statistics."
        return dict(
            status=self._status,
            elapsed=self._elapsed(),
            step=self._step,
            substep=self._substep,
            attempt=self._attempt,
            iteration=self._attempt_iterations,
            fnorm=self._fnorm,
            xnorm=self._xnorm,
            alpha=self._alpha,
            tol=self._tol,
            iterations=self._total_iterations,
            converged=self._converged,
            cutbacks=self._cutbacks,
            failed_attempts=self._failed_attempts,
            active_dofs=self._active_dofs,
            timing=dict(self._timing),
            memory=_peak_memory(),
            overhead=self.overhead,
            event=self._event,
        )

    def _index(self):
        "Return the index of the current substep in the load path."
        return self._offsets[self._step] + self._substep

    def _set_path(self, code):
        if self._step is not None and self._substep is not None:
            index = self._index()
            if index < len(self._path):
                self._path[index] = code
                self._path_changed = True

    def _frame(self, container, kind, step=None, substep=None):
        "Update the field values of the initial state or of a converged substep."
        fields = _fields(container)
        values = {}
        for index in self._display:
            if index < len(fields):
                values[str(index)] = _float32(fields[index].values)
        self.widget.frame = dict(kind=kind, step=step, substep=substep, values=values)

    def _model(self, job):
        "Update the geometry and the properties of the model."
        container = job.steps[0].items[0].field
        container = getattr(container, "x0", container)
        fields = _fields(container)
        region = fields[0].region
        mesh = region.mesh

        points = np.zeros((mesh.npoints, 3))
        points[:, : mesh.dim] = mesh.points[:, :3]
        surf = surface(mesh)

        # fields with values at the points of the mesh are displayable
        self._display = []
        info = []
        deform = -1
        for index, field in enumerate(fields):
            values = np.asarray(field.values)
            same_mesh = getattr(field.region, "mesh", None) is mesh
            at_points = values.ndim == 2 and values.shape[0] == mesh.npoints
            dim = values.shape[1] if values.ndim == 2 else 1
            if at_points and (same_mesh or index == 0):
                self._display.append(index)
                if deform < 0 and dim == mesh.dim and dim > 1:
                    deform = index
            name = getattr(field, "name", None)
            if name is None:
                name = "Displacement" if index == deform else f"Field {index}"
            info.append(
                dict(
                    index=index,
                    name=str(name),
                    kind=type(field).__name__,
                    dim=int(dim),
                    display=index in self._display,
                )
            )

        items = []
        for step in job.steps:
            for item in step.items:
                umat = getattr(item, "umat", None)
                text = type(item).__name__
                if umat is not None:
                    text += f" ({type(umat).__name__})"
                if text not in items:
                    items.append(text)

        quadrature = getattr(region, "quadrature", None)
        npoints_quadrature = getattr(quadrature, "npoints", None)
        nsubsteps = [int(step.nsubsteps) for step in job.steps]

        self._offsets = np.concatenate([[0], np.cumsum(nsubsteps)]).astype(int).tolist()
        self._path = [PENDING] * sum(nsubsteps)
        self._path_changed = True

        widget = self.widget
        widget.info = dict(
            title=self.title,
            cell_type=str(mesh.cell_type),
            dim=int(mesh.dim),
            npoints=int(mesh.npoints),
            ncells=int(mesh.ncells),
            points_per_cell=int(np.shape(mesh.cells)[1]),
            ndof=int(sum(np.asarray(f.values).size for f in fields)),
            element=type(getattr(region, "element", None)).__name__,
            quadrature=None if quadrature is None else type(quadrature).__name__,
            quadrature_points_per_cell=None
            if npoints_quadrature is None
            else int(npoints_quadrature),
            surface=surf["kind"],
            fields=info,
            deform=deform,
            nsubsteps=nsubsteps,
            items=items,
        )
        widget.points = _float32(points)
        widget.triangles = np.ascontiguousarray(surf["tri"], dtype=np.uint32).tobytes()
        widget.quads = np.ascontiguousarray(surf["quad"], dtype=np.uint32).tobytes()
        widget.lines = np.ascontiguousarray(surf["line"], dtype=np.uint32).tobytes()
        self._frame(container, kind="initial")

    def _close_attempt(self, success, message=None):
        "Finish the current attempt of the Newton-Raphson method."
        if not self._attempt_open:
            return

        self._attempt_open = False
        self._attempts[-1]["failed"] = not success

        if not success:
            self._failures += 1
            self._failed_attempts += 1
            if message is None:
                message = self._last_reason or "not converged"
            iterations = self._attempt_iterations
            self._event = dict(
                kind="fail",
                text=(
                    f"Step {self._step + 1}, substep {self._substep + 1}: attempt "
                    f"{self._attempt} failed after {iterations} iteration"
                    f"{'' if iterations == 1 else 's'} ({message.rstrip('.')})."
                ),
            )
            self._sync(force=True)

    # ---------------------------------------------------------------------- hooks

    def before_job(self, context, state):
        if not _kernel():
            if not self._warned:
                self._warned = True
                warnings.warn(
                    "The DashboardWidgetPlugin requires an IPython kernel, e.g. of "
                    "Jupyter or of the interactive window of VS Code. The dashboard "
                    "is disabled.",
                    stacklevel=2,
                )
            self._active = False
            return

        widget_class = _widget_class()

        tic = perf_counter()
        self._t0 = tic
        self._last_sync = -math.inf
        self._active = True

        self._status = "running"
        self._step = None
        self._substep = None
        self._attempt = 0
        self._attempt_open = False
        self._attempt_iterations = 0
        self._attempts = []
        self._fnorm = self._xnorm = self._alpha = self._tol = None
        self._total_iterations = 0
        self._converged = 0
        self._cutbacks = 0
        self._failures = 0
        self._failed_attempts = 0
        self._active_dofs = None
        self._timing = dict(assembly=0.0, solve=0.0, residual=0.0)
        self._load_factors = {}
        self._last_reason = None
        self._event = None
        self.overhead = 0.0

        # a new widget for each job, i.e. the outputs of previous cells are kept
        self.widget = widget_class()
        with self.widget.hold_sync():
            self._model(context.job)
            self._sync(force=True)

        _display(self.widget)
        self.overhead += perf_counter() - tic

    def before_step(self, context, state):
        if not self._active:
            return
        self._step = state.stepnumber

    def before_substep(self, context, state):
        if not self._active:
            return
        tic = perf_counter()
        self._substep = state.substepnumber
        self._attempt = 0
        self._attempt_iterations = 0
        self._attempts = []
        self._failures = 0
        self._attempt_open = False
        self._set_path(RUNNING)
        self._sync(force=True)
        self.overhead += perf_counter() - tic

    def before_newton(self, context, state):
        if not self._active or self._substep is None:
            return
        tic = perf_counter()

        # an attempt without after_newton has failed (e.g. within a cutback)
        self._close_attempt(success=False)

        self._attempt += 1
        self._attempt_open = True
        self._attempt_iterations = 0
        self._attempts.append(dict(failed=None, fnorms=[]))
        self._last_reason = None
        self._tol = _num(getattr(state, "tol", None))

        dof1 = getattr(context, "dof1", None)
        if dof1 is not None:
            self._active_dofs = int(len(dof1))

        self.overhead += perf_counter() - tic
        self._tic = perf_counter()

    def before_iteration(self, context, state):
        if not self._active or self._substep is None:
            return
        toc = perf_counter()
        # the initial residual of an attempt
        self._timing["residual"] += toc - self._tic
        self._tic = toc

    def before_linear_solve(self, context, state):
        if not self._active or self._substep is None:
            return
        toc = perf_counter()
        self._timing["assembly"] += toc - self._tic
        self._tic = toc

    def after_linear_solve(self, context, state):
        if not self._active or self._substep is None:
            return
        toc = perf_counter()
        self._timing["solve"] += toc - self._tic
        self._tic = toc

    def after_iteration(self, context, state):
        if not self._active or self._substep is None:
            return
        toc = perf_counter()
        self._timing["residual"] += toc - self._tic

        self._attempt_iterations += 1
        self._total_iterations += 1

        self._fnorm = _num(state.fnorm)
        self._xnorm = _num(state.xnorm)
        self._alpha = _num(getattr(state, "alpha", None))
        self._attempts[-1]["fnorms"].append(self._fnorm)

        # the default tolerance is derived in the first iteration
        self._tol = _num(getattr(state, "tol", None))

        if getattr(state, "error", False):
            self._last_reason = (
                "NaN in the residuals" if self._fnorm is None else "maximum iterations"
            )

        self._sync()
        self.overhead += perf_counter() - toc
        self._tic = perf_counter()

    def after_newton(self, context, state):
        if not self._active or self._substep is None:
            return
        self._close_attempt(success=True)

    def after_failed_substep(self, context, state):
        if not self._active:
            return
        tic = perf_counter()
        message = None if state.error is None else _first_line(state.error)
        self._close_attempt(success=False, message=message)
        self.overhead += perf_counter() - tic

    def after_substep(self, context, state):
        if not self._active:
            return
        tic = perf_counter()
        self._close_attempt(success=True)
        self._converged += 1

        load_factors = getattr(state, "load_factors", None)

        if self._failures > 0:
            self._cutbacks += 1
            self._set_path(CUTBACK)
            if load_factors is not None:
                self._load_factors[self._index()] = [_num(f) for f in load_factors]
            increments = len(load_factors or [])
            self._event = dict(
                kind="cut",
                text=(
                    f"Step {self._step + 1}, substep {self._substep + 1}: converged "
                    f"with {self._failures} cutback{'' if self._failures == 1 else 's'}"
                    f" in {increments} increments."
                ),
            )
        else:
            self._set_path(CONVERGED)

        with self.widget.hold_sync():
            self._frame(
                context.substep.x,
                kind="substep",
                step=self._step,
                substep=self._substep,
            )
            self._sync(force=True)

        self.overhead += perf_counter() - tic

    def after_job(self, context, state):
        if not self._active:
            return

        error = getattr(state, "error", None)

        if error is None:
            self._status = "finished"
            self._event = dict(kind="done", text="All steps converged.")
        else:
            self._close_attempt(success=False, message=_first_line(error))
            self._set_path(FAILED)
            self._status = "failed"
            self._event = dict(
                kind="fail", text=f"{type(error).__name__}: {_first_line(error)}"
            )

        self._sync(force=True)
        self._active = False
