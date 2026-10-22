# SPDX-License-Identifier: GPL-3.0-or-later

"""A compact live dashboard of a job in a Jupyter widget."""

import math
import os
import sys
import warnings
import webbrowser
from time import perf_counter

import numpy as np

from ...__about__ import __version__
from .._plugin import Plugin
from ._file import DashboardFile, DashboardState
from ._surface import CORNERS, compact, surface

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


def _peak_memory_windows():
    """Return the peak working set of the process in bytes by the Windows API (the
    module resource is not available on Windows)."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()

    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return None

    return int(counters.PeakWorkingSetSize)


def _peak_memory():
    """Return the peak resident memory of the process in bytes (or None), i.e. the
    maximum resident set size on Linux and macOS or the peak working set on Windows."""
    try:
        if sys.platform == "win32":
            return _peak_memory_windows()

        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else 1024 * peak
    except Exception:
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


def _stacklevel():
    """Return the stack level of the first caller outside of FElupe, e.g. of the
    evaluation of a job, for a warning in a hook of the dashboard."""
    package = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame = frame.f_back
        level += 1
    return level


def _shell():
    "Return the IPython shell, e.g. of a Jupyter kernel or of the IPython terminal."
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _kernel():
    "Return True if the code runs in an IPython kernel, e.g. of Jupyter or VS Code."
    return getattr(_shell(), "kernel", None) is not None


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


def _display(widget):
    "Display a widget in the output of the current cell."
    from IPython.display import display

    display(widget)


class DashboardWidgetPlugin(Plugin):
    r"""A compact live dashboard of a :class:`~felupe.Job` in a Jupyter widget.

    The dashboard is shown in the output of the cell which evaluates the job, e.g. in
    JupyterLab, Jupyter Notebook or the interactive window of VS Code. It shows the
    deformed model of the latest converged substep, coloured by the norm of the
    values of the first field (displacements, temperature, ...), the progress of the job
    including cutbacks, the properties of the model, the convergence of the
    Newton-Raphson method and performance statistics.

    Parameters
    ----------
    title : str or None, optional
        The title of the dashboard (default is None).
    interval : float, optional
        The minimum time in seconds between two updates of the solver figures during
        a substep (default is 0.1). The end of each substep is always shown.
    filename : str or None, optional
        The filename of a live HTML file of the dashboard, e.g. ``"job.html"`` (default
        is None). The file is updated during the job and the page in the web browser
        updates itself, see the notes. Without a Jupyter frontend, the page is opened
        in the web browser.
    file_interval : float, optional
        The minimum time in seconds between two updates of the live HTML file
        (default is 2.0). The file is only updated if the state of the job has changed.

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
    of the dashboard (the scale of the deformation, the mesh lines, the view) are
    evaluated in the frontend, i.e. they also work while the job is running.

    The model is drawn from the regions of the items, e.g. of several solid bodies on
    the meshes of a :class:`~felupe.MeshContainer`. There is no list of item types:
    every item with a field container ``item.field`` contributes the mesh of the
    region of its first field, except

    * boundary regions (e.g. of a :class:`~felupe.SolidBodyPressure` or of a heat
      transfer), whose meshes hold the cells of the faces,
    * meshes without a supported cell type (e.g. the vertex mesh of the field of a
      :class:`~felupe.MeshContainer` in a :class:`~felupe.PointLoad`),
    * meshes (or identical cells) which are already drawn (e.g. of a
      :class:`~felupe.SolidBodyForce` on the region of a solid body) and
    * meshes with other points than the first mesh.

    Items without a field, e.g. a :class:`~felupe.thermal.TimeStep`, are skipped.
    Faces at the interface of two bodies with merged points are not drawn. The values
    are taken from the first field of the unknowns ``x0`` of the job, i.e.
    ``job.evaluate(x0=field)`` shows the field of all bodies.

    Without a Jupyter frontend, e.g. in a script which is run in the terminal or in the
    IPython terminal (which has no kernel and can not display widgets), a warning is
    raised once and the dashboard is disabled without any overhead (unless a
    ``filename`` is given). Hence, the same script with ``# %%``-cells may be run in
    the interactive window of VS Code (with the dashboard) and in the terminal
    (without the dashboard). A frontend without support for widgets, e.g. a QtConsole,
    shows a short text instead.

    With a ``filename``, e.g. ``"job.html"``, a live HTML file of the dashboard is
    written, also without a Jupyter frontend (and without anywidget). Next to the page,
    three scripts are written: ``job.geometry.js`` (once for each job),
    ``job.state.js`` (at most every ``file_interval`` seconds, if the state of the job
    has changed) and ``job.version.js``. Each file is replaced atomically. The page in
    the web browser polls the version and updates itself without a reload, i.e. the
    view (rotation, zoom, scale) is kept. A page of a local file can not fetch other
    files, but it can load scripts, i.e. no server is required. Without a Jupyter
    frontend, the page is opened in the web browser by the first job of the plugin.
    The page follows the next jobs of the plugin and it shows the final state of the
    last job.

    ..  list-table:: What the dashboard shows.
        :header-rows: 1

        * - Section
          - Content
        * - Viewport
          - The model with banded contours of the norm of the first field of the
            unknowns for the latest converged substep, deformed by this field if
            it has ``mesh.dim`` components. Only the first field is drawn, the
            other fields may be defined on other meshes. Cells of a higher order
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

    A live HTML file in the web browser, e.g. for a script in a terminal.

    ..  code-block:: python

        dashboard = fem.DashboardWidgetPlugin(filename="job.html")
        job = fem.Job(steps=[step], plugins=[dashboard])
        job.evaluate()

    See Also
    --------
    felupe.Job : A job with a list of steps and a method to evaluate them.
    felupe.Plugin : Base class for plugins.
    felupe.CutbackPlugin : A cutback of the increment of failed substeps.
    felupe.ProgressPlugin : A job plugin to output the progress of a job evaluation in
        the terminal.
    """

    def __init__(self, title=None, interval=0.1, filename=None, file_interval=2.0):
        self.title = title
        self.interval = float(interval)
        self.filename = filename

        self.widget = None
        self.overhead = 0.0

        self._active = False
        self._warned = False
        self._opened = False

        # the live HTML file (the same for all jobs, i.e. an open page follows them)
        self._file = None
        if filename is not None:
            self._file = DashboardFile(filename, interval=file_interval)

    # ------------------------------------------------------------------ utilities

    def _elapsed(self):
        return perf_counter() - self._t0

    def _sync(self, force=False):
        "Update the traits of the widget (at most every ``interval`` seconds)."
        now = perf_counter()
        if not force and now - self._last_sync < self.interval:
            return

        self._last_sync = now
        widget = self._state

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

        self._flush()

    def _flush(self, force=False):
        """Update the live HTML file (at most every ``file_interval`` seconds), after
        it is started for the geometry of the job."""
        if self._file is not None and self._file_started:
            self._file.changed = True
            self._file.flush(self._state, force=force)

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
        "Set the status of the current substep in the load path (if there is one)."
        if self._substep is not None:
            self._path[self._index()] = code
            self._path_changed = True

    def _frame(self, container, kind, step=None, substep=None):
        """Update the values of the first field of the initial state or of a converged
        substep at the points of the surface (empty, if the first field has no values at
        the points of the mesh) and the limits of the legend (of all points)."""
        values = b""
        limits = None
        if self._displayable:
            field_values = np.asarray(_fields(container)[0].values)
            values = _float32(field_values[self._surface_points])

            # the range of the legend is evaluated by the values at all points
            norm = field_values[:, 0]
            if field_values.shape[1] > 1:
                norm = np.sqrt(np.einsum("ij,ij->i", field_values, field_values))
            limits = [_num(norm.min()), _num(norm.max())]
            if None in limits:
                limits = None

        self._state.frame = dict(
            kind=kind, step=step, substep=substep, values=values, limits=limits
        )

    @staticmethod
    def _regions(job):
        """Return the regions of the items of a job with distinct meshes, supported
        cell types and a common points-array (without boundary regions). If there is
        no such region, the region of the first item with a mesh is returned."""
        from ...region import RegionBoundary

        regions = []
        fallback = None

        for step in job.steps:
            for item in step.items:
                container = getattr(item, "field", None)
                if container is None:
                    continue
                region = getattr(_fields(container)[0], "region", None)
                mesh = getattr(region, "mesh", None)
                if mesh is None:
                    continue
                if fallback is None:
                    fallback = region

                # the mesh of a boundary region holds the cells of the faces and
                # meshes without supported cells (e.g. of the vertex field of a mesh
                # container in a point load) are not drawn
                if isinstance(region, RegionBoundary) or mesh.cell_type not in CORNERS:
                    continue

                # a mesh (or identical cells) of several items, e.g. a solid body and
                # its body force, is used only once
                duplicate = any(
                    mesh is other.mesh
                    or (
                        mesh.cell_type == other.mesh.cell_type
                        and np.shape(mesh.cells) == np.shape(other.mesh.cells)
                        and np.array_equal(mesh.cells, other.mesh.cells)
                    )
                    for other in regions
                )
                if not duplicate:
                    regions.append(region)

        if not regions:
            return [] if fallback is None else [fallback]

        # only the meshes with the points of the first mesh are drawn
        points = regions[0].mesh.points
        return [
            region
            for region in regions
            if region.mesh.points is points
            or (
                np.shape(region.mesh.points) == np.shape(points)
                and np.array_equal(region.mesh.points, points)
            )
        ]

    def _model(self, job):
        "Update the geometry and the properties of the model."
        regions = self._regions(job)
        meshes = [region.mesh for region in regions]

        # a job without a mesh has no geometry (only the solver figures are shown)
        self._points = meshes[0].points if meshes else np.zeros((0, 3))
        self._npoints, self._dim = np.shape(self._points)

        # only the points of the surface are transferred (and their field values)
        surf = surface(meshes)
        self._surface_points, surf = compact(surf, self._npoints)
        points = np.zeros((len(self._surface_points), 3))
        points[:, : self._dim] = self._points[self._surface_points, :3]

        def unique(values):
            return list(dict.fromkeys(values))

        quadratures = [getattr(region, "quadrature", None) for region in regions]
        quadratures = [q for q in quadratures if q is not None]
        names = unique(type(q).__name__ for q in quadratures)
        npoints_quadrature = unique(int(q.npoints) for q in quadratures)
        points_per_cell = unique(int(np.shape(mesh.cells)[1]) for mesh in meshes)

        items = []
        for step in job.steps:
            for item in step.items:
                umat = getattr(item, "umat", None)
                text = type(item).__name__
                if umat is not None:
                    text += f" ({type(umat).__name__})"
                if text not in items:
                    items.append(text)

        nsubsteps = [int(step.nsubsteps) for step in job.steps]

        self._offsets = np.concatenate([[0], np.cumsum(nsubsteps)]).astype(int).tolist()
        self._path = [PENDING] * sum(nsubsteps)
        self._path_changed = True

        widget = self._state
        widget.info = dict(
            title=self.title,
            version=__version__,
            cell_type=", ".join(unique(str(mesh.cell_type) for mesh in meshes)),
            dim=int(self._dim),
            npoints=int(self._npoints),
            ncells=int(sum(len(mesh.cells) for mesh in meshes)),
            points_per_cell=points_per_cell[0] if len(points_per_cell) == 1 else None,
            element=", ".join(
                unique(type(getattr(r, "element", None)).__name__ for r in regions)
            ),
            quadrature=", ".join(names) if names else None,
            quadrature_points_per_cell=(
                npoints_quadrature[0] if len(npoints_quadrature) == 1 else None
            ),
            surface=surf["kind"],
            field_dim=0,
            deform=False,
            ndof=None,
            nsubsteps=nsubsteps,
            items=items,
        )
        widget.points = _float32(points)
        widget.triangles = np.ascontiguousarray(surf["tri"], dtype=np.uint32).tobytes()
        widget.quads = np.ascontiguousarray(surf["quad"], dtype=np.uint32).tobytes()
        widget.lines = np.ascontiguousarray(surf["line"], dtype=np.uint32).tobytes()

    def _unknowns(self, container):
        """Update the first field of the unknowns of the job (which are also the
        unknowns of all items) and its initial values. Only the first field is drawn,
        the other fields may be defined on other meshes (e.g. without coordinates)."""
        fields = _fields(container)
        values = np.asarray(fields[0].values)

        # the first field is drawn if it has values at the points of the mesh, it
        # deforms the model if it has as many components as the points
        self._displayable = values.ndim == 2 and len(values) == self._npoints
        dim = values.shape[1] if self._displayable else 0

        widget = self._state
        with widget.hold_sync():
            widget.info = dict(
                widget.info,
                field_dim=int(dim),
                deform=bool(dim == self._dim and dim > 1),
                ndof=int(sum(np.asarray(f.values).size for f in fields)),
            )
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
        kernel = _kernel()

        if not kernel and self._file is None:
            if not self._warned:
                self._warned = True
                where = (
                    "the IPython terminal, which can not display widgets"
                    if _shell() is not None
                    else "a Python process without a Jupyter kernel, e.g. a script "
                    "in a terminal"
                )
                warnings.warn(
                    "The DashboardWidgetPlugin requires a Jupyter frontend which "
                    "displays widgets, e.g. JupyterLab, Jupyter Notebook or the "
                    "interactive window of VS Code. The dashboard is disabled in "
                    f'{where}. Use DashboardWidgetPlugin(filename="job.html") for a '
                    "live HTML file in the web browser.",
                    stacklevel=_stacklevel(),
                )
            self._active = False
            return

        widget_class = _widget_class() if kernel else None

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
        self._has_unknowns = False
        self._last_reason = None
        self._event = None
        self.overhead = 0.0

        # a new widget for each job, i.e. the outputs of previous cells are kept, or
        # the state of the live HTML file (without a Jupyter frontend)
        self.widget = widget_class() if kernel else None
        self._state = self.widget if kernel else DashboardState()
        self._file_started = False

        with self._state.hold_sync():
            self._model(context.job)
            self._sync(force=True)

        if self._file is not None:
            self._file.start(self._state)
            self._file_started = True

        if kernel:
            _display(self.widget)
        elif not self._opened:
            # the page of the live HTML file is opened once (it follows the next jobs)
            self._opened = True
            webbrowser.open(self._file.html.resolve().as_uri())

        self.overhead += perf_counter() - tic

    def before_step(self, context, state):
        if not self._active:
            return
        self._step = state.stepnumber
        self._substep = None

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

        # the unknowns of the job (and of all items) are known in the hooks of a substep
        if not self._has_unknowns:
            self._has_unknowns = True
            self._unknowns(context.x0)

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

        self._active_dofs = int(len(context.dof1))

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
            failures = f"{self._failures} failed attempt{'s' * (self._failures > 1)}"
            text = f"Step {self._step + 1}, substep {self._substep + 1}: converged "

            # the load factors of the increments of a subdivided substep (cutback)
            if load_factors is None:
                text += f"after {failures}."
            else:
                self._load_factors[self._index()] = [_num(f) for f in load_factors]
                cutbacks = f"{self._failures} cutback{'s' * (self._failures > 1)}"
                text += f"with {cutbacks} in {len(load_factors)} increments."

            self._event = dict(kind="cut", text=text)
        else:
            self._set_path(CONVERGED)

        with self._state.hold_sync():
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
        self._flush(force=True)
        self._active = False
