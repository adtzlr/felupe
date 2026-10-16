# SPDX-License-Identifier: GPL-3.0-or-later

"""A live HTML dashboard for the evaluation of a FElupe job."""

import base64
import json
import math
import sys
import webbrowser
from pathlib import Path
from time import perf_counter

import numpy as np

from .._plugin import Plugin
from ._server import Broadcaster, serve
from ._surface import surface

PAGE = Path(__file__).with_name("dashboard.html")


def _num(value):
    "Return a JSON-compatible float (non-finite values are converted to None)."
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _b64(array, dtype):
    "Return the base64-encoded bytes of an array."
    array = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return base64.b64encode(array.tobytes()).decode("ascii")


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


class DashboardPlugin(Plugin):
    r"""A live HTML dashboard for the evaluation of a :class:`~felupe.Job`.

    The dashboard is served by a small HTTP server (Python standard library only) and
    opened in the web browser. It shows the deformed model, coloured by the norm of
    the values of a field (displacements, temperature, ...), the progress of the job
    including cutbacks, the properties of the model, the convergence of the
    Newton-Raphson method and performance statistics.

    Parameters
    ----------
    iterations : bool, optional
        Stream the non-converged solutions of the Newton-Raphson iterations to the
        viewport (default is False). Only the converged substeps are kept.
    interval : float, optional
        The minimum time in seconds between two streamed iterations (default is
        0.1).
    port : int, optional
        The port of the server. With 0, a free port is chosen (default is 0).
    host : str, optional
        The host of the server (default is "127.0.0.1").
    open_browser : bool, optional
        Open the dashboard in the web browser on the first evaluation (default is
        True).
    filename : str or None, optional
        Save a self-contained HTML file of the dashboard after the job, which replays
        all converged substeps without a server (default is None).
    title : str or None, optional
        The title of the dashboard (default is None).
    history : bool, optional
        Keep the field values of all converged substeps for the timeline of the
        viewport (default is False). Otherwise, only the field values of the latest
        converged substep are kept, i.e. the memory of the dashboard (and the size of
        a saved file) does not grow with the number of substeps.

    Attributes
    ----------
    url : str or None
        The URL of the dashboard, available after the server is started in the hook
        ``before_job``.
    server : http.server.ThreadingHTTPServer or None
        The server of the dashboard. It runs in a daemon thread as long as the Python
        process (or until :meth:`close` is called).
    overhead : float
        The time in seconds spent in the hooks of the dashboard during the last job.

    Notes
    -----
    The server is started in the hook ``before_job`` and the URL is printed. All
    events of a job are kept, i.e. a page which is opened (or reloaded) later shows the
    whole job. A repeated evaluation of the job resets the dashboard.

    ..  list-table:: What the dashboard shows.
        :header-rows: 1

        * - Section
          - Content
        * - Viewport
          - The deformed model (by the first field with ``mesh.dim`` components)
            with banded contours of the norm of a field with values at the points
            of the mesh, an optional non-converged iteration and a timeline over the
            converged substeps.
        * - Load path
          - One cell for each substep, converged, converged with cutbacks (with
            marks at the load factors of its increments) or failed.
        * - Model
          - The points, cells and the cell type of the mesh, the finite element and
            the quadrature of the region with the number of quadrature points,
            degrees of freedom (total, active and prescribed), fields, items and
            materials.
        * - Solver
          - Residual and increment norms, tolerance, step length, attempts,
            iterations, cutbacks and failed attempts, the residual norm of each
            iteration and the iterations of each substep.
        * - Performance
          - Wall time of the assembly, the linear solve, the residual and update
            and the rest, time per iteration, slowest substep, size, nonzeros and
            memory of the sparse matrix and peak memory of the process.

    The times are measured between the hooks of :func:`~felupe.newtonraphson`: the
    assembly between ``before_iteration`` and ``before_linear_solve``, the linear
    solve between ``before_linear_solve`` and ``after_linear_solve`` and the
    residual and update (including a line search) between ``after_linear_solve`` and
    ``after_iteration``.

    The plugin is not callable on purpose. List it before a
    :class:`~felupe.CutbackPlugin` to show the error messages of failed attempts. The
    attempts of a cutback are observed in any order, because they trigger the hooks of
    :func:`~felupe.newtonraphson` again. Cells of a higher order are drawn by their
    corner points.

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

        dashboard = fem.DashboardPlugin(iterations=True, filename="job.html")
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

    def __init__(
        self,
        iterations=False,
        interval=0.1,
        port=0,
        host="127.0.0.1",
        open_browser=True,
        filename=None,
        title=None,
        history=False,
    ):
        self.iterations = bool(iterations)
        self.interval = float(interval)
        self.port = int(port)
        self.host = host
        self.open_browser = open_browser
        self.filename = filename
        self.title = title
        self.history = bool(history)

        self.broadcaster = Broadcaster()
        self.server = None
        self.url = None
        self.overhead = 0.0

        self._active = False
        self._seq = 0
        self._t0 = None
        self._opened = False

    # ------------------------------------------------------------------ utilities

    def _send(self, event, slot=None, **data):
        "Publish an event to all clients."
        self._seq += 1
        data["type"] = event
        data["seq"] = self._seq
        data["t"] = round(perf_counter() - self._t0, 6)
        message = json.dumps(data, separators=(",", ":"), allow_nan=False)
        self.broadcaster.publish(message, slot=slot)

    def _start(self):
        if self.server is None:
            self.server = serve(
                lambda: PAGE.read_bytes(),
                self.broadcaster,
                host=self.host,
                port=self.port,
            )
            host, port = self.server.server_address[:2]
            self.url = f"http://{host}:{port}/"
            print(f"FElupe dashboard: {self.url}")

        if self.open_browser and not self._opened:
            self._opened = True
            try:
                webbrowser.open(self.url)
            except Exception:  # pragma: no cover
                pass

    def close(self):
        "Shut down the server of the dashboard."
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def _frame(self, container, kind, slot=None, **meta):
        "Publish the values of the displayable fields of a field container."
        fields = _fields(container)
        values = {}
        for index in self._display:
            if index < len(fields):
                values[str(index)] = _b64(fields[index].values, np.float32)
        self._send("frame", slot=slot, kind=kind, values=values, **meta)

    def _converged_frame(self, container, kind, **meta):
        "Publish the values of the initial state or of a converged substep."
        if self.history:
            self._frame(container, kind=kind, **meta)
        else:
            # replace the values of the previous substep
            self._frame(container, kind=kind, slot="substep", replace=True, **meta)

    def _model(self, job):
        "Publish the geometry and the properties of the model."
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
            name = getattr(field, "name", None)
            if at_points and (same_mesh or index == 0):
                self._display.append(index)
                if deform < 0 and dim == mesh.dim and dim > 1:
                    deform = index
            if name is None:
                name = "Displacement" if index == deform else f"Field {index}"
            info.append(
                dict(
                    index=index,
                    name=str(name),
                    kind=type(field).__name__,
                    dim=int(dim),
                    size=int(values.size),
                    display=index in self._display,
                )
            )

        quadrature = getattr(region, "quadrature", None)
        npoints_quadrature = getattr(quadrature, "npoints", None)

        steps = []
        for step in job.steps:
            items = []
            for item in step.items:
                umat = getattr(item, "umat", None)
                items.append(
                    dict(
                        kind=type(item).__name__,
                        umat=None if umat is None else type(umat).__name__,
                    )
                )
            steps.append(dict(nsubsteps=int(step.nsubsteps), items=items))

        self._send(
            "model",
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
            history=self.history,
            fields=info,
            deform=deform,
            steps=steps,
            iterations=self.iterations,
            points=_b64(points, np.float32),
            surface=dict(
                kind=surf["kind"],
                tri=_b64(surf["tri"], np.uint32),
                quad=_b64(surf["quad"], np.uint32),
                line=_b64(surf["line"], np.uint32),
            ),
        )
        self._converged_frame(container, kind="initial", step=-1, substep=-1)

    def _close_attempt(self, success, message=None):
        "Finish the current attempt of the Newton-Raphson method."
        if self._attempt_open:
            self._attempt_open = False
            if not success:
                self._failures += 1
                if message is None:
                    message = self._last_reason or "not converged"
            self._send(
                "attempt_end",
                step=self._step,
                substep=self._substep,
                attempt=self._attempt,
                success=bool(success),
                iterations=self._attempt_iterations,
                message=message,
            )

    # ---------------------------------------------------------------------- hooks

    def before_job(self, context, state):
        tic = perf_counter()
        if self._t0 is not None:
            self.broadcaster.reset()
        self._t0 = perf_counter()
        self._seq = 0
        self._active = True
        self._step = None
        self._substep = None
        self._attempt = 0
        self._attempt_open = False
        self._dofs = None
        self._jac = None
        self._last_frame = 0.0
        self._last_reason = None
        self.overhead = 0.0
        self._start()
        self._model(context.job)
        self.overhead += perf_counter() - tic

    def before_step(self, context, state):
        if not self._active:
            return
        self._step = state.stepnumber
        self._send("step_begin", step=state.stepnumber)

    def before_substep(self, context, state):
        if not self._active:
            return
        self._substep = state.substepnumber
        self._attempt = 0
        self._failures = 0
        self._iterations = 0
        self._attempt_open = False
        self._tic_substep = perf_counter()
        self._timing = dict(assembly=0.0, solve=0.0, residual=0.0)
        self._send("substep_begin", step=self._step, substep=self._substep)

    def before_newton(self, context, state):
        if not self._active or self._substep is None:
            return
        tic = perf_counter()

        # an attempt without after_newton has failed (e.g. within a cutback)
        self._close_attempt(success=False)

        self._attempt += 1
        self._attempt_open = True
        self._attempt_iterations = 0
        self._last_reason = None

        dof1 = getattr(context, "dof1", None)
        dof0 = getattr(context, "dof0", None)
        dofs = (
            None if dof1 is None else int(len(dof1)),
            None if dof0 is None else int(len(dof0)),
        )
        if dofs != self._dofs:
            self._dofs = dofs
            self._send("dofs", active=dofs[0], prescribed=dofs[1])

        self._send(
            "attempt_begin",
            step=self._step,
            substep=self._substep,
            attempt=self._attempt,
            tol=_num(getattr(state, "tol", None)),
        )
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
        self._t_assembly = toc - self._tic
        self._timing["assembly"] += self._t_assembly

        tic = perf_counter()
        jac = getattr(state, "jac", None)
        if jac is not None and self._jac is None:
            shape = getattr(jac, "shape", None)
            nnz = getattr(jac, "nnz", None)
            nbytes = 0
            for name in ["data", "indices", "indptr"]:
                nbytes += getattr(getattr(jac, name, None), "nbytes", 0)
            self._jac = True
            self._send(
                "matrix",
                shape=None if shape is None else [int(s) for s in shape],
                nnz=None if nnz is None else int(nnz),
                nbytes=int(nbytes) if nbytes else None,
                format=type(jac).__name__,
            )
        self.overhead += perf_counter() - tic
        self._tic = perf_counter()

    def after_linear_solve(self, context, state):
        if not self._active or self._substep is None:
            return
        toc = perf_counter()
        self._t_solve = toc - self._tic
        self._timing["solve"] += self._t_solve
        self._tic = toc

    def after_iteration(self, context, state):
        if not self._active or self._substep is None:
            return
        toc = perf_counter()
        t_residual = toc - self._tic
        self._timing["residual"] += t_residual

        self._attempt_iterations += 1
        self._iterations += 1

        if getattr(state, "error", False):
            fnorm = _num(state.fnorm)
            self._last_reason = (
                "NaN in the residuals" if fnorm is None else "maximum iterations"
            )

        self._send(
            "iteration",
            step=self._step,
            substep=self._substep,
            attempt=self._attempt,
            iteration=int(state.iteration),
            fnorm=_num(state.fnorm),
            xnorm=_num(state.xnorm),
            alpha=_num(getattr(state, "alpha", None)),
            success=bool(state.success),
            assembly=self._t_assembly,
            solve=self._t_solve,
            residual=t_residual,
        )

        if self.iterations and not state.success:
            now = perf_counter()
            x = getattr(state, "x", None)
            if hasattr(x, "fields") and now - self._last_frame >= self.interval:
                self._last_frame = now
                self._frame(
                    x,
                    kind="iteration",
                    slot="iteration",
                    step=self._step,
                    substep=self._substep,
                    attempt=self._attempt,
                    iteration=int(state.iteration),
                    fnorm=_num(state.fnorm),
                )

        self.overhead += perf_counter() - toc
        self._tic = perf_counter()

    def after_newton(self, context, state):
        if not self._active or self._substep is None:
            return
        self._close_attempt(success=True)

    def after_failed_substep(self, context, state):
        if not self._active:
            return
        message = None
        if state.error is not None:
            message = str(state.error).strip().splitlines()[0][:200]
        self._close_attempt(success=False, message=message)

    def after_substep(self, context, state):
        if not self._active:
            return
        tic = perf_counter()
        self._close_attempt(success=True)

        wall = tic - self._tic_substep
        load_factors = getattr(state, "load_factors", None)
        result = context.substep

        self._converged_frame(
            result.x, kind="substep", step=self._step, substep=self._substep
        )

        self._send(
            "substep_end",
            step=self._step,
            substep=self._substep,
            iterations=self._iterations,
            attempts=self._attempt,
            failures=self._failures,
            load_factors=None
            if load_factors is None
            else [_num(f) for f in load_factors],
            wall=wall,
            assembly=self._timing["assembly"],
            solve=self._timing["solve"],
            residual=self._timing["residual"],
            memory=_peak_memory(),
            overhead=self.overhead,
        )
        self.overhead += perf_counter() - tic

    def after_step(self, context, state):
        if not self._active:
            return
        self._send("step_end", step=state.stepnumber)

    def after_job(self, context, state):
        if not self._active:
            return
        error = getattr(state, "error", None)
        if error is not None:
            self._close_attempt(
                success=False, message=str(error).strip().splitlines()[0][:200]
            )
        self._send(
            "job_end",
            status="failed" if error is not None else "finished",
            message=None
            if error is None
            else f"{type(error).__name__}: {str(error).strip().splitlines()[0][:200]}",
            memory=_peak_memory(),
            overhead=self.overhead,
        )
        self._active = False

        if self.filename is not None:
            self.save(self.filename)

    def save(self, filename):
        """Save a self-contained HTML file of the dashboard, which replays all events
        without a server (without the values of a non-converged iteration)."""
        messages = list(self.broadcaster.history)
        if "substep" in self.broadcaster.slots:
            messages.append(self.broadcaster.slots["substep"][1])
        events = ",\n".join(messages).replace("</", "<\\/")
        page = PAGE.read_text(encoding="utf-8")
        script = f"<script>window.FELUPE_EVENTS = [\n{events}\n];</script>\n"
        page = page.replace("<!--EVENTS-->", script, 1)
        Path(filename).write_text(page, encoding="utf-8")
        return filename
