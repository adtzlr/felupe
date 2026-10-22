# SPDX-License-Identifier: GPL-3.0-or-later

"""A live HTML file of the dashboard, which is updated by the job and polled by the
page in the web browser (without a server)."""

import base64
import json
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).parent

# the traits of the widget, which are written to the file of the geometry (once per
# job) and to the file of the state (updated during the job)
GEOMETRY = ["points", "triangles", "quads", "lines"]
STATE = ["info", "frame", "path", "progress", "newton"]


class DashboardState:
    """The state of the dashboard without a widget, i.e. the traits of the widget as
    plain attributes (no dependency on anywidget)."""

    def __init__(self):
        self.info = {}
        self.points = b""
        self.triangles = b""
        self.quads = b""
        self.lines = b""
        self.frame = {}
        self.path = {}
        self.progress = {}
        self.newton = {}

    @contextmanager
    def hold_sync(self):
        "A context manager like the one of a widget (without a function)."
        yield


def _encode(value):
    "Encode the binary buffers (bytes) by base64 for JSON."
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _script(name, data):
    "Return a script which assigns the data to a global variable of the page."
    text = json.dumps(data, default=_encode, separators=(",", ":"), allow_nan=False)
    return f"window.{name} = {text};\n"


def _write(path, text, attempts=5, delay=0.05):
    """Write a file atomically (by a temporary file, which replaces the file). A file
    which is read by the web browser may be locked on Windows, then the replacement is
    repeated. Return True if the file is written."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return True
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(delay)
    return False


class DashboardFile:
    """A live HTML file of the dashboard.

    Parameters
    ----------
    filename : str or pathlib.Path
        The filename of the HTML page, e.g. ``"job.html"``.
    interval : float, optional
        The minimum time in seconds between two updates of the state (default is 2.0).

    Notes
    -----
    Next to the page, three scripts are written, e.g. ``job.geometry.js`` (once for
    each job), ``job.state.js`` (updated at most every ``interval`` seconds, if the
    state has changed) and ``job.version.js`` (the versions of the geometry and of the
    state, written last). The page polls the version and loads the geometry and the
    state only if they have changed. A page of a local file can not fetch other files,
    but it can load scripts.
    """

    def __init__(self, filename, interval=2.0):
        self.html = Path(filename)
        self.interval = float(interval)

        stem = self.html.with_suffix("")
        self.files = {
            key: stem.with_name(f"{stem.name}.{key}.js")
            for key in ["geometry", "state", "version"]
        }

        # the id of the geometry is unique for each job, also for the jobs of other
        # processes which write to the same files (e.g. a script which is started
        # again while the page is open)
        self._token = secrets.token_hex(4)
        self.jobs = 0
        self.geometry = None
        self.state = 0
        self.changed = False
        self._written = -float("inf")

    def page(self):
        "Return the HTML page with the inline module and styles of the dashboard."
        module = (HERE / "dashboard.js").read_text(encoding="utf-8")
        styles = (HERE / "dashboard.css").read_text(encoding="utf-8")
        config = json.dumps({key: path.name for key, path in self.files.items()})
        page = (HERE / "dashboard.html").read_text(encoding="utf-8")
        page = page.replace("/*STYLES*/", styles)
        page = page.replace("/*CONFIG*/", config)
        return page.replace("/*MODULE*/", module.replace("</script", "<\\/script"))

    def start(self, source):
        "Write the page and the geometry of a new job (and its initial state)."
        self.html.parent.mkdir(parents=True, exist_ok=True)
        _write(self.html, self.page())

        self.jobs += 1
        self.geometry = f"{self._token}-{self.jobs}"
        data = {key: getattr(source, key) for key in GEOMETRY}
        _write(
            self.files["geometry"],
            _script("FELUPE_DASHBOARD_GEOMETRY", {"id": self.geometry, **data}),
        )

        self.changed = True
        self.flush(source, force=True)

    def flush(self, source, force=False):
        """Write the state and the version if the state has changed, at most every
        ``interval`` seconds (unless forced). Return True if the state is written."""
        now = time.monotonic()
        if not self.changed or (not force and now - self._written < self.interval):
            return False

        state = self.state + 1
        data = {key: getattr(source, key) for key in STATE}
        data.update(version=state, geometry=self.geometry, written=time.time())

        if not _write(self.files["state"], _script("FELUPE_DASHBOARD_STATE", data)):
            return False

        # the version is written last, i.e. the page loads a complete state (if the
        # version can not be written, it is written by the next update)
        version = {"geometry": self.geometry, "state": state}
        script = _script("FELUPE_DASHBOARD_VERSION", version)

        self.state = state
        self.changed = not _write(self.files["version"], script)
        self._written = now
        return True
