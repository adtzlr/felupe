# SPDX-License-Identifier: GPL-3.0-or-later

"""The Jupyter widget of the dashboard (requires the optional dependency anywidget)."""

from pathlib import Path

import anywidget
import traitlets

from ._file import module

HERE = Path(__file__).parent


class DashboardWidget(anywidget.AnyWidget):
    """The Jupyter widget of a :class:`~felupe.DashboardWidgetPlugin`.

    The state of the dashboard is kept in synchronized traits. Hence, a view of the
    widget which is rendered again (e.g. after scrolling or collapsing the output of a
    cell) shows the latest state. The field values and the geometry are transferred as
    binary buffers. The model is drawn by the WebGL viewer of the plotting backend
    ``"webgl"``, which is a part of the module of the widget.
    """

    _esm = module()
    _css = HERE / "dashboard.css"

    # the properties of the model, the fields and the steps of the job
    info = traitlets.Dict().tag(sync=True)

    # the points of the triangulated outer surface (float32, 3 components), its
    # triangles, the faces of the triangles (for the shading of linear cells), the
    # edges of the faces and the line cells (uint32)
    points = traitlets.Bytes(b"").tag(sync=True)
    triangles = traitlets.Bytes(b"").tag(sync=True)
    faces = traitlets.Bytes(b"").tag(sync=True)
    edges = traitlets.Bytes(b"").tag(sync=True)
    lines = traitlets.Bytes(b"").tag(sync=True)

    # the field values (float32) of the initial state or of the latest converged substep
    frame = traitlets.Dict().tag(sync=True)

    # the status of each substep and the load factors of subdivided substeps
    path = traitlets.Dict().tag(sync=True)

    # the solver figures and the performance statistics of the job
    progress = traitlets.Dict().tag(sync=True)

    # the residual norms of the attempts of the current substep
    newton = traitlets.Dict().tag(sync=True)

    def __repr__(self):
        return "FElupe dashboard (requires a Jupyter frontend with widget support)"
