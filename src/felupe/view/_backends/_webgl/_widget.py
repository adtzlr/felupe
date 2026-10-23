# SPDX-License-Identifier: GPL-3.0-or-later

"""The Jupyter widget of the WebGL viewer (requires the optional dependency
anywidget)."""

from functools import cache

from ._page import HERE, encode, viewer


def module():
    "Return the source code of the module of the widget (the viewer and its render)."

    return viewer() + (HERE / "widget.js").read_text(encoding="utf-8")


@cache
def widget_class():
    """Return the class of the Jupyter widget of the WebGL viewer. The class is created
    on first use, i.e. anywidget is only imported for a widget."""

    import anywidget
    import traitlets

    class WebGLWidget(anywidget.AnyWidget):
        """The Jupyter widget of the WebGL viewer of a plot. The arrays of the scene are
        transferred as binary buffers.

        Parameters
        ----------
        scene : dict
            The scene of a :class:`WebGLPlotter`.
        height : str, optional
            The height of the widget as CSS length (default is ``"480px"``).
        """

        _esm = module()

        scene = traitlets.Dict().tag(sync=True)
        height = traitlets.Unicode("480px").tag(sync=True)

        def __init__(self, scene, height="480px", **kwargs):
            super().__init__(scene=encode(scene, binary=True), height=height, **kwargs)

        def __repr__(self):
            return "FElupe WebGL viewer (requires a Jupyter frontend with widgets)"

    return WebGLWidget
