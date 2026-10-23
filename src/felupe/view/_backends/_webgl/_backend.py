# SPDX-License-Identifier: GPL-3.0-or-later

"""The plotting backend of the scenes by WebGL in the web browser."""

import tempfile
import time
import warnings
import webbrowser
from pathlib import Path

import numpy as np

from ..._scalars import surface_scalars
from ..._shapes import arrows, outline
from ..._surface import polynomial_order, triangulate_surface
from .._base import PlottingBackend
from .._notebook import in_kernel
from ._page import array, iframe, page

DEFAULT_COLOR = "#add8e6"

# temporary pages older than this (in seconds) are removed by the next show()
TEMPORARY_MAX_AGE = 24 * 60 * 60

# the options which were already warned about (each option is warned once)
_WARNED = set()


def _warn(option, name):
    "Warn once about an option which is not supported by the plotting backend."

    if option not in _WARNED:
        _WARNED.add(option)
        warnings.warn(
            f'The option "{option}" is not supported by the plotting backend '
            f'"{name}" and it is ignored.',
            UserWarning,
            stacklevel=4,
        )


def _pad(x):
    "Pad the last axis of the coordinates of points to three components."

    x = np.asarray(x, dtype=float)
    return np.pad(x, ((0, 0),) * (x.ndim - 1) + ((0, 3 - x.shape[-1]),))


def _color(color, default=None):
    """Return a color as hex-string, e.g. ``"#ff0000"`` for ``"red"`` (by matplotlib,
    if available). Other colors are given to the viewer as CSS colors."""

    if color is None:
        return default

    try:
        from matplotlib.colors import to_hex

        return to_hex(color)

    except (ImportError, ValueError):
        return str(color)


def _lut(cmap):
    """Return the lookup table of 256 colors (RGB, uint8) of a colormap of matplotlib,
    or None (then, the viewer takes its built-in colormap "turbo")."""

    try:
        import matplotlib

        colormap = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        colors = colormap(np.linspace(0, 1, 256))[:, :3]

    except (ImportError, KeyError, TypeError, ValueError):
        return None

    return np.round(255 * colors).astype(np.uint8)


def _temporary_page():
    """Return the path of a new temporary page in the folder ``felupe`` of the
    temporary directory and remove the pages there which are older than a day.

    The page is not removed after it is shown, because the web browser loads it later
    (e.g. after a script has ended) and the page should stay reloadable."""

    folder = Path(tempfile.gettempdir()) / "felupe"
    folder.mkdir(exist_ok=True)

    now = time.time()
    for old in folder.glob("plot-*.html"):
        try:
            if now - old.stat().st_mtime > TEMPORARY_MAX_AGE:
                old.unlink()

        # e.g. removed by another process in the meantime or not removable
        except OSError:
            pass

    with tempfile.NamedTemporaryFile(
        dir=folder, prefix="plot-", suffix=".html", delete=False
    ) as file:
        return Path(file.name)


class WebGLPlotter:
    """The plotter of the plotting backend ``"webgl"``, a scene which is shown in the
    web browser by a WebGL viewer.

    Attributes
    ----------
    objects : list of dict
        The objects of the scene, i.e. surfaces, points and lines.
    legend : list of dict or None
        The entries of the legend.
    view : str or None
        The view of the camera, e.g. ``"xy"``. If ``"default"``, planar models are
        viewed from the top (``"xy"``) and other models from an oblique view.
    axes : bool
        A flag to show the axes.

    Notes
    -----
    The page of the viewer is a self-contained HTML page (without a server and without
    other files). Like a plotter of PyVista, the plotter is shown by :meth:`show`. In a
    notebook, the page is shown in the output of the cell (in an ``iframe``, without a
    Jupyter widget, e.g. also in the interactive window of VS Code). Otherwise, the page
    is written to a file, which is opened in the web browser. A Jupyter widget of the
    viewer, e.g. for a layout of widgets, is returned by :meth:`widget`.
    """

    def __init__(self):
        self.objects = []
        self.legend = None
        self.view = "default"
        self.axes = False
        self._views = 0

    def add(self, obj):
        "Add an object (a JSON-serializable dict) to the scene and return the plotter."

        self.objects.append(obj)
        return self

    def scene(self):
        "Return the scene as a JSON-serializable dict."

        return dict(
            objects=self.objects,
            legend=self.legend,
            view=self.view,
            axes=self.axes,
        )

    @property
    def html(self):
        "The self-contained HTML page of the viewer."

        return page(self.scene())

    def export_html(self, filename):
        """Write the self-contained HTML page of the viewer to a file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The filename of the HTML file, e.g. ``"plot.html"``.

        Returns
        -------
        pathlib.Path
            The path of the HTML file.
        """

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html, encoding="utf-8")

        return path

    def widget(self, height="480px"):
        """Return a Jupyter widget of the viewer, e.g. to show it in a layout of
        widgets. The arrays of the scene are transferred as binary buffers. This
        requires the optional dependency anywidget (``pip install felupe[web]``).

        Parameters
        ----------
        height : str, optional
            The height of the widget as CSS length (default is ``"480px"``).

        Returns
        -------
        anywidget.AnyWidget
            The widget of the viewer.
        """

        from ._widget import widget_class

        return widget_class()(self.scene(), height=height)

    def show(self, filename=None):
        """Show the viewer. In a notebook (without a given filename), the page of the
        viewer is shown in the output of the cell (in an ``iframe``). Otherwise, the
        page is written to a file, which is opened in the web browser.

        Parameters
        ----------
        filename : str, pathlib.Path or None, optional
            The filename of the HTML file (default is None). If None, a temporary file
            is written (outside of a notebook) in the folder ``felupe`` of the
            temporary directory, where the pages older than a day are removed.

        Returns
        -------
        pathlib.Path or None
            The path of the HTML file (or None in a notebook).
        """

        if filename is None and in_kernel():
            from IPython.display import display

            # the page in an iframe works in all frontends (also without a manager of
            # widgets), as a raw mime-bundle (an HTML object of an iframe warns)
            page_in_iframe = iframe(page(self.scene(), height="100vh"))
            display({"text/html": page_in_iframe}, raw=True)

            return None

        if filename is None:
            filename = _temporary_page()

        path = self.export_html(filename)
        webbrowser.open(path.resolve().as_uri())

        return path


class WebGLBackend(PlottingBackend):
    """The plotting backend of the scenes by WebGL in the web browser. The plotter is a
    :class:`WebGLPlotter`, which is shown by a self-contained HTML page (e.g. in the
    output of a cell of a notebook or as a file in the web browser). Screenshots are
    not supported.

    Notes
    -----
    The surface of the meshes is triangulated by
    :func:`~felupe.view._surface.triangulate_surface`, i.e. PyVista is not required.
    With ``nonlinear_subdivision=k``, the edges of the faces of quadratic and Lagrange
    cells are subdivided into ``order * 2 ** (k - 1)`` segments, i.e. the faces are
    curved for ``k > 1``. The options of the plots which are specific to PyVista, e.g.
    ``theme``, ``scalar_bar_args``, ``show_logo`` or other keyword-arguments of
    :meth:`pyvista.Plotter.add_mesh`, raise a warning once and they are ignored.

    See Also
    --------
    felupe.view.PlottingBackend : Base class of a plotting backend.
    felupe.set_plotting_backend : Set the current plotting backend by its name.
    """

    def is_plotter(self, plotter):
        "Return True if the given object is a plotter of WebGL."

        return isinstance(plotter, WebGLPlotter)

    def prepare_plotter(
        self, plotter=None, theme="default", off_screen=False, notebook=False
    ):
        """Return the given plotter or a new plotter. The plotter is shown in a
        notebook if the code runs in a notebook (``off_screen`` and ``notebook`` have
        no effect). Themes are not supported."""

        if theme not in (None, "default"):
            _warn("theme", self.name)

        if plotter is None:
            plotter = WebGLPlotter()

        return plotter

    def default_color(self):
        "Return the default color of the meshes."

        return DEFAULT_COLOR

    def add_view(
        self,
        plotter,
        scene,
        name=None,
        component=0,
        label=None,
        factor=1.0,
        show_edges=True,
        show_undeformed=True,
        cmap="turbo",
        view="default",
        scalar_bar_args=None,
        scalar_bar_vertical=False,
        add_axes=True,
        extract_surface=False,
        nonlinear_subdivision=1,
        smooth_shading=True,
        split_sharp_edges=True,
        edge_color="black",
        line_width=1.0,
        show_logo=False,
        **kwargs,
    ):
        """Add the outer surface of the mesh of a scene with scalars, selected by name
        and component, to a plotter. The arguments are documented in
        :meth:`felupe.view.Scene.plot`. Supported keyword-arguments are ``color``,
        ``opacity``, ``style`` (``"surface"`` or ``"wireframe"``) and
        ``show_scalar_bar``."""

        required = ["points", "cells", "cell_type", "point_data", "cell_data"]
        if not all(hasattr(scene, attribute) for attribute in required):
            raise self._not_supported("scenes without points and cells, e.g. ViewXdmf")

        if scalar_bar_args:
            _warn("scalar_bar_args", self.name)

        if show_logo:
            _warn("show_logo", self.name)

        color = kwargs.pop("color", None)
        opacity = kwargs.pop("opacity", 1.0)
        style = kwargs.pop("style", "surface")
        show_scalar_bar = kwargs.pop("show_scalar_bar", True)

        if style not in ("surface", "wireframe"):
            _warn("style", self.name)
            style = "surface"

        for option in kwargs:
            _warn(option, self.name)

        # the curved faces of quadratic and Lagrange cells by subdivisions
        order = polynomial_order(scene)
        subdivisions = order * 2 ** (int(nonlinear_subdivision) - 1) if order > 1 else 1
        surface = triangulate_surface(scene, subdivisions=subdivisions)

        positions = _pad(surface.interpolate(scene.points))
        triangles = surface.triangles
        edges = surface.edges

        displacement = None
        if "Displacement" in scene.point_data:
            displacement = _pad(surface.interpolate(scene.point_data["Displacement"]))

        if show_undeformed and len(triangles) > 0:
            plotter.add(
                dict(
                    type="surface",
                    positions=array(positions),
                    triangles=array(triangles, "u4"),
                    color=DEFAULT_COLOR,
                    opacity=0.2,
                    ghost=True,
                )
            )

        obj = dict(
            type="surface",
            positions=array(positions),
            triangles=array(triangles, "u4"),
            color=_color(color, DEFAULT_COLOR),
            opacity=float(opacity),
            faces=style == "surface",
            smooth=bool(smooth_shading) and int(nonlinear_subdivision) > 1,
            sharp=bool(split_sharp_edges),
        )

        # the faces of linear cells are shaded by a common normal (like polygons)
        if subdivisions == 1 and not obj["smooth"]:
            obj.update(faceIds=array(surface.faces, "u4"))

        if displacement is not None:
            obj.update(displacement=array(displacement), factor=float(factor))

        if show_edges or style == "wireframe":
            obj.update(
                edges=array(edges, "u4"),
                edgeColor=_color(edge_color, "#000000"),
                lineWidth=float(line_width),
            )

        if name is not None:
            values, where, values_range, title = surface_scalars(
                scene, surface, name, component, label
            )
            obj.update(
                values=array(values),
                valuesAt=where,
                range=values_range,
                label=title,
                scalarBar=bool(show_scalar_bar),
                verticalBar=bool(scalar_bar_vertical),
            )

            lut = _lut(cmap)
            if lut is not None:
                obj.update(lut=array(lut.ravel(), "u1"))

        plotter.add(obj)

        # the camera and the axes are taken from the first view
        if plotter._views == 0:
            plotter.view = view
            plotter.axes = bool(add_axes)

        plotter._views += 1

        return plotter

    def add_points(self, plotter, points, color=None, point_size=None, **kwargs):
        """Add points to a plotter. Supported keyword-arguments are ``label`` (for the
        legend) and ``opacity``."""

        label = kwargs.pop("label", None)
        opacity = kwargs.pop("opacity", 1.0)
        kwargs.pop("render_points_as_spheres", None)

        for option in kwargs:
            _warn(option, self.name)

        return plotter.add(
            dict(
                type="points",
                positions=array(_pad(points)),
                color=_color(color, DEFAULT_COLOR),
                size=float(5.0 if point_size is None else point_size),
                opacity=float(opacity),
                legendLabel=label,
            )
        )

    def add_lines(self, plotter, lines, color=None, line_width=None, **kwargs):
        """Add line segments to a plotter. A supported keyword-argument is
        ``opacity``."""

        opacity = kwargs.pop("opacity", 1.0)

        # the properties of the lines of PyVista have no effect here
        for option in ["lighting", "show_edges", "edge_color"]:
            kwargs.pop(option, None)

        for option in kwargs:
            _warn(option, self.name)

        return plotter.add(
            dict(
                type="lines",
                positions=array(_pad(lines).reshape(-1, 3)),
                color=_color(color, "#000000"),
                width=float(1.0 if line_width is None else line_width),
                opacity=float(opacity),
            )
        )

    def add_arrows(self, plotter, points, directions, scale=1.0, color=None, **kwargs):
        """Add arrows to a plotter, given by the points of the triangles of a glyph
        (a cylinder with a cone at the tip). A supported keyword-argument is
        ``opacity``."""

        opacity = kwargs.pop("opacity", 1.0)

        for option in kwargs:
            _warn(option, self.name)

        positions, triangles = arrows(_pad(points), _pad(directions), scale)

        return plotter.add(
            dict(
                type="surface",
                positions=array(positions),
                triangles=array(triangles, "u4"),
                color=_color(color, DEFAULT_COLOR),
                opacity=float(opacity),
                faces=True,
            )
        )

    def add_shape(
        self, plotter, shape, color=None, opacity=None, show_edges=None, **kwargs
    ):
        """Add a shape to a plotter by the triangles of ``shape.triangulate()``. With
        ``show_edges=True``, the boundary edges of the triangles are shown. A supported
        keyword-argument is ``line_width`` (of the edges)."""

        line_width = kwargs.pop("line_width", None)

        for option in kwargs:
            _warn(option, self.name)

        positions, triangles = shape.triangulate()

        if len(triangles) == 0:
            return plotter

        obj = dict(
            type="surface",
            positions=array(positions),
            triangles=array(triangles, "u4"),
            color=_color(color, DEFAULT_COLOR),
            opacity=float(1.0 if opacity is None else opacity),
            faces=True,
        )

        if show_edges:
            obj.update(
                edges=array(outline(triangles), "u4"),
                edgeColor=_color(color, "#000000"),
                lineWidth=float(1.0 if line_width is None else line_width),
            )

        return plotter.add(obj)

    def add_legend(
        self, plotter, entries=None, size=None, background_color=None, **kwargs
    ):
        """Add a legend to a plotter. Without entries, the entries are taken from the
        labels of the points."""

        for option in kwargs:
            _warn(option, self.name)

        if entries is None:
            entries = [
                (obj["legendLabel"], obj["color"])
                for obj in plotter.objects
                if obj.get("legendLabel") is not None
            ]

        plotter.legend = [
            dict(label=str(label).strip(), color=_color(color, "#000000"))
            for label, color in entries
        ]

        return plotter
