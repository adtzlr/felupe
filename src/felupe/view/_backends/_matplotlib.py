# SPDX-License-Identifier: GPL-3.0-or-later

"""The plotting backend of the scenes by matplotlib, e.g. for off-screen screenshots
in raster or vector formats (PNG, PDF, SVG or EPS) without a display."""

import io
import sys
import warnings
import weakref
from pathlib import Path

import numpy as np

from .._scalars import surface_scalars
from .._shapes import arrows, outline
from .._surface import CORNERS, FACES, polynomial_order, triangulate_surface
from ._base import PlottingBackend
from ._notebook import in_kernel

DEFAULT_COLOR = "#add8e6"

# surfaces with at least this opacity are drawn as opaque surfaces, e.g. the meshes of
# a mesh container, which are drawn with an opacity of 0.99 by default
OPAQUE = 0.99

# the field of view of the perspective projection in degrees (like the WebGL viewer)
FIELD_OF_VIEW = 30.0

# the line width (in points) of the strokes in the colors of the faces, which close the
# gaps between adjacent faces by antialiasing (about a pixel at 100 dots per inch)
SEAM = 0.75

# the size of a pixel in points, i.e. sizes in pixels (e.g. of PyVista) are converted
PIXEL = 0.75

# the margin of the model in the axes, relative to its larger extent
MARGIN = 0.05

# the ranks of the items at equal depth: the faces are drawn first, then the outlines
# of the faces, the lines and the points
FACE, OUTLINE, LINE, POINT = 0, 1, 2, 3

# the direction of the key light in the camera coordinates (like the WebGL viewer)
KEY_LIGHT = np.array([0.2, 0.7, 0.7]) / np.linalg.norm([0.2, 0.7, 0.7])

# the colors of the axes x, y and z (like the WebGL viewer)
AXES_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]

# the options which were already warned about (each option is warned once)
_WARNED = set()

# the plotters of given axes of matplotlib
_PLOTTERS = weakref.WeakKeyDictionary()


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


def _rgba(color, alpha=1.0):
    "Return a color as RGBA-array with a given alpha (multiplied by its own alpha)."

    from matplotlib.colors import to_rgba

    rgba = np.array(to_rgba(color), dtype=float)
    rgba[3] *= alpha

    return rgba


def _colormap(cmap):
    "Return a colormap of matplotlib by its name or the given colormap."

    import matplotlib

    return matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap


def _map_colors(values, cmap, values_range):
    """Return the RGB-colors of values by a colormap in a range. A uniform range is
    drawn in the color of the middle of the colormap and values which are not finite
    in grey."""

    values = np.asarray(values, dtype=float)
    a, b = values_range if values_range is not None else (0.0, 0.0)

    if b > a:
        t = np.clip((values - a) / (b - a), 0, 1)
    else:
        t = np.full(len(values), 0.5)

    finite = np.isfinite(values)
    t[~finite] = 0.5
    rgb = cmap(t)[:, :3]
    rgb[~finite] = 0.6

    return rgb


def _format_number(x):
    "Return the label of a tick of a scalar bar (like the WebGL viewer)."

    if x == 0 or not np.isfinite(x):
        return f"{x:g}"

    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.2e}"

    return f"{float(f'{x:.3g}'):g}"


def _oblique_view(azimuth, elevation):
    """Return the camera basis (right, up, back) of a view from a direction, given by
    its azimuth and elevation in degrees, with the z-axis upwards."""

    az, el = np.radians(azimuth), np.radians(elevation)
    back = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)

    return np.array([right, np.cross(back, right), back])


# the camera basis (right, up, back) of the views, like the WebGL viewer
VIEWS = {
    "xy": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    "yx": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float),
    "xz": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
    "zx": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
    "yz": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    "zy": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
    "iso": _oblique_view(45, 35.264),
}

# the default view of the three-dimensional models of FElupe (the isometric view of
# PyVista, rotated by an azimuth of -100 and an elevation of -15 degrees)
DEFAULT_VIEW = _oblique_view(-55, 20.264)


class _Camera:
    """A camera, which looks at a target from a direction (the back-vector of the
    basis) with a parallel or a perspective projection."""

    def __init__(self, basis, target, radius, parallel):
        self.basis = np.asarray(basis, dtype=float)
        self.parallel = bool(parallel)

        # the bounding sphere of the scene fits into the field of view
        distance = radius / np.sin(np.radians(FIELD_OF_VIEW) / 2)
        self.eye = np.asarray(target, dtype=float) + distance * self.basis[2]

    def project(self, points):
        """Return the projected coordinates of shape ``(npoints, 2)`` and the depths of
        shape ``(npoints,)`` (the distances along the direction of view) of points of
        shape ``(npoints, 3)``."""

        local = (np.asarray(points, dtype=float).reshape(-1, 3) - self.eye) @ (
            self.basis.T
        )
        depth = -local[:, 2]

        if self.parallel:
            return local[:, :2], depth

        return local[:, :2] / np.maximum(depth, 1e-300)[:, None], depth

    def shade(self, colors, normals):
        """Return the RGB-colors shaded by the normals of the faces (in the global
        coordinates) by a key light and a headlight, lit on both sides (like the WebGL
        viewer, without specular highlights). The light is normalized, i.e. faces
        towards the camera are drawn in their colors, e.g. of the colormap."""

        n = normals @ self.basis.T
        length = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, length, out=np.zeros_like(n), where=length > 0)

        key = np.abs(n @ KEY_LIGHT)
        head = np.abs(n[:, 2])
        light = (0.25 + 0.55 * key + 0.25 * head) / (0.5 + 0.55 * KEY_LIGHT[2])

        return np.clip(colors * light[:, None], 0, 1)


def _signed_areas(polygons):
    """Return the signed areas of projected polygons of shape ``(n, k, 2)``, positive
    for a counter-clockwise order of the points."""

    x, y = polygons[..., 0], polygons[..., 1]
    return 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)


def _normals(polygons):
    "Return the (unnormalized) normal vectors of polygons of shape ``(n, k, 3)``."

    centered = polygons - polygons.mean(axis=1, keepdims=True)
    return np.cross(centered, np.roll(centered, -1, axis=1)).sum(axis=1)


def _polygons(triangles, faces, subdivisions):
    """Return the polygons of a triangulated surface, i.e. the two triangles of each
    (subdivided) quad of a face are merged into a quad, with their faces and the first
    triangle of each polygon. A list of the triangles and the quads is returned.

    The faces are contiguous blocks of triangles: a quad face with ``n`` subdivisions
    has the triangles ``(p, q, r)`` of its ``n**2`` quads followed by their triangles
    ``(p, r, s)``, a triangle face has ``n**2`` triangles (see
    :func:`~felupe.view._surface.triangulate_surface`). Triangles which do not match,
    e.g. of other faces, are kept."""

    n = int(subdivisions)
    triangles = np.asarray(triangles).reshape(-1, 3)
    faces = np.asarray(faces)

    if len(triangles) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return [(np.zeros((0, 3), dtype=np.int64), empty, empty)]

    counts = np.bincount(faces)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    local = np.arange(len(triangles)) - starts[faces]

    first = np.flatnonzero((counts[faces] == 2 * n**2) & (local < n**2))
    second = first + n**2
    t0, t1 = triangles[first], triangles[second]
    match = (t1[:, 0] == t0[:, 0]) & (t1[:, 1] == t0[:, 2]) & (t1[:, 2] != t0[:, 1])
    first, second = first[match], second[match]

    merged = np.zeros(len(triangles), dtype=bool)
    merged[first] = True
    merged[second] = True
    single = np.flatnonzero(~merged)

    quads = np.column_stack([triangles[first], triangles[second, 2]])

    return [
        (triangles[single], faces[single], single),
        (quads, faces[first], first),
    ]


def _face_outlines(triangles, faces):
    """Return the line segments of shape ``(nsegments, 2)`` of the outlines of the
    faces of a triangulated surface and the faces of the line segments, i.e. the edges
    of the triangles which belong to only one triangle of a face."""

    triangles = np.asarray(triangles).reshape(-1, 3)
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    owners = np.tile(faces, 3)

    keys = np.column_stack([owners, np.sort(edges, axis=1)])
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    single = counts[inverse.ravel()] == 1

    return edges[single], owners[single]


def _closed_paths(polygons):
    "Return the closed paths of polygons of shape ``(n, k, 2)``."

    from matplotlib.path import Path

    k = polygons.shape[1]
    vertices = np.concatenate([polygons, polygons[:, :1]], axis=1)
    codes = np.full(k + 1, Path.LINETO, dtype=Path.code_type)
    codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY

    return [Path(v, codes) for v in vertices]


def _open_paths(segments):
    "Return the paths of line segments of shape ``(n, 2, 2)``."

    from matplotlib.path import Path

    return [Path(v) for v in segments]


def _segments_path(segments):
    "Return one path of several line segments of shape ``(n, 2, 2)``."

    from matplotlib.path import Path

    codes = np.tile(
        np.array([Path.MOVETO, Path.LINETO], dtype=Path.code_type), len(segments)
    )

    return Path(segments.reshape(-1, 2), codes)


class _Items:
    """The items of a scene (paths with colors and line widths), which are drawn in
    the order of their depths (painter's algorithm): far items first."""

    def __init__(self):
        self.paths = []
        self.depths = []
        self.ranks = []
        self.order = []
        self.faces = []
        self.edges = []
        self.widths = []

    def add(self, paths, depths, rank, order, face, edge, width):
        "Add paths with their depths, rank, order, face- and edge-colors and widths."

        n = len(paths)

        if n == 0:
            return

        self.paths.extend(paths)
        self.depths.append(np.broadcast_to(np.asarray(depths, dtype=float), (n,)))
        self.ranks.append(np.full(n, rank))
        self.order.append(np.full(n, order))
        self.faces.append(np.broadcast_to(np.asarray(face, dtype=float), (n, 4)))
        self.edges.append(np.broadcast_to(np.asarray(edge, dtype=float), (n, 4)))
        self.widths.append(np.broadcast_to(np.asarray(width, dtype=float), (n,)))

    def collection(self):
        """Return a collection of the paths, sorted by their depths (far items first).
        Items of equal depth are sorted by their rank and then by the order of their
        objects."""

        from matplotlib.collections import PathCollection

        if not self.paths:
            return PathCollection([])

        depths = np.concatenate(self.depths)
        ranks = np.concatenate(self.ranks)
        order = np.concatenate(self.order)
        indices = np.lexsort((order, ranks, -depths))

        collection = PathCollection(
            [self.paths[i] for i in indices],
            facecolors=np.concatenate(self.faces)[indices],
            edgecolors=np.concatenate(self.edges)[indices],
            linewidths=np.concatenate(self.widths)[indices],
        )
        collection.set_capstyle("round")
        collection.set_joinstyle("round")

        return collection


class MatplotlibPlotter:
    r"""The plotter of the plotting backend ``"matplotlib"``, a scene which is drawn on
    the axes of a figure of matplotlib.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        The (two-dimensional) axes of matplotlib to draw on (default is None). If None,
        a new figure with axes is created. A scene on given axes is drawn whenever an
        object is added, i.e. the figure of the axes may be saved by
        ``fig.savefig()``, e.g. a figure with several subplots in a vector format.
    off_screen : bool, optional
        A flag to create the new figure without pyplot (default is False), i.e. it is
        not shown in a window, e.g. for screenshots. In a notebook, the figure is
        always created without pyplot and it is shown by :meth:`show`.

    Attributes
    ----------
    objects : list of dict
        The objects of the scene, i.e. surfaces, points and lines.
    legend : list of tuple or None
        The entries ``(label, color)`` of the legend.
    view : str or None
        The view of the camera, e.g. ``"xy"``, ``"xz"`` or ``"iso"``. If
        ``"default"``, planar models are viewed from the top (``"xy"``) and other
        models from an oblique view (the default view of FElupe).
    axes : bool
        A flag to show the axes.
    parallel_projection : bool or None
        A flag for a parallel projection (default is None). If None, planar models are
        drawn by a parallel projection and other models by a perspective projection.
    rasterized : bool
        A flag to rasterize the model in vector formats (default is False), e.g. for
        large models. The scalar bars, the legend and the axes are not rasterized.

    Notes
    -----
    The model is drawn by projected polygons (with a flat color each), which are drawn
    in the order of their depths (painter's algorithm), i.e. far polygons first. The
    faces on the back of the closed surfaces of opaque solid bodies are not drawn
    (back-face culling). Hence, a screenshot in a vector format, e.g. PDF or SVG,
    contains a polygon for each visible face (with its edges), see
    :meth:`screenshot`. Overlapping faces, which intersect each other, e.g. at a
    contact, may be drawn in a wrong order.

    Examples
    --------
    >>> import felupe as fem
    >>> import matplotlib.pyplot as plt
    >>>
    >>> fig, ax = plt.subplots(1, 2)
    >>> plotter = fem.view.MatplotlibPlotter(ax=ax[0])
    >>> plotter = fem.Cube(n=3).plot(plotter=plotter)
    >>> plotter = fem.Rectangle(n=3).plot(plotter=ax[1])
    >>> fig.savefig("meshes.pdf")
    """

    def __init__(self, ax=None, off_screen=False):
        self.objects = []
        self.legend = None
        self.legend_background = None
        self.view = "default"
        self.axes = False
        self.parallel_projection = None
        self.rasterized = False
        self._views = 0
        self._artists = []
        self._stale = True
        self._eager = ax is not None

        if ax is None:
            if off_screen or in_kernel():
                from matplotlib.figure import Figure

                ax = Figure().add_subplot()
            else:
                import matplotlib.pyplot as plt

                _, ax = plt.subplots()

        self._ax = ax
        _PLOTTERS[ax] = self

    @property
    def ax(self):
        "The axes of matplotlib with the drawn scene."

        if self._stale:
            self.render()

        return self._ax

    @property
    def figure(self):
        "The (root) figure of matplotlib with the drawn scene."

        return self.ax.figure.figure

    def _drawn_figure(self):
        "Draw the scene (e.g. with changed attributes) and return the (root) figure."

        return self.render()._ax.figure.figure

    def add(self, obj):
        """Add an object (a dict) to the scene and return the plotter. The scene is drawn
        again on given axes, otherwise it is drawn on the next access of the axes or
        the figure, e.g. by :meth:`show` or :meth:`screenshot`."""

        self.objects.append(obj)
        self._stale = True

        if self._eager:
            self.render()

        return self

    def enable_parallel_projection(self):
        "Enable the parallel projection of the camera and return the plotter."

        self.parallel_projection = True
        self._stale = True

        if self._eager:
            self.render()

        return self

    def _camera(self):
        "Return the camera of the scene by the view and the bounds of the objects."

        points = [
            obj["positions"].reshape(-1, 3)
            for obj in self.objects
            if len(obj["positions"]) > 0
        ]

        if points:
            points = np.vstack(points)
            lower, upper = points.min(axis=0), points.max(axis=0)
        else:
            lower, upper = -np.ones(3), np.ones(3)

        extent = upper - lower
        planar = extent[2] <= 1e-9 * max(extent.max(), 1e-300)

        if self.view in VIEWS:
            basis = VIEWS[self.view]
        else:
            basis = VIEWS["xy"] if planar else DEFAULT_VIEW

        parallel = (
            planar if self.parallel_projection is None else self.parallel_projection
        )
        radius = max(0.5 * np.linalg.norm(extent), 1e-12)

        return _Camera(basis, (lower + upper) / 2, radius, parallel)

    def _bias(self):
        """Return the bias of the depths of the points and lines towards the camera,
        i.e. the median length of the edges of the faces of the surfaces. Then, points
        and lines on a surface are drawn after the faces of the surface."""

        lengths = []
        for obj in self.objects:
            if obj["type"] == "surface":
                for polygons, *_ in obj["polygons"]:
                    if len(polygons) > 0:
                        x = obj["positions"]
                        edges = x[polygons[:, 1]] - x[polygons[:, 0]]
                        lengths.append(np.linalg.norm(edges, axis=1))

        if not lengths:
            return 0.0

        return float(np.median(np.concatenate(lengths)))

    def render(self):
        """Draw the scene on the axes, i.e. the model as a collection of paths, the
        scalar bars, the legend and the axes. A previously drawn scene is removed. The
        scene is drawn by :meth:`show`, :meth:`screenshot` and :meth:`savefig`, e.g.
        with changed attributes like the view.

        Returns
        -------
        MatplotlibPlotter
            The plotter.
        """

        ax = self._ax

        for artist in self._artists:
            artist.remove()

        self._artists = []
        self._stale = False

        camera = self._camera()
        bias = self._bias()
        items = _Items()
        points = []
        projected = []

        for order, obj in enumerate(self.objects):
            uv, depth = camera.project(obj["positions"])
            projected.append(uv)

            if obj["type"] == "surface":
                self._add_surface(items, obj, order, camera, uv, depth, bias)

            elif obj["type"] == "lines":
                segments = uv.reshape(-1, 2, 2)
                depths = depth.reshape(-1, 2).mean(axis=1) - bias
                items.add(
                    _open_paths(segments),
                    depths,
                    LINE,
                    order,
                    (0, 0, 0, 0),
                    obj["color"],
                    obj["width"] * PIXEL,
                )

            elif obj["type"] == "points":
                points.append((obj, order, uv, depth - bias))

        bars = self._scalar_bars()
        vertical = any(bar["vertical"] for bar in bars)
        scale = self._frame(np.vstack(projected or [np.zeros((0, 2))]), bars, vertical)

        from matplotlib.path import Path

        circle = Path.unit_circle()
        for obj, order, uv, depths in points:
            radius = 0.5 * obj["size"] * PIXEL * scale
            paths = [
                Path(center + radius * circle.vertices, circle.codes) for center in uv
            ]
            items.add(paths, depths, POINT, order, obj["color"], (0, 0, 0, 0), 0.0)

        collection = items.collection()
        collection.set_rasterized(self.rasterized)
        ax.add_collection(collection, autolim=False)
        self._artists.append(collection)
        self._collection = collection

        self._draw_scalar_bars(bars, vertical)
        self._draw_legend()

        if self.axes:
            self._draw_axes(camera)

        return self

    def _add_surface(self, items, obj, order, camera, uv, depth, bias):
        "Add the items of a surface, i.e. its faces, the outlines and the lines."

        positions = obj["positions"]
        nfaces = obj["nfaces"]
        nearest = np.full(nfaces, np.inf)
        opacity = obj["opacity"]
        edge = obj["edge_color"]
        width = obj["line_width"] * PIXEL

        for polygons, faces, colors in obj["polygons"]:
            if len(polygons) == 0:
                continue

            projected = uv[polygons]
            depths = depth[polygons].mean(axis=1)

            # the undeformed model is pushed back, i.e. behind (nearly) coincident faces
            if obj.get("ghost", False):
                depths = depths + bias

            # back-face culling of the closed surfaces of opaque solid bodies
            if obj["cull"]:
                front = _signed_areas(projected) > 0
                polygons, faces, colors = polygons[front], faces[front], colors[front]
                projected, depths = projected[front], depths[front]

            np.minimum.at(nearest, faces, depths)

            if not (obj["fill"] or obj["stroke"]) or len(polygons) == 0:
                continue

            if obj["lighting"]:
                colors = camera.shade(colors, _normals(positions[polygons]))

            face = np.column_stack([colors, np.full(len(colors), opacity)])

            if obj["stroke"]:
                edges, widths = edge, width
            elif obj["fill"] and opacity >= OPAQUE:
                edges, widths = face, SEAM
            else:
                edges, widths = (0, 0, 0, 0), 0.0

            if not obj["fill"]:
                face = (0, 0, 0, 0)

            items.add(
                _closed_paths(projected), depths, FACE, order, face, edges, widths
            )

        # the outlines of the faces (with at least one visible polygon)
        if obj["outlines"] is not None:
            segments, faces = obj["outlines"]
            visible = np.isfinite(nearest[faces])
            segments, faces = segments[visible], faces[visible]

            sort = np.argsort(faces, kind="stable")
            segments, faces = segments[sort], faces[sort]
            unique, starts = np.unique(faces, return_index=True)

            paths = [
                _segments_path(group)
                for group in np.split(uv[segments], starts[1:])
                if len(group) > 0
            ]
            items.add(paths, nearest[unique], OUTLINE, order, (0, 0, 0, 0), edge, width)

        # the line cells
        if obj["lines"] is not None:
            lines, colors = obj["lines"]
            edges = np.column_stack([colors, np.full(len(colors), opacity)])
            items.add(
                _open_paths(uv[lines]),
                depth[lines].mean(axis=1) - bias,
                LINE,
                order,
                (0, 0, 0, 0),
                edges,
                width,
            )

    def _scalar_bars(self):
        "Return the scalar bars of the objects, one for each label of the scalars."

        bars = []
        labels = set()

        for obj in self.objects:
            bar = obj.get("scalar_bar")

            if bar is not None and bar["label"] not in labels:
                labels.add(bar["label"])
                bars.append(bar)

        return bars

    def _frame(self, projected, bars, vertical):
        """Set the limits of the axes to the projected points (with a margin and with
        space for the scalar bars) with an equal aspect ratio and return the size of a
        point in the units of the data."""

        ax = self._ax

        if len(projected) > 0:
            lower, upper = projected.min(axis=0), projected.max(axis=0)
        else:
            lower, upper = -np.ones(2), np.ones(2)

        size = max((upper - lower).max(), 1e-300)
        lower = lower - MARGIN * size
        upper = upper + MARGIN * size

        # the space of the scalar bars
        if bars:
            width, height = upper - lower
            if vertical:
                fraction = 0.13 * len(bars) + 0.04
                upper[0] += fraction / (1 - fraction) * width
            else:
                fraction = 0.2
                lower[1] -= fraction / (1 - fraction) * height

        # the equal aspect ratio of the axes
        box = ax.get_window_extent()
        aspect = box.height / max(box.width, 1e-300)
        width, height = upper - lower

        if height < aspect * width:
            grow = aspect * width - height
            lower[1] -= grow / 2
            upper[1] += grow / 2
        else:
            grow = height / aspect - width
            lower[0] -= grow / 2
            upper[0] += grow / 2

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()

        points = box.width / ax.figure.dpi * 72

        return (upper[0] - lower[0]) / max(points, 1e-300)

    def _draw_scalar_bars(self, bars, vertical):
        "Draw the scalar bars as inset axes at the bottom or at the right."

        from matplotlib.cm import ScalarMappable
        from matplotlib.colorbar import Colorbar
        from matplotlib.colors import ListedColormap, Normalize

        ax = self._ax
        nbars = len(bars)

        for i, bar in enumerate(bars):
            if vertical:
                bounds = [1 - 0.13 * (nbars - i) + 0.01, 0.2, 0.03, 0.6]
            else:
                total = 0.5 if nbars == 1 else 0.8
                width = (total - 0.05 * (nbars - 1)) / nbars
                bounds = [0.5 - total / 2 + i * (width + 0.05), 0.06, width, 0.035]

            cax = ax.inset_axes(bounds)
            self._artists.append(cax)

            a, b = bar["range"] if bar["range"] is not None else (0.0, 0.0)
            cmap = bar["cmap"]
            ticks = np.linspace(a, b, bar["n_labels"])

            # a uniform range is drawn in the color of its value
            if not b > a:
                cmap = ListedColormap([cmap(0.5)])
                a, b = a - 0.5, a + 0.5
                ticks = [(a + b) / 2]

            mappable = ScalarMappable(norm=Normalize(a, b), cmap=cmap)
            colorbar = Colorbar(
                cax, mappable, orientation="vertical" if vertical else "horizontal"
            )

            fmt = bar["fmt"]
            labels = [fmt % t if fmt else _format_number(t) for t in ticks]
            colorbar.set_ticks(ticks, labels=labels)
            colorbar.outline.set_linewidth(0.5)
            cax.tick_params(labelsize=8, width=0.5)
            cax.set_title(bar["label"], fontsize=9, fontweight="bold")

    def _draw_legend(self):
        "Draw the legend in the upper right corner of the axes."

        if not self.legend:
            return

        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=color, edgecolor="none", label=label)
            for label, color in self.legend
        ]
        background = self.legend_background
        legend = self._ax.legend(
            handles=handles,
            loc="upper right",
            fontsize="small",
            facecolor="white" if background is None else background,
        )
        self._artists.append(legend)

    def _draw_axes(self, camera):
        """Draw the axes x, y and z of the view at the lower left corner of the axes
        (like the WebGL viewer)."""

        ax = self._ax
        box = ax.get_window_extent()
        size = 0.6 * ax.figure.dpi
        inset = ax.inset_axes(
            [0, 0, size / max(box.width, 1e-300), size / max(box.height, 1e-300)]
        )
        self._artists.append(inset)

        inset.set_xlim(-1, 1)
        inset.set_ylim(-1, 1)
        inset.set_aspect("equal")
        inset.set_axis_off()

        right, up, back = camera.basis
        length = 0.64
        order = np.argsort(back)

        for axis in order:
            x, y = right[axis], up[axis]
            color = AXES_COLORS[axis]
            name = "xyz"[axis]

            # an axis along the direction of view is labelled at the lower left
            if np.hypot(x, y) < 0.05:
                inset.text(
                    -0.25 * length,
                    -0.25 * length,
                    name,
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                continue

            inset.plot([0, length * x], [0, length * y], color=color, lw=1.5)
            inset.text(
                1.3 * length * x,
                1.3 * length * y,
                name,
                color=color,
                ha="center",
                va="center",
                fontsize=8,
            )

    def show(self):
        """Show the figure. A figure of pyplot is shown by :func:`pyplot.show()
        <matplotlib.pyplot.show>`. In a notebook, the image of the figure is shown in
        the output of the cell. An off-screen figure (outside of a notebook) is not
        shown.

        Returns
        -------
        None
        """

        figure = self._drawn_figure()

        if getattr(figure.canvas, "manager", None) is not None:
            import matplotlib.pyplot as plt

            plt.show()

        elif in_kernel():
            from IPython.display import Image, display

            buffer = io.BytesIO()
            figure.savefig(buffer, format="png")
            display(Image(data=buffer.getvalue()))

        return None

    def savefig(self, *args, **kwargs):
        """Save the figure, see :meth:`matplotlib.figure.Figure.savefig`, e.g.
        ``plotter.savefig("model.pdf")``."""

        return self._drawn_figure().savefig(*args, **kwargs)

    def screenshot(self, filename=None, transparent_background=None, scale=None):
        """Take a screenshot of the figure and return the image data. Optionally, the
        screenshot is saved to a file. The format of the file is given by the suffix of
        the filename, e.g. ``"model.png"`` (raster) or ``"model.pdf"``,
        ``"model.svg"`` and ``"model.eps"`` (vector formats).

        Parameters
        ----------
        filename : str, pathlib.Path or None, optional
            The filename of the screenshot (default is None). If None, no file is
            written.
        transparent_background : bool or None, optional
            A flag for a transparent background (default is None). If True, the image
            data has an alpha channel.
        scale : float or None, optional
            A factor of the resolution (the dots per inch) of the figure (default is
            None).

        Returns
        -------
        ndarray
            The image data of the screenshot of shape ``(height, width, 3)``, or
            ``(height, width, 4)`` with a transparent background, as unsigned integers
            (``uint8``).
        """

        from matplotlib.image import imread

        figure = self._drawn_figure()
        dpi = figure.dpi * (1.0 if scale is None else float(scale))
        transparent = bool(transparent_background)

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, transparent=transparent)

        if filename is not None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.suffix.lower() == ".png":
                path.write_bytes(buffer.getvalue())
            else:
                figure.savefig(path, dpi=dpi, transparent=transparent)

        buffer.seek(0)
        image = np.round(255 * imread(buffer, format="png")).astype(np.uint8)

        return image if transparent else image[..., :3]

    def close(self):
        "Close the figure of pyplot (if the figure is a figure of pyplot)."

        figure = self._ax.figure.figure

        if getattr(figure.canvas, "manager", None) is not None:
            import matplotlib.pyplot as plt

            plt.close(figure)


def _as_plotter(plotter):
    "Return the plotter of given axes of matplotlib or the given plotter."

    if isinstance(plotter, MatplotlibPlotter):
        return plotter

    if not MatplotlibBackend().is_plotter(plotter):
        raise TypeError(
            "The plotter must be a plotter of matplotlib or two-dimensional axes of "
            f"matplotlib, got {type(plotter).__name__}."
        )

    if plotter not in _PLOTTERS:
        MatplotlibPlotter(ax=plotter)

    return _PLOTTERS[plotter]


class MatplotlibBackend(PlottingBackend):
    """The plotting backend of the scenes by matplotlib. The plotter is a
    :class:`~felupe.view.MatplotlibPlotter`, which draws on (given) axes of
    matplotlib. Screenshots are supported, also off-screen (without a display) and in
    vector formats, e.g. PDF or SVG.

    Notes
    -----
    The outer surface of the meshes is triangulated by
    :func:`~felupe.view._surface.triangulate_surface`, i.e. PyVista is not required.
    The (subdivided) faces are drawn by polygons with a flat color each. With
    ``nonlinear_subdivision=k``, the edges of the faces are subdivided into
    ``order * 2 ** (k - 1)`` segments, also of linear cells, i.e. the faces of
    quadratic and Lagrange cells are curved for ``k > 1`` and the colors of point-data
    are smoother. The options of the plots which are specific to PyVista, e.g.
    ``theme``, ``show_logo`` or other keyword-arguments of
    :meth:`pyvista.Plotter.add_mesh`, raise a warning once and they are ignored.

    See Also
    --------
    felupe.view.PlottingBackend : Base class of a plotting backend.
    felupe.set_plotting_backend : Set the current plotting backend by its name.
    """

    def is_plotter(self, plotter):
        """Return True if the given object is a plotter of matplotlib or
        two-dimensional axes of matplotlib. Matplotlib is not imported."""

        if isinstance(plotter, MatplotlibPlotter):
            return True

        axes = sys.modules.get("matplotlib.axes")

        return (
            axes is not None
            and isinstance(plotter, axes.Axes)
            and plotter.name == "rectilinear"
        )

    def prepare_plotter(
        self, plotter=None, theme="default", off_screen=False, notebook=False
    ):
        """Return the given plotter (or the plotter of given axes) or a new plotter.
        A new plotter of an off-screen plot or of a plot in a notebook creates its
        figure without pyplot. Themes are not supported."""

        if theme not in (None, "default"):
            _warn("theme", self.name)

        if plotter is None:
            return MatplotlibPlotter(off_screen=off_screen or notebook)

        return _as_plotter(plotter)

    def default_color(self):
        "Return the default color of the meshes."

        return DEFAULT_COLOR

    def screenshot(
        self,
        plot,
        *args,
        filename=None,
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the off-screen plotter, which is returned by
        ``plot(*args, off_screen=True, **kwargs)``. The format of the file is given by
        the suffix of the filename, e.g. a raster format (PNG) or a vector format
        (PDF, SVG or EPS).

        See Also
        --------
        felupe.view.MatplotlibPlotter.screenshot : Take a screenshot of the figure.
        """

        plotter = plot(*args, off_screen=True, **kwargs)

        return _as_plotter(plotter).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

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
        ``opacity``, ``style`` (``"surface"`` or ``"wireframe"``), ``lighting`` and
        ``show_scalar_bar``. Supported items of ``scalar_bar_args`` are ``"title"``,
        ``"vertical"``, ``"n_labels"`` and ``"fmt"``, e.g. ``"%.2e"``."""

        plotter = _as_plotter(plotter)

        required = ["points", "cells", "cell_type", "point_data", "cell_data"]
        if not all(hasattr(scene, attribute) for attribute in required):
            raise self._not_supported("scenes without points and cells, e.g. ViewXdmf")

        bar_args = dict(scalar_bar_args or {})
        title = bar_args.pop("title", None)
        vertical = bool(bar_args.pop("vertical", scalar_bar_vertical))
        n_labels = int(bar_args.pop("n_labels", 5))
        fmt = bar_args.pop("fmt", None)

        for option in bar_args:
            _warn(f"scalar_bar_args: {option}", self.name)

        if show_logo:
            _warn("show_logo", self.name)

        color = kwargs.pop("color", None)
        opacity = float(kwargs.pop("opacity", 1.0))
        style = kwargs.pop("style", "surface")
        show_scalar_bar = kwargs.pop("show_scalar_bar", True)
        lighting = bool(kwargs.pop("lighting", True))

        if style not in ("surface", "wireframe"):
            _warn("style", self.name)
            style = "surface"

        for option in kwargs:
            _warn(option, self.name)

        # the subdivided faces, i.e. curved faces of quadratic and Lagrange cells
        order = polynomial_order(scene)
        subdivisions = order * 2 ** (max(int(nonlinear_subdivision), 1) - 1)
        surface = triangulate_surface(scene, subdivisions=subdivisions)

        undeformed = _pad(surface.interpolate(scene.points))
        positions = undeformed

        if "Displacement" in scene.point_data:
            displacement = surface.interpolate(scene.point_data["Displacement"])
            positions = undeformed + factor * _pad(displacement)

        polygons = _polygons(surface.triangles, surface.faces, subdivisions)
        nfaces = int(surface.faces.max()) + 1 if len(surface.faces) > 0 else 0
        lines = surface.edges[surface.line_edges]
        wireframe = style == "wireframe"

        # the faces on the back of closed surfaces are not visible
        closed = CORNERS.get(str(scene.cell_type), (None, None))[0] in FACES

        # the camera and the axes are taken from the first view
        if plotter._views == 0:
            plotter.view = view
            plotter.axes = bool(add_axes)

        plotter._views += 1

        if show_undeformed and len(surface.triangles) > 0:
            ghost = _rgba(DEFAULT_COLOR)[:3]
            plotter.add(
                dict(
                    type="surface",
                    positions=undeformed,
                    polygons=[
                        (p, f, np.tile(ghost, (len(p), 1))) for p, f, t in polygons
                    ],
                    nfaces=nfaces,
                    outlines=None,
                    lines=None,
                    opacity=0.2,
                    fill=True,
                    stroke=False,
                    cull=False,
                    lighting=True,
                    edge_color=np.zeros(4),
                    line_width=0.0,
                    ghost=True,
                )
            )

        base = _rgba(DEFAULT_COLOR if color is None else color)[:3]
        face_colors = [np.tile(base, (len(p), 1)) for p, f, t in polygons]
        line_colors = np.tile(_rgba(edge_color)[:3], (len(lines), 1))
        scalar_bar = None
        values = None

        if name is not None:
            values, where, values_range, title_of_values = surface_scalars(
                scene, surface, name, component, label
            )
            colormap = _colormap(cmap)

            if where == "point":
                face_values = [values[p].mean(axis=1) for p, f, t in polygons]
                line_colors = _map_colors(
                    values[lines].mean(axis=1), colormap, values_range
                )
            else:
                face_values = [values[t] for p, f, t in polygons]

            face_colors = [_map_colors(v, colormap, values_range) for v in face_values]

            if show_scalar_bar:
                scalar_bar = dict(
                    label=title_of_values if title is None else title,
                    cmap=colormap,
                    range=values_range,
                    vertical=vertical,
                    n_labels=n_labels,
                    fmt=fmt,
                )

        # the edges of the faces are drawn by the strokes of the polygons, or by the
        # outlines of the faces if they are subdivided
        stroke = (show_edges or wireframe) and subdivisions == 1
        outlines = None

        if (show_edges or wireframe) and subdivisions > 1:
            outlines = _face_outlines(surface.triangles, surface.faces)

        plotter.add(
            dict(
                type="surface",
                positions=positions,
                polygons=[(p, f, c) for (p, f, t), c in zip(polygons, face_colors)],
                nfaces=nfaces,
                outlines=outlines,
                lines=(lines, line_colors),
                opacity=opacity,
                fill=not wireframe,
                stroke=stroke,
                cull=closed and opacity >= OPAQUE and not wireframe,
                lighting=lighting,
                edge_color=_rgba(edge_color, opacity),
                line_width=float(line_width),
                scalar_bar=scalar_bar,
                values=values,
            )
        )

        return plotter

    def add_points(self, plotter, points, color=None, point_size=None, **kwargs):
        """Add points to a plotter. Supported keyword-arguments are ``label`` (for the
        legend) and ``opacity``."""

        plotter = _as_plotter(plotter)

        label = kwargs.pop("label", None)
        opacity = float(kwargs.pop("opacity", 1.0))
        kwargs.pop("render_points_as_spheres", None)

        for option in kwargs:
            _warn(option, self.name)

        return plotter.add(
            dict(
                type="points",
                positions=_pad(points).reshape(-1, 3),
                color=_rgba(DEFAULT_COLOR if color is None else color, opacity),
                size=float(5.0 if point_size is None else point_size),
                label=label,
            )
        )

    def add_lines(self, plotter, lines, color=None, line_width=None, **kwargs):
        """Add line segments to a plotter. A supported keyword-argument is
        ``opacity``."""

        plotter = _as_plotter(plotter)

        opacity = float(kwargs.pop("opacity", 1.0))

        # the properties of the lines of PyVista have no effect here
        for option in ["lighting", "show_edges", "edge_color"]:
            kwargs.pop(option, None)

        for option in kwargs:
            _warn(option, self.name)

        return plotter.add(
            dict(
                type="lines",
                positions=_pad(lines).reshape(-1, 3),
                color=_rgba("black" if color is None else color, opacity),
                width=float(1.0 if line_width is None else line_width),
            )
        )

    def add_arrows(self, plotter, points, directions, scale=1.0, color=None, **kwargs):
        """Add arrows to a plotter, given by the triangles of a glyph (a cylinder with a
        cone at the tip). A supported keyword-argument is ``opacity``."""

        plotter = _as_plotter(plotter)

        opacity = float(kwargs.pop("opacity", 1.0))

        for option in kwargs:
            _warn(option, self.name)

        positions, triangles = arrows(_pad(points), _pad(directions), scale)

        return plotter.add(
            _shape_surface(positions, triangles, color, opacity, None, 1.0)
        )

    def add_shape(
        self, plotter, shape, color=None, opacity=None, show_edges=None, **kwargs
    ):
        """Add a shape to a plotter by the triangles of ``shape.triangulate()``. With
        ``show_edges=True``, the boundary edges of the triangles are shown. A supported
        keyword-argument is ``line_width`` (of the edges)."""

        plotter = _as_plotter(plotter)

        line_width = kwargs.pop("line_width", None)

        for option in kwargs:
            _warn(option, self.name)

        positions, triangles = shape.triangulate()

        if len(triangles) == 0:
            return plotter

        edges = outline(triangles) if show_edges else None

        return plotter.add(
            _shape_surface(
                positions,
                triangles,
                color,
                1.0 if opacity is None else float(opacity),
                edges,
                1.0 if line_width is None else float(line_width),
            )
        )

    def add_legend(
        self, plotter, entries=None, size=None, background_color=None, **kwargs
    ):
        """Add a legend to a plotter. Without entries, the entries are taken from the
        labels of the points."""

        plotter = _as_plotter(plotter)

        for option in kwargs:
            _warn(option, self.name)

        if entries is None:
            entries = [
                (obj["label"], obj["color"])
                for obj in plotter.objects
                if obj["type"] == "points" and obj.get("label") is not None
            ]

        plotter.legend = [
            (str(label).strip(), _rgba(color)) for label, color in entries
        ]
        plotter.legend_background = background_color
        plotter._stale = True

        if plotter._eager:
            plotter.render()

        return plotter


def _shape_surface(positions, triangles, color, opacity, edges, line_width):
    """Return a surface of triangles, e.g. of arrows or shapes, with optional edges.
    The edges are drawn after the nearest triangle (as the outline of one face)."""

    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    faces = np.zeros(len(triangles), dtype=np.int64)
    base = _rgba(DEFAULT_COLOR if color is None else color)[:3]
    edge_color = _rgba("black" if color is None else color, opacity)

    # the two triangles of a plane or a rectangle are merged into a quad
    polygons = _polygons(triangles, faces, subdivisions=1)

    return dict(
        type="surface",
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        polygons=[(p, f, np.tile(base, (len(p), 1))) for p, f, t in polygons],
        nfaces=1,
        outlines=None if edges is None else (edges, np.zeros(len(edges), np.int64)),
        lines=None,
        opacity=opacity,
        fill=True,
        stroke=False,
        cull=False,
        lighting=True,
        edge_color=edge_color,
        line_width=line_width,
    )
