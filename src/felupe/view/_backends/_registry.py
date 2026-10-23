# SPDX-License-Identifier: GPL-3.0-or-later

"""The registry of the plotting backends with the current plotting backend."""

from ._base import PlottingBackend
from ._pyvista import PyVistaBackend
from ._webgl import WebGLBackend

_BACKENDS = {}
_CURRENT = "pyvista"


def register_plotting_backend(name, backend):
    r"""Register a plotting backend by a name. An already registered plotting backend
    with the same name is replaced.

    Parameters
    ----------
    name : str
        The name of the plotting backend, e.g. ``"pyvista"``.
    backend : felupe.view.PlottingBackend
        An instance of a plotting backend. Its attribute ``name`` is set to the given
        name.

    Raises
    ------
    TypeError
        If the backend is not an instance of :class:`~felupe.view.PlottingBackend`.

    See Also
    --------
    felupe.view.PlottingBackend : Base class of a plotting backend.
    felupe.set_plotting_backend : Set the current plotting backend by its name.
    """

    if not isinstance(backend, PlottingBackend):
        raise TypeError(
            "The plotting backend must be an instance of "
            f"felupe.view.PlottingBackend, got {type(backend).__name__}."
        )

    backend.name = name
    _BACKENDS[name] = backend


def set_plotting_backend(name):
    r"""Set the current plotting backend by its name. It draws the plots without a
    given plotter, e.g. ``mesh.plot()``. A plot with a given plotter, e.g.
    ``mesh.plot(plotter=plotter)``, is drawn by the plotting backend of the given
    plotter.

    Parameters
    ----------
    name : str
        The name of a registered plotting backend. Available are ``"pyvista"``
        (default) and ``"webgl"``.

    Raises
    ------
    ValueError
        If no plotting backend is registered with the given name.

    Notes
    -----
    The plotting backend ``"pyvista"`` draws the plots by :class:`pyvista.Plotter`
    (with screenshots, e.g. for :meth:`Mesh.screenshot() <felupe.Mesh.screenshot>`).

    The plotting backend ``"webgl"`` draws the plots by a WebGL viewer in the web
    browser, without PyVista. Its plotter has the methods ``show(filename=None)``,
    ``export_html(filename)`` and ``widget(height="480px")``. In a notebook, the viewer
    is shown in the output of the cell, as a Jupyter widget with binary buffers (this
    requires anywidget, ``pip install felupe[web]``) or as an ``iframe``. Otherwise,
    the self-contained HTML page of the viewer is written to a file, which is opened in
    the web browser. The outer surface of the meshes is curved for
    quadratic and Lagrange cells with ``nonlinear_subdivision > 1``. Screenshots, the
    plots of finite elements and regions and the options which are specific to PyVista
    (which raise a warning once) are not supported. Drag to rotate, shift-drag to pan,
    scroll to zoom and double-click to reset the view.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> fem.set_plotting_backend("webgl")
    >>> mesh = fem.Cube(n=3)
    >>> plotter = mesh.plot()
    >>> path = plotter.export_html("cube.html")
    >>>
    >>> fem.set_plotting_backend("pyvista")
    >>> fem.get_plotting_backend()
    'pyvista'

    See Also
    --------
    felupe.get_plotting_backend : Return the name of the current plotting backend.
    felupe.view.register_plotting_backend : Register a plotting backend by a name.
    """

    global _CURRENT

    if name not in _BACKENDS:
        available = ", ".join(f'"{key}"' for key in _BACKENDS)
        raise ValueError(
            f'Unknown plotting backend "{name}". Available are {available}.'
        )

    _CURRENT = name


def get_plotting_backend():
    r"""Return the name of the current plotting backend.

    Returns
    -------
    str
        The name of the current plotting backend, e.g. ``"pyvista"``.

    See Also
    --------
    felupe.set_plotting_backend : Set the current plotting backend by its name.
    """

    return _CURRENT


def backend_of(plotter=None):
    """Return the plotting backend of a given plotter. If no plotter is given or if no
    registered plotting backend recognizes the plotter, the current plotting backend
    is returned. The current plotting backend is checked first."""

    current = _BACKENDS[_CURRENT]

    if plotter is not None and not current.is_plotter(plotter):
        for backend in _BACKENDS.values():
            if backend.is_plotter(plotter):
                return backend

    return current


def require_pyvista(plotter, what):
    """Return the plotting backend of a given plotter or the current plotting backend,
    if it is the plotting backend of PyVista. Raise a :class:`NotImplementedError`
    for another plotting backend, e.g. for plots which require PyVista."""

    backend = backend_of(plotter)

    if not isinstance(backend, PyVistaBackend):
        raise backend._not_supported(what)

    return backend


def take_screenshot(plot, *args, **kwargs):
    """Take a screenshot of the plotter which is returned by ``plot(*args, **kwargs)``
    by the plotting backend of a given plotter (keyword-argument ``plotter``) or by
    the current plotting backend."""

    backend = backend_of(kwargs.get("plotter"))

    return backend.screenshot(plot, *args, **kwargs)


register_plotting_backend("pyvista", PyVistaBackend())
register_plotting_backend("webgl", WebGLBackend())
