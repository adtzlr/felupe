# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod


class PlottingBackend(ABC):
    r"""Base class of a plotting backend, which draws the scenes of the plot-methods,
    e.g. of :meth:`Mesh.plot() <felupe.Mesh.plot>`,
    :meth:`FieldContainer.plot() <felupe.FieldContainer.plot>` or
    :meth:`SolidBody.plot() <felupe.SolidBody.plot>`.

    A plotting backend draws on its own type of plotter. A subclass implements
    :meth:`is_plotter`, :meth:`prepare_plotter` and :meth:`add_view`. The other methods
    are optional and raise a :class:`NotImplementedError` by default: the primitives
    :meth:`add_points`, :meth:`add_lines` and :meth:`add_arrows`, which draw the
    decorations of a model, e.g. of boundary conditions, point loads and constraints,
    and :meth:`screenshot`.

    Attributes
    ----------
    name : str or None
        The name of the plotting backend, which is set by
        :func:`~felupe.view.register_plotting_backend`.

    Notes
    -----
    A plot-method draws its scene on a given plotter or on a new plotter. The argument
    ``plotter`` combines several plots in one plotter, e.g.
    ``mesh.plot(plotter=other_mesh.plot())``. A given plotter is drawn by the plotting
    backend which recognizes it (by :meth:`is_plotter`), independent of the current
    plotting backend. Without a given plotter, the current plotting backend creates a
    new plotter, see :func:`~felupe.set_plotting_backend`.

    Examples
    --------
    A minimal plotting backend, which collects the scenes in a list.

    >>> import felupe as fem
    >>>
    >>> class ListPlotter(list):
    ...     def show(self):
    ...         print(f"Plotter with {len(self)} scene(s)")
    >>>
    >>> class ListBackend(fem.view.PlottingBackend):
    ...     def is_plotter(self, plotter):
    ...         return isinstance(plotter, ListPlotter)
    ...
    ...     def prepare_plotter(self, plotter=None, **kwargs):
    ...         if plotter is None:
    ...             plotter = ListPlotter()
    ...         return plotter
    ...
    ...     def add_view(self, plotter, scene, **kwargs):
    ...         plotter.append(scene)
    ...         return plotter
    >>>
    >>> fem.view.register_plotting_backend("list", ListBackend())
    >>> fem.set_plotting_backend("list")
    >>>
    >>> mesh = fem.Cube(n=3)
    >>> mesh.plot(plotter=mesh.plot()).show()
    Plotter with 2 scene(s)

    >>> fem.set_plotting_backend("pyvista")

    See Also
    --------
    felupe.set_plotting_backend : Set the current plotting backend by its name.
    felupe.get_plotting_backend : Return the name of the current plotting backend.
    felupe.view.register_plotting_backend : Register a plotting backend by a name.
    """

    name = None

    @abstractmethod
    def is_plotter(self, plotter):
        """Return True if the given object is a plotter of this plotting backend.

        Parameters
        ----------
        plotter : object
            The object to check.

        Returns
        -------
        bool
            True if the object is a plotter of this plotting backend.
        """

    @abstractmethod
    def prepare_plotter(self, plotter=None, **kwargs):
        """Return the given plotter or a new plotter.

        Parameters
        ----------
        plotter : object or None, optional
            A plotter of this plotting backend or None (default is None). If None, a
            new plotter is created.
        **kwargs : dict, optional
            The options of the plotter of :meth:`felupe.view.Scene.plot`, i.e.
            ``theme``, ``off_screen`` and ``notebook``.

        Returns
        -------
        object
            The given or a new plotter.
        """

    @abstractmethod
    def add_view(self, plotter, scene, **kwargs):
        """Add a scene to a plotter and return the plotter.

        Parameters
        ----------
        plotter : object
            A plotter of this plotting backend.
        scene : felupe.view.Scene
            The scene to add, e.g. a :class:`~felupe.ViewMesh` with the attributes
            ``points``, ``cells``, ``cell_type``, ``point_data`` and ``cell_data``.
        **kwargs : dict, optional
            The other arguments of :meth:`felupe.view.Scene.plot`, e.g. ``name``,
            ``component`` and ``factor``.

        Returns
        -------
        object
            The plotter.
        """

    def screenshot(
        self,
        plot,
        *args,
        filename=None,
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the plotter, which is returned by
        ``plot(*args, **kwargs)``. Not supported by default.

        Parameters
        ----------
        plot : callable
            A function which returns a plotter, e.g. the plot-method of a mesh.
        *args : tuple, optional
            Optional positional arguments of the plot-function.
        filename : str or None, optional
            The filename of the screenshot (default is None).
        transparent_background : bool or None, optional
            A flag for a transparent background (default is None).
        scale : int or None, optional
            A factor of the resolution of the screenshot (default is None).
        **kwargs : dict, optional
            Optional keyword-arguments of the plot-function.

        Returns
        -------
        ndarray
            The image data of the screenshot.

        Raises
        ------
        NotImplementedError
            If the plotting backend does not support screenshots.
        """

        raise self._not_supported("screenshots")

    def add_points(self, plotter, points, color=None, point_size=None, **kwargs):
        """Add points to a plotter and return the plotter. Not supported by default.

        Parameters
        ----------
        plotter : object
            A plotter of this plotting backend.
        points : ndarray
            The point coordinates of shape ``(npoints, dim)`` with ``dim <= 3``.
        color : str or None, optional
            The color of the points (default is None).
        point_size : float or None, optional
            The size of the points (default is None).
        **kwargs : dict, optional
            Optional keyword-arguments of the points, e.g. a ``label`` for the legend.

        Returns
        -------
        object
            The plotter.
        """

        raise self._not_supported("points")

    def add_lines(self, plotter, lines, color=None, line_width=None, **kwargs):
        """Add straight line segments to a plotter and return the plotter. Not
        supported by default.

        Parameters
        ----------
        plotter : object
            A plotter of this plotting backend.
        lines : ndarray
            The start and end points of the line segments of shape
            ``(nlines, 2, dim)`` with ``dim <= 3``.
        color : str or None, optional
            The color of the lines (default is None).
        line_width : float or None, optional
            The width of the lines (default is None).
        **kwargs : dict, optional
            Optional keyword-arguments of the lines.

        Returns
        -------
        object
            The plotter.
        """

        raise self._not_supported("lines")

    def add_arrows(self, plotter, points, directions, scale=1.0, color=None, **kwargs):
        """Add arrows to a plotter and return the plotter. Not supported by default.

        Parameters
        ----------
        plotter : object
            A plotter of this plotting backend.
        points : ndarray
            The start points of the arrows of shape ``(narrows, dim)`` with
            ``dim <= 3``.
        directions : ndarray
            The vectors of the arrows of shape ``(narrows, dim)``.
        scale : float, optional
            A factor of the vectors of the arrows (default is 1.0).
        color : str or None, optional
            The color of the arrows (default is None).
        **kwargs : dict, optional
            Optional keyword-arguments of the arrows.

        Returns
        -------
        object
            The plotter.
        """

        raise self._not_supported("arrows")

    def _not_supported(self, what):
        "Return the error of an unsupported feature of this plotting backend."

        return NotImplementedError(
            f'The plotting backend "{self.name}" does not support {what}. Use the '
            'plotting backend "pyvista", e.g. by felupe.set_plotting_backend("pyvista").'
        )
