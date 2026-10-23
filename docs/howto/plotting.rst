Choose a Plotting Backend
~~~~~~~~~~~~~~~~~~~~~~~~~

The plot-methods of FElupe, e.g. :meth:`Mesh.plot() <felupe.Mesh.plot>`,
:meth:`FieldContainer.plot() <felupe.FieldContainer.plot>` or
:meth:`SolidBody.plot() <felupe.SolidBody.plot>`, and the plots of the decorations of a
model, e.g. of boundary conditions, point loads or rigid contact surfaces, are drawn by
a plotting backend. The current plotting backend is selected by its name with
:func:`~felupe.set_plotting_backend` and returned by
:func:`~felupe.get_plotting_backend`.

..  code-block:: python

    import felupe as fem

    fem.set_plotting_backend("matplotlib")  # "pyvista" (default), "webgl"
    fem.get_plotting_backend()  # "matplotlib"

The arguments of the plot-methods are the same for all plotting backends, e.g. the name
and the component of the scalars, the scale ``factor`` of the deformation, the ``view``
or ``nonlinear_subdivision`` for curved faces of quadratic cells. A plot-method returns
the plotter of the plotting backend. Options which are specific to PyVista, e.g. a
``theme``, raise a warning once in the other plotting backends and they are ignored.

..  list-table::
    :header-rows: 1

    * - Plotting backend
      - Draws by
      - Interactive
      - Screenshots
      - Optional dependencies
    * - ``"pyvista"`` (default)
      - :class:`pyvista.Plotter`, in a window or in a notebook
      - yes
      - raster formats (PNG)
      - ``felupe[view]``
    * - ``"webgl"``
      - a WebGL viewer in the web browser (a self-contained HTML page)
      - yes
      - no
      - ``felupe[web]`` (only for the Jupyter widget)
    * - ``"matplotlib"``
      - the axes of a figure of matplotlib
      - no
      - raster and vector formats (PNG, PDF, SVG, EPS), off-screen
      - ``felupe[plot]``

Use the plotting backend ``"pyvista"`` for an interactive exploration of a model in a
window or in a notebook, the plotting backend ``"webgl"`` for a fast, lightweight view
in the web browser (also to share a model as a HTML file) and the plotting backend
``"matplotlib"`` for figures of publications, screenshots without a display (e.g. on a
server or in a CI-pipeline) or a model together with other plots in one figure.

The examples of this section use the following model.

..  pyvista-plot::
    :context:
    :nofigs:

    import felupe as fem

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=5), field=field)

    move = fem.math.linsteps([0, 0.5], num=5)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    curve = fem.CharacteristicCurvePlugin(boundary=boundaries["move"])
    job = fem.Job(steps=[step], plugins=[curve]).evaluate()

PyVista
*******

The plotting backend ``"pyvista"`` is the default plotting backend. A plot-method
returns a :class:`pyvista.Plotter`, i.e. the plot may be customized by the API of
PyVista before it is shown. Screenshots are taken by an off-screen plotter. The plots of
finite elements and regions, e.g. of a :class:`~felupe.Hexahedron` or of a
:class:`~felupe.RegionHexahedron`, are only supported by this plotting backend.

..  pyvista-plot::
    :context:
    :force_static:

    fem.set_plotting_backend("pyvista")

    solid.plot("Principal Values of Cauchy Stress").show()

..  code-block:: python

    img = solid.screenshot(
        "Principal Values of Cauchy Stress",
        filename="solid.png",
        transparent_background=True,
    )

WebGL
*****

The plotting backend ``"webgl"`` draws the plots by a dependency-free WebGL viewer in
the web browser, without PyVista. The plotter is shown by ``show()``: in a notebook,
the self-contained HTML page of the viewer is shown in the output of the cell,
otherwise it is opened in the web browser. The page works offline and it may be
written to a file by ``export_html()``, e.g. to share a model. Drag to rotate,
shift-drag to pan, scroll to zoom and double-click to reset the view. The live
dashboard of a job, :class:`~felupe.DashboardWidgetPlugin`, uses the same viewer.

..  code-block:: python

    fem.set_plotting_backend("webgl")

    plotter = solid.plot("Principal Values of Cauchy Stress")
    plotter.show()
    plotter.export_html("solid.html")

In a notebook, ``widget()`` returns a Jupyter widget of the viewer, e.g. for a layout
of widgets. This requires the optional dependency ``anywidget``
(``pip install felupe[web]``). Screenshots are not supported by this plotting
backend.

..  code-block:: python

    import ipywidgets as widgets

    left = solid.plot("Principal Values of Cauchy Stress").widget(height="320px")
    right = solid.plot("Displacement", component=None).widget(height="320px")
    widgets.HBox([left, right])

Matplotlib
**********

The plotting backend ``"matplotlib"`` draws the plots on the axes of a figure of
matplotlib, without PyVista. The plotter, a :class:`~felupe.view.MatplotlibPlotter`,
has the attributes ``figure`` and ``ax``, e.g. to add a title, and it is shown by
``show()``: by pyplot in a window or as image in the output of a cell of a notebook.

..  plot::
    :context:
    :nofigs:
    :include-source: false

    import felupe as fem

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=5), field=field)

    move = fem.math.linsteps([0, 0.5], num=5)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    curve = fem.CharacteristicCurvePlugin(boundary=boundaries["move"])
    job = fem.Job(steps=[step], plugins=[curve]).evaluate(verbose=False)

..  plot::
    :context: close-figs

    fem.set_plotting_backend("matplotlib")

    plotter = solid.plot("Principal Values of Cauchy Stress")
    plotter.ax.set_title("Uniaxial tension")
    plotter.show()

The screenshots are taken off-screen, i.e. without a display and without a figure of
pyplot. The format of the file is given by the suffix of the filename, i.e. a raster
format (PNG) or a vector format (PDF, SVG or EPS). A screenshot in a vector format
contains a polygon for each visible face of the model (with its edges), i.e. it stays
sharp at any zoom level, e.g. in a publication.

..  code-block:: python

    img = solid.screenshot("Principal Values of Cauchy Stress", filename="solid.pdf")
    img = solid.screenshot(filename="solid.png", scale=2, transparent_background=True)

Axes of matplotlib are accepted as plotter, i.e. a model may be drawn together with
other plots in one figure, e.g. with its force-displacement curve. The scene on given
axes is drawn whenever an object is added, i.e. the figure is saved as usual by
:meth:`fig.savefig() <matplotlib.figure.Figure.savefig>`, e.g. in a vector format.

..  plot::
    :context: close-figs

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    solid.plot("Principal Values of Cauchy Stress", plotter=ax[0])
    curve.plot(ax=ax[1], xlabel="Displacement $u$", ylabel="Force $F$", marker="o")

    fig.tight_layout()
    fig.savefig("figure.pdf")

..  plot::
    :context: close-figs
    :nofigs:
    :include-source: false

    fem.set_plotting_backend("pyvista")

The faces are drawn as polygons with a flat color each, in the order of their depths
(painter's algorithm), and the faces on the back of closed surfaces are not drawn.
Hence, faces which intersect each other, e.g. at a contact, may be drawn in a wrong
order. The edges of the faces are subdivided by ``nonlinear_subdivision=k`` into
``order * 2 ** (k - 1)`` segments, also of linear cells, i.e. the colors of point-data
are smoother for ``k > 1``. Planar models are drawn by a parallel projection, other
models by a perspective projection, unless it is changed by
``plotter.enable_parallel_projection()``. For very large models, the model may be
rasterized in vector formats by ``plotter.rasterized = True``, while the scalar bars,
the legend and the text stay vector graphics.

..  code-block:: python

    plotter = field.plot("Displacement", component=None, nonlinear_subdivision=3)
    plotter.enable_parallel_projection()
    plotter.savefig("displacement.svg")

Combine Plots
*************

The plotter of a plot-method is passed to other plot-methods by the argument
``plotter``, e.g. to draw the boundary conditions or a point load on top of the
deformed model. A given plotter is always drawn by its own plotting backend, i.e. a
:class:`pyvista.Plotter` is drawn by PyVista, independent of the current plotting
backend.

..  code-block:: python

    plotter = solid.plot("Displacement", component=None)
    plotter = boundaries.plot(plotter=plotter)
    plotter.show()

Custom Plotting Backends
************************

A custom plotting backend is a subclass of :class:`~felupe.view.PlottingBackend`. It
implements the methods ``is_plotter(plotter)``, ``prepare_plotter(plotter=None,
**kwargs)`` and ``add_view(plotter, scene, **kwargs)``. The primitives of the
decorations, e.g. ``add_points()``, ``add_lines()``, ``add_arrows()``, ``add_shape()``
and ``add_legend()``, and ``screenshot()`` are optional. The scenes provide their
points, cells, cell type, point- and cell-data as arrays, i.e. a plotting backend
doesn't depend on PyVista. A plotting backend is registered by a name with
:func:`~felupe.view.register_plotting_backend`.

..  code-block:: python

    fem.view.register_plotting_backend("custom", CustomBackend())
    fem.set_plotting_backend("custom")
