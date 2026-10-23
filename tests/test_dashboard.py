# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import json
import os
import re
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import felupe as fem
from felupe.plugins._dashboard import _plugin
from felupe.plugins._dashboard._file import HERE, DashboardState, module
from felupe.plugins._dashboard._snapshot import DashboardSnapshot
from felupe.view._surface import triangulate_surface


@pytest.fixture
def kernel(monkeypatch):
    "Simulate an IPython kernel (the widget is created but not displayed)."
    pytest.importorskip("anywidget")
    displayed = []
    monkeypatch.setattr(_plugin, "_kernel", lambda: True)
    monkeypatch.setattr(_plugin, "_display", displayed.append)
    return displayed


def block(n=3):
    mesh = fem.Cube(n=n)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
    return field, boundaries, solid


def values(frame):
    "Return the decoded field values of a frame."
    return np.frombuffer(frame["values"], dtype=np.float32)


def surface_values(dashboard, values):
    "Return the values at the points of the surface of the dashboard, e.g. of a field."
    return dashboard._surface.interpolate(values).ravel()


def buffer(widget, name, shape, dtype=np.uint32):
    "Return a decoded binary buffer of the geometry of a widget."
    return np.frombuffer(getattr(widget, name), dtype=dtype).reshape(-1, *shape)


def test_dashboard_without_kernel(monkeypatch):
    """Outside of a kernel, a warning is raised once and the dashboard is not
    displayed, but the state of the job is collected (e.g. for a snapshot)."""

    def fail(uri):
        raise AssertionError("A page is only opened for a live HTML file.")

    monkeypatch.setattr(_plugin.webbrowser, "open", fail)

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.2], num=2)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardWidgetPlugin()
    assert not callable(dashboard)

    with pytest.warns(
        UserWarning, match="not displayed in a Python process without a Jupyter kernel"
    ) as w:
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    # the warning points to the evaluation of the job (outside of FElupe)
    assert w[0].filename == __file__
    assert "snapshot().save" in str(w[0].message)

    assert dashboard.widget is None
    assert not dashboard._active
    assert isinstance(dashboard._state, DashboardState)
    assert dashboard._state.progress["status"] == "finished"
    assert dashboard._state.path["status"] == "ccc"

    # the warning is raised only once
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    # a failed substep is shown, the error is raised
    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.8], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    with pytest.raises(fem.NewtonConvergenceError):
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    assert dashboard.widget is None
    assert dashboard._state.progress["status"] == "failed"
    assert dashboard._state.path["status"] == "cf"


def test_dashboard_ipython_terminal(monkeypatch):
    "The IPython terminal has a shell without a kernel, the widget is not displayed."

    monkeypatch.setattr(_plugin, "_shell", lambda: SimpleNamespace())
    assert not _plugin._kernel()

    field, boundaries, solid = block()
    step = fem.Step(items=[solid], boundaries=boundaries)
    dashboard = fem.DashboardWidgetPlugin()

    with pytest.warns(UserWarning, match="not displayed in the IPython terminal"):
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    assert dashboard.widget is None


def test_dashboard_without_anywidget(monkeypatch):
    "In a kernel without anywidget, an ImportError with a hint is raised."

    monkeypatch.setattr(_plugin, "_kernel", lambda: True)
    monkeypatch.setitem(sys.modules, "felupe.plugins._dashboard._widget", None)

    field, boundaries, solid = block()
    step = fem.Step(items=[solid], boundaries=boundaries)

    with pytest.raises(ImportError, match=r"felupe\[dashboard\]"):
        fem.Job(steps=[step], plugins=[fem.DashboardWidgetPlugin()]).evaluate(verbose=0)


def test_dashboard_job(kernel):
    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.2], num=2)
    steps = [
        fem.Step(items=[solid], ramp={boundaries["move"]: m}, boundaries=boundaries)
        for m in [move, move[::-1]]
    ]

    dashboard = fem.DashboardWidgetPlugin(title="T", interval=0)
    fem.Job(steps=steps, plugins=[dashboard]).evaluate(verbose=0)

    widget = dashboard.widget
    assert kernel == [widget]
    assert "widget support" in repr(widget)

    info = widget.info
    mesh = field.region.mesh
    assert info["title"] == "T"
    assert info["version"] == fem.__version__
    assert info["npoints"] == mesh.npoints
    assert info["ncells"] == mesh.ncells
    assert info["cell_type"] == "hexahedron"
    assert info["ndof"] == field[0].values.size
    assert info["element"] == "Hexahedron"
    assert info["quadrature"] == "GaussLegendre"
    assert info["quadrature_points_per_cell"] == 8
    assert info["deform"]
    assert info["field_dim"] == 3
    assert info["nsubsteps"] == [3, 3]
    assert info["items"] == ["SolidBody (NeoHooke)"]

    # the triangulated outer surface of the mesh, only the points of the surface are
    # transferred (the cube has one inner point)
    surface = triangulate_surface(mesh)
    points = buffer(widget, "points", (3,), np.float32)
    assert len(points) == 26
    assert np.allclose(points, surface.interpolate(mesh.points))
    triangles = buffer(widget, "triangles", (3,))
    assert len(triangles) == 2 * 24
    assert np.array_equal(triangles, surface.triangles)

    # the faces of the triangles (of linear cells), the edges and no line cells
    assert np.array_equal(buffer(widget, "faces", ()), surface.faces)
    assert np.array_equal(buffer(widget, "edges", (2,)), surface.edges)
    assert len(surface.edges) == 48
    assert widget.lines == b""
    assert info["surface"] == "hexahedron"

    # only the values of the latest converged substep are kept
    frame = widget.frame
    assert (frame["kind"], frame["step"], frame["substep"]) == ("substep", 1, 2)
    assert len(values(frame)) == 3 * 26
    assert np.allclose(
        values(frame), surface.interpolate(field[0].values).ravel(), atol=1e-6
    )

    assert widget.path == dict(status="cccccc", load_factors={})

    progress = widget.progress
    assert progress["status"] == "finished"
    assert progress["converged"] == 6
    assert progress["cutbacks"] == 0
    assert progress["failed_attempts"] == 0
    assert progress["iterations"] > 0
    assert progress["active_dofs"] < info["ndof"]
    assert progress["tol"] > 0
    assert progress["event"]["kind"] == "done"
    assert set(progress["timing"]) == {"assembly", "solve", "residual"}
    assert dashboard.overhead > 0

    newton = widget.newton
    assert len(newton["attempts"]) == 1
    assert newton["attempts"][0]["failed"] is False
    assert newton["attempts"][0]["fnorms"][-1] < newton["tol"]

    # a repeated evaluation creates a new widget, the previous one is kept
    fem.Job(steps=steps[:1], plugins=[dashboard]).evaluate(verbose=0)
    assert dashboard.widget is not widget
    assert len(kernel) == 2
    assert widget.progress["converged"] == 6
    assert dashboard.widget.progress["converged"] == 3
    assert dashboard.widget.path["status"] == "ccc"


@pytest.mark.parametrize("first", [True, False])
def test_dashboard_cutback(kernel, first):
    "The attempts of a cutback are observed in any order of the plugins."

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.7], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    dashboard = fem.DashboardWidgetPlugin()
    cutback = fem.CutbackPlugin()
    plugins = [dashboard, cutback] if first else [cutback, dashboard]
    fem.Job(steps=[step], plugins=plugins).evaluate(verbose=0)

    widget = dashboard.widget
    assert widget.path["status"] == "cb"
    assert widget.path["load_factors"] == {"1": cutback.load_factors[-1]}

    progress = widget.progress
    assert progress["cutbacks"] == 1
    assert progress["failed_attempts"] == cutback.cutbacks[-1]

    attempts = widget.newton["attempts"]
    assert [a["failed"] for a in attempts] == [True] + len(cutback.load_factors[-1]) * [
        False
    ]

    # the last event (the end of the job) and the values of the last substep
    assert progress["event"]["kind"] == "done"
    assert np.allclose(
        values(widget.frame), surface_values(dashboard, field[0].values), atol=1e-6
    )


def test_dashboard_cutback_event(kernel):
    "The event of a substep which converged with cutbacks."

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.7], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardWidgetPlugin()
    events = []

    class Recorder(fem.Plugin):
        def after_substep(self, context, state):
            events.append(dashboard.widget.progress["event"])

    plugins = [dashboard, fem.CutbackPlugin(), Recorder()]
    fem.Job(steps=[step], plugins=plugins).evaluate(verbose=0)

    assert events[0] is None
    assert events[1]["kind"] == "cut"
    assert "converged with 1 cutback in 2 increments" in events[1]["text"]


def test_dashboard_failed_job(kernel):
    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.8], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardWidgetPlugin()

    with pytest.raises(fem.NewtonConvergenceError):
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    widget = dashboard.widget
    assert widget.path["status"] == "cf"
    assert widget.progress["status"] == "failed"
    assert widget.progress["failed_attempts"] == 1
    assert widget.progress["event"]["kind"] == "fail"
    assert widget.progress["event"]["text"].startswith("NewtonConvergenceError")
    assert widget.newton["attempts"][-1]["failed"] is True

    # the values of the last converged substep are kept
    assert widget.frame["substep"] == 0


def test_dashboard_scalar_field(kernel):
    mesh = fem.Rectangle(n=5)
    region = fem.RegionQuad(mesh)
    temperature = fem.Field(region, dim=1)
    field = fem.FieldContainer([temperature])
    boundaries = fem.BoundaryDict(
        left=fem.Boundary(temperature, fx=0),
        right=fem.Boundary(temperature, fx=1),
    )
    solid = fem.thermal.SolidBodyThermal(
        field=field,
        mass_density=1.0,
        specific_heat_capacity=1.0,
        thermal_conductivity=1.0,
    )
    step = fem.Step(
        items=[solid],
        ramp={boundaries["right"]: fem.math.linsteps([0, 1], num=2)},
        boundaries=boundaries,
    )
    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    info = dashboard.widget.info
    assert not info["deform"]
    assert info["field_dim"] == 1
    assert info["surface"] == "quad"
    # all points of a two-dimensional mesh are points of the surface
    assert np.allclose(values(dashboard.widget.frame), temperature.values.ravel())


def test_dashboard_newton_without_job(kernel):
    "The hooks of the Newton-Raphson method outside of a job are ignored."

    field, boundaries, solid = block()
    boundaries, loadcase = fem.dof.uniaxial(field, move=-0.1, return_loadcase=True)
    dashboard = fem.DashboardWidgetPlugin()
    fem.newtonraphson(items=[solid], plugins=[dashboard], verbose=0, **loadcase)

    assert dashboard.widget is None
    assert kernel == []

    # the hooks of a job are ignored before the job (and after the job)
    context = SimpleNamespace(job=None, x0=field)
    state = SimpleNamespace(stepnumber=0, substepnumber=0, error=None)
    for hook in [
        "before_step",
        "before_substep",
        "after_failed_substep",
        "after_substep",
        "after_job",
    ]:
        getattr(dashboard, hook)(context, state)

    assert dashboard._state is None


def test_dashboard_interval(kernel):
    "The solver figures are updated at most every interval during a substep."

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.2], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    counts = {}
    for interval in [0, 1e6]:
        dashboard = fem.DashboardWidgetPlugin(interval=interval)
        changes = []

        class Observer(fem.Plugin):
            def before_job(self, context, state):
                dashboard.widget.observe(changes.append, names="progress")

        fem.Job(steps=[step], plugins=[dashboard, Observer()]).evaluate(verbose=0)
        counts[interval] = len(changes)

    # the forced updates at the begin and the end of the substeps remain
    assert counts[1e6] < counts[0]
    assert counts[1e6] > 0


def test_dashboard_multiple_bodies(kernel):
    """Two solid bodies on the meshes of a container with merged points. The model is
    drawn from the cells of both bodies and the field values are taken from the
    unknowns of the job."""

    meshes = [fem.Cube(n=3), fem.Cube(n=3).translate(1, axis=0)]
    container = fem.MeshContainer(meshes, merge=True)
    field = fem.Field.from_mesh_container(container).as_container()
    regions = [fem.RegionHexahedron(mesh) for mesh in container.meshes]
    fields = [fem.FieldContainer([fem.Field(r, dim=3)]) for r in regions]
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solids = [
        fem.SolidBody(umat=fem.LinearElasticLargeStrain(E=10, nu=0.3), field=fields[0]),
        fem.SolidBodyNearlyIncompressible(
            umat=fem.NeoHooke(mu=1), field=fields[1], bulk=5000
        ),
    ]
    move = fem.math.linsteps([0, 0.5], num=2)
    step = fem.Step(
        items=solids, ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(x0=field, verbose=0)

    widget = dashboard.widget
    info = widget.info
    assert info["npoints"] == len(container.points) == 45
    assert info["ncells"] == 16
    assert info["cell_type"] == "hexahedron"
    assert info["ndof"] == field[0].values.size
    assert info["deform"]
    assert info["items"] == [
        "SolidBody (LinearElasticLargeStrain)",
        "SolidBodyNearlyIncompressible (NeoHooke)",
    ]

    # the outer surface of both cubes, without the faces at the interface (and without
    # the three points inside)
    triangles = buffer(widget, "triangles", (3,))
    assert len(triangles) == 2 * 40
    assert len(buffer(widget, "points", (3,), np.float32)) == 42
    assert triangles.max() == 41

    assert np.allclose(
        values(widget.frame), surface_values(dashboard, field[0].values), atol=1e-6
    )
    assert np.abs(values(widget.frame)).max() > 0


def test_dashboard_boundary_items(kernel):
    "The cells of the boundary regions of items (e.g. of a pressure) are not drawn."

    field, boundaries, solid = block()
    region = fem.RegionHexahedronBoundary(
        field.region.mesh, mask=field.region.mesh.x == 1
    )
    pressure = fem.SolidBodyPressure(
        fem.FieldContainer([fem.Field(region, dim=3)]), pressure=0.1
    )
    step = fem.Step(
        items=[solid, pressure],
        ramp={pressure: fem.math.linsteps([0, 1], num=2)},
        boundaries=boundaries,
    )
    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(x0=field, verbose=0)

    info = dashboard.widget.info
    assert info["ncells"] == 8
    assert info["items"] == ["SolidBody (NeoHooke)", "SolidBodyPressure"]
    assert len(buffer(dashboard.widget, "triangles", (3,))) == 2 * 24


def test_dashboard_point_load_on_container_field(kernel):
    """A point load on the vertex field of a mesh container (the unknowns of the job)
    does not add cells to the model."""

    meshes = [fem.Cube(n=3), fem.Cube(n=3).translate(1, axis=0)]
    container = fem.MeshContainer(meshes, merge=True)
    field = fem.Field.from_mesh_container(container).as_container()
    regions = [fem.RegionHexahedron(mesh) for mesh in container.meshes]
    umat = fem.NeoHooke(mu=1, bulk=5)
    solids = [
        fem.SolidBody(umat, fem.FieldContainer([fem.Field(r, dim=3)])) for r in regions
    ]
    boundaries = fem.dof.uniaxial(field, clamped=True)
    load = fem.PointLoad(field, points=[44], values=[0, 0, -0.01])
    step = fem.Step(items=[*solids, load], boundaries=boundaries)

    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(x0=field, verbose=0)

    info = dashboard.widget.info
    assert info["ncells"] == 16
    assert info["cell_type"] == "hexahedron"
    assert info["items"] == ["SolidBody (NeoHooke)", "PointLoad"]


def test_dashboard_thermal(kernel):
    """A transient thermal job with a heat transfer on a boundary region and a time
    step (an item without a field)."""

    mesh = fem.Cube(n=4)
    region = fem.RegionHexahedron(mesh)
    temperature = fem.Field(region, dim=1)
    field = fem.FieldContainer([temperature])
    boundaries = fem.BoundaryDict(left=fem.Boundary(temperature, fx=0))

    solid = fem.thermal.SolidBodyThermal(
        field,
        mass_density=1.0,
        specific_heat_capacity=1.0,
        thermal_conductivity=1.0,
        time_step=0.1,
    )
    right = fem.RegionHexahedronBoundary(mesh, mask=mesh.x == 1)
    transfer = fem.thermal.SolidBodySurfaceHeatTransfer(
        fem.FieldContainer([fem.Field(right, dim=1)]), coefficient=1.0, temperature=0.0
    )
    time = fem.thermal.TimeStep([solid])
    table = fem.math.linsteps([0, 1], num=4)
    step = fem.Step(
        items=[solid, transfer, time],
        ramp={boundaries["left"]: 10 * table, time: 0.1 * table},
        boundaries=boundaries,
    )
    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    widget = dashboard.widget
    info = widget.info
    assert info["ncells"] == mesh.ncells
    assert not info["deform"]
    assert info["field_dim"] == 1
    assert info["items"] == [
        "SolidBodyThermal (Laplace)",
        "SolidBodySurfaceHeatTransfer",
        "TimeStep",
    ]
    assert len(buffer(widget, "triangles", (3,))) == 2 * 54
    assert widget.path["status"] == "ccccc"
    assert len(buffer(widget, "points", (3,), np.float32)) < mesh.npoints
    assert np.allclose(
        values(widget.frame), surface_values(dashboard, temperature.values)
    )
    assert values(widget.frame).max() > 0


def test_dashboard_mixed_fields(kernel):
    "Only the first field of mixed fields is drawn."

    region = fem.RegionHexahedron(fem.Cube(n=3))
    field = fem.FieldsMixed(region, n=3)
    boundaries = fem.dof.uniaxial(field, clamped=True)
    umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=50))
    solid = fem.SolidBody(umat, field)
    move = fem.math.linsteps([0, -0.1], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    info = dashboard.widget.info
    assert info["field_dim"] == 3
    assert info["deform"]
    assert info["ndof"] == sum(f.values.size for f in field.fields)
    assert np.allclose(
        values(dashboard.widget.frame), surface_values(dashboard, field[0].values)
    )


def test_dashboard_recovered_substep(kernel):
    """A substep which is recovered by another plugin than a cutback (without load
    factors) is shown as converged after failed attempts."""

    class Failing(fem.Plugin):
        "Fail the first attempt of the second substep."

        def __init__(self):
            self.count = 0

        def before_newton(self, context, state):
            self.count += 1

        def after_linear_solve(self, context, state):
            if self.count == 2:
                raise fem.NewtonConvergenceError("Injected failure.")

    class Recovery(fem.Plugin):
        "Solve the failed substep again with the same values."

        def after_failed_substep(self, context, state):
            state.result = context.solve(state.values)

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.1], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardWidgetPlugin()
    events = []

    class Recorder(fem.Plugin):
        def after_substep(self, context, state):
            events.append(dashboard.widget.progress["event"])

    plugins = [dashboard, Failing(), Recovery(), Recorder()]
    fem.Job(steps=[step], plugins=plugins).evaluate(verbose=0)

    widget = dashboard.widget
    assert widget.path == dict(status="cb", load_factors={})
    assert widget.progress["failed_attempts"] == 1
    assert events[1]["text"] == ("Step 1, substep 2: converged after 1 failed attempt.")
    assert [a["failed"] for a in widget.newton["attempts"]] == [True, False]


def test_dashboard_error_before_substep(kernel):
    "An error of another plugin before the first substep."

    class Error(fem.Plugin):
        def before_step(self, context, state):
            raise RuntimeError("Injected error.")

    field, boundaries, solid = block()
    step = fem.Step(items=[solid], boundaries=boundaries)
    dashboard = fem.DashboardWidgetPlugin()

    with pytest.raises(RuntimeError, match="Injected error"):
        fem.Job(steps=[step], plugins=[dashboard, Error()]).evaluate(verbose=0)

    progress = dashboard.widget.progress
    assert progress["status"] == "failed"
    assert progress["event"]["text"] == "RuntimeError: Injected error."
    assert dashboard.widget.path["status"] == "."


def test_dashboard_helpers(monkeypatch):
    "The helpers of the plugin, also without IPython and without the module resource."

    assert _plugin._num(None) is None
    assert _plugin._num("text") is None
    assert _plugin._num(np.nan) is None
    assert _plugin._num(np.float32(2.5)) == 2.5

    assert _plugin._first_line(ValueError("first\nsecond")) == "first"
    assert _plugin._first_line(ValueError()) == "ValueError"

    assert _plugin._peak_memory() > 0
    assert not _plugin._kernel()

    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setitem(sys.modules, "IPython", None)
    assert _plugin._peak_memory() is None
    assert not _plugin._kernel()


def test_dashboard_peak_memory_windows(monkeypatch):
    """The peak memory on Windows by the function GetProcessMemoryInfo of the Windows
    API (by a fake of ctypes.windll on other platforms)."""

    import ctypes

    calls = []

    def GetCurrentProcess():
        return -1

    def GetProcessMemoryInfo(process, counters, size):
        calls.append((process, size))
        counters._obj.PeakWorkingSetSize = 123456789
        return success

    windll = SimpleNamespace(
        kernel32=SimpleNamespace(GetCurrentProcess=GetCurrentProcess),
        psapi=SimpleNamespace(GetProcessMemoryInfo=GetProcessMemoryInfo),
    )
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    success = True
    assert _plugin._peak_memory() == 123456789
    assert calls[-1] == (-1, 80 if ctypes.sizeof(ctypes.c_size_t) == 8 else 40)

    # a failed call of the Windows API
    success = False
    assert _plugin._peak_memory() is None


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows")
def test_dashboard_peak_memory_windows_api():
    "The peak memory by the Windows API (only on Windows)."

    peak = _plugin._peak_memory_windows()
    assert peak > 10 * 1024**2
    assert _plugin._peak_memory() >= peak


def test_dashboard_display(monkeypatch):
    "The widget is displayed by IPython.display.display."

    display = pytest.importorskip("IPython.display")
    displayed = []
    monkeypatch.setattr(display, "display", displayed.append)

    widget = object()
    _plugin._display(widget)
    assert displayed == [widget]


def test_dashboard_regions():
    """The regions of the items of a job: items without a field or without a region
    are skipped, a boundary region is only used if there is no other region and the
    meshes with other points than the first mesh are not drawn."""

    def job(*items):
        return SimpleNamespace(steps=[SimpleNamespace(items=list(items))])

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    boundary = fem.RegionHexahedronBoundary(mesh)
    solid = SimpleNamespace(field=fem.FieldContainer([fem.Field(region, dim=3)]))
    pressure = SimpleNamespace(field=fem.FieldContainer([fem.Field(boundary, dim=3)]))
    without_field = SimpleNamespace()
    without_region = SimpleNamespace(field=SimpleNamespace(region=None))

    regions = _plugin.DashboardWidgetPlugin._regions
    assert regions(job()) == []
    assert regions(job(without_field, without_region)) == []
    assert regions(job(without_field, without_region, solid)) == [region]
    assert regions(job(pressure)) == [boundary]
    assert regions(job(pressure, solid)) == [region]

    # a body with other points is not drawn
    other = fem.RegionHexahedron(fem.Cube(n=3).translate(5, axis=0))
    body = SimpleNamespace(field=fem.FieldContainer([fem.Field(other, dim=3)]))
    assert regions(job(solid, body)) == [region]


def test_dashboard_without_mesh(kernel):
    "A job without a mesh has no geometry, the solver figures are still shown."

    dashboard = fem.DashboardWidgetPlugin()
    dashboard.widget = dashboard._state = _plugin._widget_class()()
    dashboard._model(SimpleNamespace(steps=[SimpleNamespace(items=[], nsubsteps=1)]))

    info = dashboard.widget.info
    assert info["npoints"] == 0
    assert info["ncells"] == 0
    assert info["surface"] == "points"
    assert dashboard.widget.points == b""

    # a first field without values at the points of the mesh is not drawn
    field, boundaries, solid = block()
    dashboard._unknowns(field)
    assert dashboard.widget.info["field_dim"] == 0
    assert not dashboard.widget.info["deform"]
    assert dashboard.widget.info["ndof"] == field[0].values.size
    assert dashboard.widget.frame["values"] == b""


def test_dashboard_limits(kernel):
    """The limits of the legend are evaluated by the values of the first field at all
    points, also at the points inside (which are not transferred)."""

    field, boundaries, solid = block()
    step = fem.Step(items=[solid], boundaries=boundaries)
    dashboard = fem.DashboardWidgetPlugin()
    dashboard.widget = dashboard._state = _plugin._widget_class()()
    dashboard._model(fem.Job(steps=[step]))

    # the cube has one inner point (13) with the largest value
    field[0].values[:] = 0.0
    field[0].values[13] = [0.0, 3.0, 4.0]
    field[0].values[0] = [1.0, 0.0, 0.0]
    dashboard._unknowns(field)

    frame = dashboard.widget.frame
    assert 13 not in dashboard._surface.interpolation.indices
    assert np.abs(values(frame)).max() == 1.0
    assert frame["limits"] == [0.0, 5.0]

    # no limits for non-finite values (the frontend uses the values of the surface)
    field[0].values[13] = np.nan
    dashboard._unknowns(field)
    assert dashboard.widget.frame["limits"] is None


def read_script(path):
    "Return the data of a script of the live HTML file."
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("=") + 1 :].strip().rstrip(";"))


def read_config(page):
    "Return the inline config of a HTML page of the dashboard."
    start = page.index("const CONFIG = ") + len("const CONFIG = ")
    return json.loads(page[start : page.index(";\n", start)])


def test_dashboard_snapshot(tmp_path):
    "A static snapshot of the dashboard without a Jupyter kernel."

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.2], num=2)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardWidgetPlugin(title="A </script> title")

    with pytest.raises(ValueError, match="requires an evaluated job"):
        dashboard.snapshot()

    with pytest.warns(UserWarning, match="DashboardWidgetPlugin"):
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    snapshot = dashboard.snapshot()
    assert isinstance(snapshot, DashboardSnapshot)

    # the inline data, the title can not close the script of the page
    page = snapshot.html
    assert page.count("</script>") == 1
    config = read_config(page)
    assert set(config) == {"snapshot"}
    data = config["snapshot"]
    assert data["state"]["info"]["title"] == "A </script> title"
    assert data["state"]["progress"]["status"] == "finished"
    assert data["state"]["path"]["status"] == "ccc"

    points = np.frombuffer(base64.b64decode(data["geometry"]["points"]), np.float32)
    mesh_points = field.region.mesh.points
    assert np.allclose(points, surface_values(dashboard, mesh_points))
    frame = np.frombuffer(
        base64.b64decode(data["state"]["frame"]["values"]), np.float32
    )
    assert np.allclose(frame, surface_values(dashboard, field[0].values), atol=1e-6)
    assert set(data["geometry"]) == {"points", "triangles", "faces", "edges", "lines"}

    # the rich representation: the page in an iframe, in a single line
    html = snapshot._repr_html_()
    assert html.startswith("<iframe ") and html.endswith("</iframe>")
    assert "\n" not in html and "&#10;" in html
    assert 'srcdoc="&lt;!doctype html&gt;' in html

    # the snapshot is saved as a self-contained HTML file
    path = snapshot.save(tmp_path / "results" / "snapshot.html")
    assert path.read_text(encoding="utf-8") == page
    assert not list((tmp_path / "results").glob("*.js"))

    # the snapshot is not changed by the next job
    field, boundaries, solid = block(n=4)
    step = fem.Step(items=[solid], boundaries=boundaries)
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)
    assert snapshot.html == page
    assert dashboard.snapshot().html != page


def test_dashboard_snapshot_in_kernel(kernel):
    "A snapshot of the widget in a Jupyter kernel."

    field, boundaries, solid = block()
    step = fem.Step(items=[solid], boundaries=boundaries)
    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    data = read_config(dashboard.snapshot().html)["snapshot"]
    widget = dashboard.widget
    assert data["state"]["progress"] == json.loads(json.dumps(widget.progress))
    assert data["state"]["info"] == widget.info
    points = base64.b64decode(data["geometry"]["points"])
    assert points == widget.points


def test_dashboard_file(tmp_path, monkeypatch):
    """A live HTML file without a Jupyter kernel: no warning, the files are written and
    the page is opened once in the web browser."""

    opened = []
    monkeypatch.setattr(_plugin.webbrowser, "open", opened.append)

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.7], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    filename = tmp_path / "results" / "job.html"
    dashboard = fem.DashboardWidgetPlugin(title="T", filename=filename)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*DashboardWidgetPlugin")
        plugins = [dashboard, fem.CutbackPlugin()]
        fem.Job(steps=[step], plugins=plugins).evaluate(verbose=0)

    assert dashboard.widget is None
    assert opened == [filename.resolve().as_uri()]

    # the page with the inline module, the styles and the names of the scripts
    page = filename.read_text(encoding="utf-8")
    assert "/*" + "MODULE" + "*/" not in page and "function render(" in page
    assert "export function createViewer(" in page
    assert read_config(page) == {
        "files": {
            "geometry": "job.geometry.js",
            "state": "job.state.js",
            "version": "job.version.js",
        }
    }

    version = read_script(tmp_path / "results" / "job.version.js")
    geometry = read_script(tmp_path / "results" / "job.geometry.js")
    state = read_script(tmp_path / "results" / "job.state.js")

    assert version == {"geometry": dashboard._file.geometry, "state": state["version"]}
    assert geometry["id"] == state["geometry"] == dashboard._file.geometry
    assert geometry["id"].endswith("-1")
    assert state["info"]["title"] == "T"
    assert state["progress"]["status"] == "finished"
    assert state["path"]["status"] == "cb"
    assert state["written"] > 0

    # the binary buffers are encoded by base64
    points = np.frombuffer(base64.b64decode(geometry["points"]), dtype=np.float32)
    mesh_points = field.region.mesh.points
    assert np.allclose(points, surface_values(dashboard, mesh_points))
    frame = np.frombuffer(base64.b64decode(state["frame"]["values"]), dtype=np.float32)
    assert np.allclose(frame, surface_values(dashboard, field[0].values), atol=1e-6)

    # a second job: a new geometry, the page is not opened again
    field, boundaries, solid = block()
    step = fem.Step(items=[solid], boundaries=boundaries)
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)
    assert len(opened) == 1
    second = read_script(tmp_path / "results" / "job.version.js")["geometry"]
    assert second.endswith("-2") and second != geometry["id"]

    # a job of another process: another geometry (the counter starts again)
    other = fem.DashboardWidgetPlugin(filename=filename)
    fem.Job(steps=[step], plugins=[other]).evaluate(verbose=0)
    third = read_script(tmp_path / "results" / "job.version.js")["geometry"]
    assert third.endswith("-1") and third != geometry["id"]


@pytest.mark.parametrize("file_interval", [0, 1e6])
def test_dashboard_file_interval(tmp_path, monkeypatch, file_interval):
    "The state is written at most every file interval (and at the end of the job)."

    monkeypatch.setattr(_plugin.webbrowser, "open", lambda uri: None)

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.2], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    filename = tmp_path / "job.html"
    dashboard = fem.DashboardWidgetPlugin(
        filename=filename, file_interval=file_interval, interval=0
    )
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    written = read_script(tmp_path / "job.version.js")["state"]
    if file_interval:
        # the initial state and the final state
        assert written == 2
    else:
        assert written > 10
    assert read_script(tmp_path / "job.state.js")["progress"]["status"] == "finished"


def test_dashboard_file_in_kernel(kernel, tmp_path, monkeypatch):
    "In a Jupyter kernel, the live HTML file is written next to the widget."

    def fail(uri):
        raise AssertionError("The page must not be opened in a kernel.")

    monkeypatch.setattr(_plugin.webbrowser, "open", fail)

    field, boundaries, solid = block()
    step = fem.Step(items=[solid], boundaries=boundaries)
    dashboard = fem.DashboardWidgetPlugin(filename=tmp_path / "job.html")
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    assert kernel == [dashboard.widget]
    state = read_script(tmp_path / "job.state.js")
    assert state["progress"] == dashboard.widget.progress


def test_dashboard_file_writer(tmp_path, monkeypatch):
    """The atomic writes of the live HTML file, e.g. with a file which is locked by
    the web browser on Windows."""

    from felupe.plugins._dashboard import _file

    replace = os.replace
    failures = []

    def locked(source, destination):
        if failures and failures.pop():
            raise PermissionError("The file is locked.")
        replace(source, destination)

    monkeypatch.setattr(_file.os, "replace", locked)
    path = tmp_path / "data.js"

    # a locked file is written by a repeated replacement
    failures[:] = [True, True]
    assert _file._write(path, "a", delay=0)
    assert path.read_text() == "a"

    # a file which stays locked is not written
    failures[:] = [True] * 5
    assert not _file._write(path, "b", delay=0)
    assert path.read_text() == "a"

    with pytest.raises(TypeError, match="not JSON serializable"):
        _file._script("X", {"a": object()})

    # the state is kept as changed if it can not be written
    writer = _file.DashboardFile(tmp_path / "job.html", interval=0)
    source = _file.DashboardState()
    with source.hold_sync():
        writer.start(source)
    assert writer.state == 1 and not writer.changed
    assert not writer.flush(source)

    writer.changed = True
    failures[:] = [True] * 5
    assert not writer.flush(source)
    assert writer.changed and writer.state == 1

    # the version can not be written: written by the next update
    writer.changed = True
    failures[:] = [True] * 5 + [False]
    assert writer.flush(source)
    assert writer.changed and writer.state == 2
    assert read_script(tmp_path / "job.version.js")["state"] == 1


def test_dashboard_quadratic_cells(kernel):
    """The faces of quadratic cells are triangulated by all their points (e.g. the
    mid-edge points), they are shaded smoothly (without the faces of the triangles)."""

    mesh = fem.mesh.convert(fem.Cube(n=3), order=2)
    region = fem.RegionQuadraticHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=5), field)
    move = fem.math.linsteps([0, 0.1], num=1)
    step = fem.Step([solid], ramp={boundaries["move"]: move}, boundaries=boundaries)

    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    widget = dashboard.widget
    surface = triangulate_surface(mesh)
    points = buffer(widget, "points", (3,), np.float32)
    assert len(points) == surface.interpolation.shape[0] > 26
    assert np.allclose(points, surface.interpolate(mesh.points))
    assert len(buffer(widget, "triangles", (3,))) == 4 * 2 * 24
    assert widget.faces == b""
    assert len(buffer(widget, "edges", (2,))) == 2 * 48
    assert widget.info["surface"] == "hexahedron"
    assert np.allclose(
        values(widget.frame), surface.interpolate(field[0].values).ravel(), atol=1e-6
    )


def test_dashboard_line_cells(kernel):
    """The line cells (e.g. of a truss) are transferred as lines, also together with
    the faces of other cells (the edges of the faces are the mesh lines)."""

    mesh = fem.Mesh(
        points=[[0, 0], [1, 1], [2.0, 0]], cells=[[0, 1], [1, 2]], cell_type="line"
    )
    field = fem.Field(fem.RegionTruss(mesh), dim=2).as_container()
    boundaries = fem.BoundaryDict(fixed=fem.Boundary(field[0], fy=0))
    truss = fem.TrussBody(fem.LinearElastic1D(E=np.ones(2)), field, area=np.ones(2))
    load = fem.PointLoad(field, [1])
    move = fem.math.linsteps([0, -0.1], num=2, axis=1, axes=2)
    step = fem.Step(items=[truss, load], ramp={load: move}, boundaries=boundaries)

    dashboard = fem.DashboardWidgetPlugin()
    fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    widget = dashboard.widget
    lines = buffer(widget, "lines", (2,))
    points = buffer(widget, "points", (3,), np.float32)
    assert len(lines) == 2
    assert np.allclose(
        np.sort(points[lines, :2], axis=1), np.sort(mesh.points[mesh.cells], axis=1)
    )
    assert widget.triangles == widget.edges == widget.faces == b""
    assert widget.info["surface"] == "line"
    assert widget.info["deform"]

    # the faces of quads and line cells on the same points
    rectangle = fem.Rectangle(n=3)
    diagonal = fem.Mesh(rectangle.points, np.array([[0, 4], [4, 8]]), "line")
    dashboard = fem.DashboardWidgetPlugin()
    dashboard.widget = dashboard._state = _plugin._widget_class()()
    regions = [fem.RegionQuad(rectangle), fem.RegionTruss(diagonal)]
    items = [SimpleNamespace(field=fem.Field(r, dim=2).as_container()) for r in regions]
    dashboard._model(SimpleNamespace(steps=[SimpleNamespace(items=items, nsubsteps=1)]))

    widget = dashboard.widget
    assert widget.info["surface"] == "quad, line"
    assert len(buffer(widget, "triangles", (3,))) == 2 * 4
    assert len(buffer(widget, "edges", (2,))) == 12
    assert len(buffer(widget, "lines", (2,))) == 2


def test_dashboard_points_without_cells(kernel):
    "A mesh without supported cells is drawn by all its points."

    mesh = fem.Mesh(np.eye(3), np.arange(3).reshape(-1, 1), "vertex")
    item = SimpleNamespace(field=SimpleNamespace(region=SimpleNamespace(mesh=mesh)))
    dashboard = fem.DashboardWidgetPlugin()
    dashboard.widget = dashboard._state = _plugin._widget_class()()
    dashboard._model(
        SimpleNamespace(steps=[SimpleNamespace(items=[item], nsubsteps=1)])
    )

    widget = dashboard.widget
    assert dashboard._surface is None
    assert widget.info["surface"] == "points"
    assert np.allclose(buffer(widget, "points", (3,), np.float32), mesh.points)
    assert widget.triangles == widget.edges == widget.lines == b""

    # the field values of all points
    values_at_points = np.arange(9.0).reshape(3, 3)
    container = SimpleNamespace(values=values_at_points)
    dashboard._unknowns(container)
    assert np.allclose(values(widget.frame), values_at_points.ravel())


def test_dashboard_module():
    """The module of the dashboard is the WebGL viewer and the dashboard. The top-level
    names of the module and of the script of the page are unique (a module can not
    declare a name twice)."""

    pytest.importorskip("anywidget")
    from felupe.plugins._dashboard._widget import DashboardWidget

    source = module()
    assert source.index("export function createViewer(") < source.index(
        "function render("
    )
    assert DashboardWidget._esm == source

    page = (HERE / "dashboard.html").read_text(encoding="utf-8")
    script = page[page.index("/*" + "MODULE" + "*/") : page.index("</script>")]
    pattern = r"^(?:export )?(?:async )?(?:function|const|let|var|class) (\w+)"
    names = re.findall(pattern, source + script, flags=re.MULTILINE)
    assert len(names) > 20
    assert sorted(names) == sorted(set(names))
