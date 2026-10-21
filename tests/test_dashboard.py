# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import warnings

import numpy as np
import pytest

import felupe as fem
from felupe.plugins._dashboard import _plugin, surface


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


def values(frame, index=0):
    "Return the decoded field values of a frame."
    return np.frombuffer(frame["values"][str(index)], dtype=np.float32)


def test_dashboard_without_kernel():
    "Outside of a kernel, a warning is raised once and the dashboard is disabled."

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.2], num=2)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardWidgetPlugin()
    assert not callable(dashboard)

    with pytest.warns(UserWarning, match="requires an IPython kernel"):
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)

    assert dashboard.widget is None
    assert not dashboard._active

    # the warning is raised only once
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)


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
    assert info["npoints"] == mesh.npoints
    assert info["ncells"] == mesh.ncells
    assert info["cell_type"] == "hexahedron"
    assert info["ndof"] == field[0].values.size
    assert info["element"] == "Hexahedron"
    assert info["quadrature"] == "GaussLegendre"
    assert info["quadrature_points_per_cell"] == 8
    assert info["deform"] == 0
    assert info["fields"][0]["name"] == "Displacement"
    assert info["nsubsteps"] == [3, 3]
    assert info["items"] == ["SolidBody (NeoHooke)"]

    # the geometry and the surface of the mesh
    points = np.frombuffer(widget.points, dtype=np.float32).reshape(-1, 3)
    assert np.allclose(points, mesh.points)
    assert len(widget.quads) == 4 * 4 * 24
    assert widget.triangles == b""

    # only the values of the latest converged substep are kept
    frame = widget.frame
    assert (frame["kind"], frame["step"], frame["substep"]) == ("substep", 1, 2)
    assert np.allclose(values(frame), field[0].values.ravel(), atol=1e-6)

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
    assert np.allclose(values(widget.frame), field[0].values.ravel(), atol=1e-6)


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
    temperature = fem.Field(region, dim=1, name="Temperature")
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
    assert info["deform"] == -1
    assert info["fields"][0]["name"] == "Temperature"
    assert info["fields"][0]["display"]
    assert info["surface"] == "quad"
    assert np.allclose(values(dashboard.widget.frame), temperature.values.ravel())


def test_dashboard_newton_without_job(kernel):
    "The hooks of the Newton-Raphson method outside of a job are ignored."

    field, boundaries, solid = block()
    boundaries, loadcase = fem.dof.uniaxial(field, move=-0.1, return_loadcase=True)
    dashboard = fem.DashboardWidgetPlugin()
    fem.newtonraphson(items=[solid], plugins=[dashboard], verbose=0, **loadcase)

    assert dashboard.widget is None
    assert kernel == []


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


def test_surface():
    meshes = {
        "hexahedron": fem.Cube(n=3),
        "tetra": fem.Cube(n=3).triangulate(),
        "hexahedron20": fem.Cube(n=3).add_midpoints_edges(),
        "quad": fem.Rectangle(n=3),
        "triangle": fem.Rectangle(n=3).triangulate(),
    }
    counts = {
        "hexahedron": (0, 24),
        "tetra": (48, 0),
        "hexahedron20": (0, 24),
        "quad": (0, 4),
        "triangle": (8, 0),
    }

    for name, mesh in meshes.items():
        surf = surface(mesh)
        assert (len(surf["tri"]), len(surf["quad"])) == counts[name]

    line = fem.mesh.Line(n=4)
    assert len(surface(line)["line"]) == 3

    unknown = fem.Mesh(np.zeros((1, 3)), np.zeros((1, 1), dtype=int), "vertex")
    assert surface(unknown)["kind"] == "points"


if __name__ == "__main__":
    test_surface()
