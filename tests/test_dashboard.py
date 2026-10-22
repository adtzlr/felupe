# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import warnings
from types import SimpleNamespace

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


def values(frame):
    "Return the decoded field values of a frame."
    return np.frombuffer(frame["values"], dtype=np.float32)


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

    # the hooks of a failed substep are ignored, the error is raised
    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.8], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    with pytest.raises(fem.NewtonConvergenceError):
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

    # the outer surface of both cubes, without the faces at the interface
    quads = np.frombuffer(widget.quads, dtype=np.uint32).reshape(-1, 4)
    assert len(quads) == 40
    assert quads.max() < 45

    assert np.allclose(values(widget.frame), field[0].values.ravel(), atol=1e-6)
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
    assert len(np.frombuffer(dashboard.widget.quads, dtype=np.uint32)) == 4 * 24


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
    assert len(np.frombuffer(widget.quads, dtype=np.uint32)) == 4 * 54
    assert widget.path["status"] == "ccccc"
    assert np.allclose(values(widget.frame), temperature.values.ravel())
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
    assert np.allclose(values(dashboard.widget.frame), field[0].values.ravel())


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
    dashboard.widget = _plugin._widget_class()()
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
    assert surface([])["kind"] == "points"

    from felupe.plugins._dashboard._surface import _unique_faces

    assert _unique_faces(np.zeros((0, 4), dtype=int)).shape == (0, 4)

    # several meshes with a common points-array: the inner faces at the interface
    # are removed
    cubes = [fem.Cube(n=3), fem.Cube(n=3).translate(1, axis=0)]
    container = fem.MeshContainer(cubes, merge=True)
    assert len(surface(container.meshes)["quad"]) == 40

    hexahedron, tetra = fem.Cube(n=2), fem.Cube(n=2).translate(1, axis=0).triangulate()
    container = fem.MeshContainer([hexahedron, tetra], merge=True)
    surf = surface(container.meshes)
    assert surf["kind"] == "hexahedron, tetra"

    # a quad and two triangles at the interface are not identical, both are kept
    assert len(surf["quad"]) == 6
    assert len(surf["tri"]) == 12


if __name__ == "__main__":
    test_surface()
