# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import json
import urllib.request

import numpy as np
import pytest

import felupe as fem
from felupe.plugins._dashboard import surface


def events(plugin, kind=None):
    "Return the (decoded) events of the dashboard, optionally of a single type."
    decoded = [json.loads(m) for m in plugin.broadcaster.history]
    return [e for e in decoded if kind is None or e["type"] == kind]


def block(n=3):
    mesh = fem.Cube(n=n)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
    return field, boundaries, solid


def test_dashboard_job(tmp_path):
    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.2], num=2)
    steps = [
        fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries),
        fem.Step(
            items=[solid], ramp={boundaries["move"]: move[::-1]}, boundaries=boundaries
        ),
    ]

    filename = tmp_path / "job.html"
    dashboard = fem.DashboardPlugin(open_browser=False, filename=filename, title="T")
    assert not callable(dashboard)

    fem.Job(steps=steps, plugins=[dashboard]).evaluate(verbose=0)

    try:
        model = events(dashboard, "model")[0]
        assert model["npoints"] == field.region.mesh.npoints
        assert model["ncells"] == field.region.mesh.ncells
        assert model["ndof"] == field[0].values.size
        assert model["deform"] == 0
        assert model["fields"][0]["name"] == "Displacement"
        assert model["steps"][0]["items"][0] == dict(kind="SolidBody", umat="NeoHooke")
        assert model["title"] == "T"

        substeps = events(dashboard, "substep_end")
        assert len(substeps) == 6
        assert all(e["failures"] == 0 for e in substeps)

        frames = events(dashboard, "frame")
        assert [f["kind"] for f in frames] == ["initial"] + 6 * ["substep"]

        dofs = events(dashboard, "dofs")[0]
        assert dofs["active"] + dofs["prescribed"] == model["ndof"]

        matrix = events(dashboard, "matrix")[0]
        assert matrix["shape"] == [model["ndof"], model["ndof"]]

        iterations = events(dashboard, "iteration")
        assert sum(e["iterations"] for e in substeps) == len(iterations)
        assert iterations[-1]["success"]

        end = events(dashboard, "job_end")[0]
        assert end["status"] == "finished"

        # the values of the last frame are the values of the field
        values = np.frombuffer(
            base64.b64decode(frames[-1]["values"]["0"]), dtype=np.float32
        )
        assert np.allclose(values, field[0].values.ravel(), atol=1e-6)

        # the self-contained file replays all events
        page = filename.read_text(encoding="utf-8")
        assert "window.FELUPE_EVENTS" in page
        assert page.count('"type":"substep_end"') == 6

        # the page and the events are served
        with urllib.request.urlopen(dashboard.url, timeout=5) as response:
            assert b"FElupe" in response.read()

        with urllib.request.urlopen(dashboard.url + "events", timeout=5) as response:
            assert response.headers["Content-Type"] == "text/event-stream"
            assert response.readline().startswith(b"data: ")

        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(dashboard.url + "missing", timeout=5)

        # a repeated evaluation resets the dashboard
        fem.Job(steps=steps[:1], plugins=[dashboard]).evaluate(verbose=0)
        assert len(events(dashboard, "substep_end")) == 3
        assert len(events(dashboard, "model")) == 1

    finally:
        dashboard.close()

    assert dashboard.server is None
    dashboard.close()


@pytest.mark.parametrize("first", [True, False])
def test_dashboard_cutback(first):
    "The attempts of a cutback are observed in any order of the plugins."

    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.7], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    dashboard = fem.DashboardPlugin(open_browser=False, iterations=True, interval=0)
    cutback = fem.CutbackPlugin()
    plugins = [dashboard, cutback] if first else [cutback, dashboard]

    try:
        fem.Job(steps=[step], plugins=plugins).evaluate(verbose=0)
    finally:
        dashboard.close()

    substep = events(dashboard, "substep_end")[-1]
    assert substep["load_factors"] == cutback.load_factors[-1]
    assert substep["failures"] == cutback.cutbacks[-1]
    assert substep["attempts"] == 1 + len(cutback.load_factors[-1])

    attempts = [e for e in events(dashboard, "attempt_end") if e["substep"] == 1]
    assert [e["success"] for e in attempts] == [False] + len(
        cutback.load_factors[-1]
    ) * [True]
    assert attempts[0]["message"]

    # the transient frame of the last non-converged iteration
    assert dashboard.broadcaster._transient[0] > 0


def test_dashboard_failed_job():
    field, boundaries, solid = block()
    move = fem.math.linsteps([0, -0.8], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    dashboard = fem.DashboardPlugin(open_browser=False, frames=False)

    try:
        with pytest.raises(fem.NewtonConvergenceError):
            fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)
    finally:
        dashboard.close()

    end = events(dashboard, "job_end")[0]
    assert end["status"] == "failed"
    assert end["message"].startswith("NewtonConvergenceError")

    failed = events(dashboard, "attempt_end")[-1]
    assert not failed["success"]


def test_dashboard_scalar_field():
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
    dashboard = fem.DashboardPlugin(open_browser=False)

    try:
        fem.Job(steps=[step], plugins=[dashboard]).evaluate(verbose=0)
    finally:
        dashboard.close()

    model = events(dashboard, "model")[0]
    assert model["deform"] == -1
    assert model["fields"][0]["name"] == "Temperature"
    assert model["fields"][0]["display"]
    assert model["surface"]["kind"] == "quad"


def test_dashboard_newton_without_job():
    "The hooks of the Newton-Raphson method outside of a job are ignored."

    field, boundaries, solid = block()
    boundaries, loadcase = fem.dof.uniaxial(field, move=-0.1, return_loadcase=True)
    dashboard = fem.DashboardPlugin(open_browser=False)
    fem.newtonraphson(items=[solid], plugins=[dashboard], verbose=0, **loadcase)

    assert dashboard.server is None
    assert dashboard.broadcaster.history == []


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
