# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""


def hello_world(
    pypardiso=False,
    parallel=False,
    axisymmetric=False,
    planestrain=False,
    curve=False,
    xdmf=False,
    container=False,
):
    """Print FElupe's hyperelastic hello world.

    Parameters
    ----------
    pypardiso : bool, optional
        Use PyPardiso's sparse solver instead the sparse solver of SciPy (default is
        False).
    parallel : bool, optional
        Flag to activate a threaded vector- and matrix-assembly (default is False).
    axisymmetric : bool, optional
        Flag to create a template for an axisymmetric analysis (default is False).
    planestrain : bool, optional
        Flag to create a template for an plane strain analysis (default is False).
    curve : bool, optional
        Flag to use a characteristic-curve job (default is False).
    xdmf : bool, optional
        Flag to write a XDMF time-series result file (default is False).
    container : bool, optional
        Flag to use a mesh-container with multiple solid bodies (default is False).
    """

    imports = [
        "import felupe as fem",
    ]

    kwargs = []

    if pypardiso:
        imports.append("from pypardiso import spsolve")
        kwargs.append("solver=spsolve")

    if parallel:
        kwargs.append("parallel=True")

    if xdmf:
        kwargs.append('filename="result.xdmf"')

    region = "Region"
    field = "Field"

    if axisymmetric:
        mesh = "Rectangle"
        region += "Quad"
        field += "Axisymmetric"
        dim = 2

    elif planestrain:
        mesh = "Rectangle"
        region += "Quad"
        field += "PlaneStrain"
        dim = 2

    else:
        mesh = "Cube"
        region += "Hexahedron"
        dim = 3

    job = "Job"
    kwargs_job = []
    plot = []
    if curve:
        job = "CharacteristicCurve"
        kwargs_job.append("")
        kwargs_job.append('boundary=boundaries["move"]')
        plot = [
            "fig, ax = job.plot(",
            r'    xlabel=r"Displacement $d$ in mm $\longrightarrow$",',
            r'    ylabel=r"Normal Force $F$ in N $\longrightarrow$",',
            ")",
            "",
        ]

    kwargs_job = ", ".join(kwargs_job)
    plot = "\n".join(plot)

    if not container:
        kwargs = ", ".join(kwargs)

        # fmt: off
        lines = [
            f"mesh = fem.{mesh}(n=6)",
            f"region = fem.{region}(mesh)",
            f"field = fem.FieldContainer([fem.{field}(region, dim={dim})])",
            "",
            "boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)",
            "umat = fem.NeoHooke(mu=1, bulk=2)",
            "solid = fem.SolidBody(umat=umat, field=field)",
            "",
            "move = fem.math.linsteps([0, 1], num=5)",
            'ramp = {boundaries["move"]: move}',
            "step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)",
            "",
            f"job = fem.{job}(steps=[step]{kwargs_job})",
            f"job.evaluate({kwargs})",
            f"{plot}",
            'ax = solid.imshow("Principal Values of Cauchy Stress")',
        ]
        # fmt: on

    else:
        kwargs.insert(0, "x0=field")
        kwargs = ", ".join(kwargs)

        # fmt: off
        lines = [
            "meshes = [",
            f"    fem.{mesh}(n=6),",
            f"    fem.{mesh}(n=6).translate(1, axis=0),",
            "]",
            "container = fem.MeshContainer(meshes, merge=True)",
            "field = fem.Field.from_mesh_container(container).as_container()",
            "",
            "regions = [",
            f"    fem.{region}(container.meshes[0]),",
            f"    fem.{region}(container.meshes[1]),",
            "]",
            "fields = [",
            f"    fem.FieldContainer([fem.{field}(regions[0], dim={dim})]),",
            f"    fem.FieldContainer([fem.{field}(regions[1], dim={dim})]),",
            "]",
            "",
            "boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)",
            "umats = [",
            "    fem.LinearElasticLargeStrain(E=2.1e5, nu=0.3),",
            "    fem.NeoHooke(mu=1),",
            "]",
            "solids = [",
            "    fem.SolidBody(umat=umats[0], field=fields[0]),",
            "    fem.SolidBodyNearlyIncompressible(umat=umats[1], field=fields[1], bulk=5000),",
            "]",
            "",
            "move = fem.math.linsteps([0, 1], num=5)",
            'ramp = {boundaries["move"]: move}',
            "step = fem.Step(items=solids, ramp=ramp, boundaries=boundaries)",
            "",
            f"job = fem.{job}(steps=[step]{kwargs_job})",
            f"job.evaluate({kwargs})",
            f"{plot}",
            'plotter = solids[0].plot(show_undeformed=False, style="wireframe")',
            'solids[1].plot(',
            '    "Principal Values of Cauchy Stress",',
            '    plotter=plotter,',
            '    show_undeformed=False',
            ').show()',
        ]
        # fmt: off

    print("\n\n".join(["\n".join(imports), "\n".join(lines)]))
