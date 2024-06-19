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


def hello_world(pypardiso=False, parallel=False):
    """Print felupe's hyperelastic hello world.

    Parameters
    ----------
    pypardiso : bool, optional
        Use PyPardiso's sparse solver instead the sparse solver of SciPy (default is
        False).
    parallel : bool, optional
        Flag to activate a threaded vector- and matrix-assembly (default is False).
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

    lines = [
        "mesh = fem.Cube(n=6)",
        "region = fem.RegionHexahedron(mesh)",
        "field = fem.FieldContainer([fem.Field(region, dim=3)])",
        "",
        "boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)",
        "umat = fem.NeoHooke(mu=1, bulk=2)",
        "solid = fem.SolidBody(umat=umat, field=field)",
        "",
        "move = fem.math.linsteps([0, 1], num=5)",
        'ramp = {boundaries["move"]: move}',
        "step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)",
        "",
        "job = fem.Job(steps=[step])",
        f'job.evaluate({", ".join(kwargs)})',
        "",
        'ax = solid.imshow("Principal Values of Cauchy Stress")',
    ]

    print("\n\n".join(["\n".join(imports), "\n".join(lines)]))
