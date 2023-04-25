# 🔍 FElupe

[![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![Documentation Status](https://readthedocs.org/projects/felupe/badge/?version=latest)](https://felupe.readthedocs.io/en/latest/?badge=latest) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/gh/adtzlr/felupe/branch/main/graph/badge.svg?token=J2QP6Y6LVH)](https://codecov.io/gh/adtzlr/felupe) [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![GitHub Repo stars](https://img.shields.io/github/stars/adtzlr/felupe?logo=github) ![PyPI - Downloads](https://img.shields.io/pypi/dm/felupe) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adtzlr/felupe/HEAD) [![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://adtzlr.github.io/felupe-web/lab?path=01_Getting+Started.ipynb) <a target="_blank" href="https://colab.research.google.com/github/adtzlr/felupe-web/blob/main/content/01_Getting%20Started.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

> Finite Element Analysis

<p align="center">
  <img src="https://raw.githubusercontent.com/adtzlr/felupe/main/docs/_static/logo_light.svg" height="120px"/> <a href="https://felupe.readthedocs.io/en/latest/examples/rubberspring.html"><img src="https://user-images.githubusercontent.com/5793153/230604246-5a416081-6777-4f33-afdf-efdb51338722.png" height="120px"/></a> <a href="https://felupe.readthedocs.io/en/latest/examples/platewithhole.html"><img src="https://user-images.githubusercontent.com/5793153/230604587-42e3e339-e08c-4cc8-8000-f7046a8d95df.png" height="120px"/></a>
</p>

FElupe is a Python 3.7+ finite element analysis package focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. Its name is a combination of FE (finite element) and the german word *Lupe* (magnifying glass) as a synonym for getting an insight how a finite element analysis code looks like under the hood.

# Installation
Install Python, fire up a terminal and run

```shell
pip install felupe[all]
```

where `[all]` installs all optional dependencies. By default, FElupe depends on `numpy`, `scipy` and `tensortrax`. In order to make use of all features of FElupe, it is suggested to install all optional dependencies (`einsumt`, `h5py`, `matplotlib`, `meshio` and `pyvista`).

# Getting Started
This tutorial covers the essential high-level parts of creating and solving problems with FElupe. As an introductory example, a quarter model of a solid cube with hyperelastic material behaviour is subjected to a uniaxial elongation applied at a clamped end-face. 

First, let’s import FElupe and create a meshed cube out of hexahedron cells with 6 points per axis. A numeric region, pre-defined for hexahedrons, is created on the mesh. A vector-valued displacement field is initiated on the region. Next, a field container is created on top of this field. 

A uniaxial load case is applied on the displacement field stored inside the field container. This involves setting up symmetry planes as well as the absolute value of the prescribed displacement at the mesh-points on the right-end face of the cube. The right-end face is *clamped*: only displacements in direction *x* are allowed. The dict of boundary conditions for this pre-defined load case are returned as `boundaries` and the partitioned degrees of freedom as well as the external displacements are stored within the returned dict `loadcase`. 

An isotropic pseudo-elastic Ogden-Roxburgh Mullins-softening model formulation in combination with an isotropic hyperelastic Neo-Hookean material formulation is applied on the displacement field of a nearly-incompressible solid body. 

A step generates the consecutive substep-movements of a given boundary condition. The step is further added to a list of steps of a job (here, a characteristic-curve job is used). During evaluation, each substep of each step is solved by an iterative Newton-Rhapson procedure. The solution is exported after each completed substep as a time-series XDMF file. Finally, the result of the last completed substep is plotted.

For more details beside this high-level code snippet, please have a look at the [documentation](https://felupe.readthedocs.io/en/latest/?badge=latest).

```python
import felupe as fem

# create a hexahedron-region on a cube
mesh = fem.Cube(n=6)
region = fem.RegionHexahedron(mesh)

# add a field container (with a vector-valued displacement field)
field = fem.FieldContainer([fem.Field(region, dim=3)])

# apply a uniaxial elongation on the cube
boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

# define the constitutive material behaviour
# and create a nearly-incompressible (u,p,J - formulation) solid body
umat = fem.OgdenRoxburgh(material=fem.NeoHooke(mu=1), r=3, m=1, beta=0)
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

# prepare a step with substeps
move = fem.math.linsteps([0, 1, 0, 1, 2, 1], num=5)
step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)

# add the step to a job, evaluate all substeps and create a plot
job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
job.evaluate(filename="result.xdmf")
fig, ax = job.plot(
    xlabel="Displacement $u$ in mm $\longrightarrow$",
    ylabel="Normal Force $F$ in N $\longrightarrow$",
)

# visualize the results
view = fem.View(field)
view.plot("Principal Values of Logarithmic Strain").show()
```

![curve](https://user-images.githubusercontent.com/5793153/234382805-d9a56108-9dd7-4f57-a029-571a5a2486a4.svg)

![cube](https://user-images.githubusercontent.com/5793153/234405093-2f5201c1-3bba-46ee-bd91-af87813609d9.png)

# Documentation
The documentation is located [here](https://felupe.readthedocs.io/en/latest/?badge=latest).

# Extension Packages
The capabilities of FElupe may be enhanced with extension packages created by the community.

| **Package** |                     **Description**                     |                **Link**               |
|:-----------:|:-------------------------------------------------------:|:-------------------------------------:|
|  tensortrax |     Math on (Hyper-Dual) Tensors with Trailing Axes     |  https://github.com/adtzlr/tensortrax |
|    matADi   | Material Definition with Automatic Differentiation (AD) |    https://github.com/adtzlr/matadi   |
|    FEplot   |             A visualization tool for FElupe             | https://github.com/ZAARAOUI999/feplot |

# Changelog
All notable changes to this project will be documented in [this file](CHANGELOG.md). The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# License
FElupe - finite element analysis (C) 2023 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
