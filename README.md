<p align="center">
  <a href="https://felupe.readthedocs.io/en/latest/?badge=latest"><img src="https://user-images.githubusercontent.com/5793153/235789118-eb03eb25-2556-401d-8a0f-580f37e72f8d.png" height="80px"/></a>
  <p align="center"><i>Finite element analysis for continuum mechanics of solid bodies.</i></p>
</p>

[![FElupe](https://img.shields.io/badge/%F0%9F%94%8D-FElupe-white)](https://felupe.readthedocs.io) [![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![Documentation Status](https://readthedocs.org/projects/felupe/badge/?version=latest)](https://felupe.readthedocs.io/en/latest/?badge=latest) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![codecov](https://codecov.io/gh/adtzlr/felupe/branch/main/graph/badge.svg?token=J2QP6Y6LVH)](https://codecov.io/gh/adtzlr/felupe) [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![PyPI - Downloads](https://img.shields.io/pypi/dm/felupe) [![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://adtzlr.github.io/felupe-web/lab?path=01_Getting-Started.ipynb) <a target="_blank" href="https://colab.research.google.com/github/adtzlr/felupe-web/blob/main/notebooks/colab/01_Getting-Started.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

FElupe is a Python 3.8+ 🐍 finite element analysis package 📦 focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics 🔧 of solid bodies 🚂. Its name is a combination of FE (finite element) and the german word *Lupe* 🔍 (magnifying glass) as a synonym for getting an insight 📖 how a finite element analysis code 🧮 looks like under the hood 🕳️.

<p align="center">
  <a href="https://felupe.readthedocs.io/en/latest/examples/ex01_beam.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex01_beam_thumb.png" height="120px"/></a> <a href="https://felupe.readthedocs.io/en/latest/examples/ex02_plate-with-hole.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex02_plate-with-hole_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex03_plasticity.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex03_plasticity_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex04_balloon.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex04_balloon_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex05_rubber-metal-bushing.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex05_rubber-metal-bushing_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex06_rubber-metal-spring.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex06_rubber-metal-spring_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex07_engine-mount.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex07_engine-mount_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex08_shear.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex08_shear_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex09_numeric-continuation.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex09_numeric-continuation_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex10_poisson-equation.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex10_poisson-equation_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex11_notch-stress.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex11_notch-stress_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex12_foot-bone.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex12_foot-bone_thumb.png" height="120px"/></a> <a 
  href="https://felupe.readthedocs.io/en/latest/examples/ex13_morph-rubber-wheel.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex13_morph-rubber-wheel_thumb.png" height="120px"/></a>
</p>

# Installation
Install Python, fire up 🔥 a terminal and run 🏃

```shell
pip install felupe[all]
```

where `[all]` is a combination of `[io,parallel,plot,progress,view]` and installs all optional dependencies. FElupe has minimal requirements, all available at PyPI supporting all platforms.
* [`numpy`](https://github.com/numpy/numpy) for array operations
* [`scipy`](https://github.com/scipy/scipy) for sparse matrices
* [`tensortrax`](https://github.com/adtzlr/tensortrax) for automatic differentiation

In order to make use of all features of FElupe 💎💰💍👑💎, it is suggested to install all optional dependencies.
* [`einsumt`](https://github.com/mrkwjc/einsumt) for parallel (threaded) assembly
* [`h5py`](https://github.com/h5py/h5py) for writing XDMF result files
* [`matplotlib`](https://github.com/matplotlib/matplotlib) for plotting graphs
* [`meshio`](https://github.com/nschloe/meshio) for mesh-related I/O
* [`pyvista`](https://github.com/pyvista/pyvista) for interactive visualizations
* [`tqdm`](https://github.com/tqdm/tqdm) for showing progress bars during job evaluations

# Getting Started
This tutorial covers the essential high-level parts of creating and solving problems with FElupe. As an introductory example 👨‍🏫, a quarter model of a solid cube with hyperelastic material behaviour is subjected to a uniaxial elongation applied at a clamped end-face. 

First, let’s import FElupe and create a meshed cube out of hexahedron cells with a given number of points per axis. A numeric region, pre-defined for hexahedrons, is created on the mesh. A vector-valued displacement field is initiated on the region. Next, a field container is created on top of this field. 

A uniaxial load case is applied on the displacement field stored inside the field container. This involves setting up symmetry planes as well as the absolute value of the prescribed displacement at the mesh-points on the right-end face of the cube. The right-end face is *clamped* 🛠️: only displacements in direction *x* are allowed. The dict of boundary conditions for this pre-defined load case are returned as `boundaries` and the partitioned degrees of freedom as well as the external displacements are stored within the returned dict `loadcase`. 

An isotropic pseudo-elastic Ogden-Roxburgh Mullins-softening model formulation in combination with an isotropic hyperelastic Neo-Hookean material formulation is applied on the displacement field of a nearly-incompressible solid body. 

A step generates the consecutive substep-movements of a given boundary condition. The step is further added to a list of steps of a job 👩‍💻 (here, a characteristic-curve 📈 job is used). During evaluation ⏳, each substep of each step is solved by an iterative Newton-Rhapson procedure ⚖️. The solution is exported after each completed substep as a time-series ⌚ XDMF file. Finally, the result of the last completed substep is plotted.

For more details beside this high-level code snippet, please have a look at the 📝 [documentation](https://felupe.readthedocs.io).

```python
import felupe as fem

mesh = fem.Cube(n=6)
region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

umat = fem.OgdenRoxburgh(material=fem.NeoHooke(mu=1), r=3, m=1, beta=0)
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

move = fem.math.linsteps([0, 1, 0, 1, 2, 1], num=5)
step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)

job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
job.evaluate(filename="result.xdmf")
fig, ax = job.plot(
    xlabel="Displacement $u$ in mm $\longrightarrow$",
    ylabel="Normal Force $F$ in N $\longrightarrow$",
)

solid.plot("Principal Values of Cauchy Stress").show()
```

![curve](https://user-images.githubusercontent.com/5793153/234382805-d9a56108-9dd7-4f57-a029-571a5a2486a4.svg)

<img src="https://github.com/adtzlr/felupe/assets/5793153/2a236f27-53a5-42aa-a45e-85628ecddf72" alt="Solid Body" width="512px">

# Documentation
The documentation is located [here](https://felupe.readthedocs.io/en/latest/?badge=latest).

# Extension Packages
The capabilities of FElupe may be enhanced with extension packages created by the community.

|                    **Package**                          |                     **Description**                     |
|:-------------------------------------------------------:|:-------------------------------------------------------:|
|  [hyperelastic](https://github.com/adtzlr/hyperelastic) |     Constitutive hyperelastic material formulations     |
|    [matadi](https://github.com/adtzlr/matadi)           | Material Definition with Automatic Differentiation (AD) |
|  [tensortrax](https://github.com/adtzlr/tensortrax)     |     Math on (Hyper-Dual) Tensors with Trailing Axes     |
|    [feplot](https://github.com/ZAARAOUI999/feplot)      |             A visualization tool for FElupe             |

# Testing

To run the FElupe unit tests, check out this repository and type

```
tox
```

# Scientific Publications
A list of articles in which [![FElupe](https://img.shields.io/badge/%F0%9F%94%8D-FElupe-white)](https://felupe.readthedocs.io) is involved. If you use FElupe in your scientific work, please star this repository, cite it [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) and add your publication to this list.

1. A. Dutzler, C. Buzzi, and M. Leitner, "Nondimensional translational characteristics of elastomer components", [Journal of Applied Engineering Design and Simulation](https://jaeds.uitm.edu.my/index.php/jaeds), vol. 1, no. 1. UiTM Press, Universiti Teknologi MARA, Sep. 21, 2021. doi: [10.24191/jaeds.v1i1.20](https://doi.org/10.24191/jaeds.v1i1.20). [![medium-story](https://img.shields.io/badge/medium-story-white)](https://medium.com/@adtzlr/nonlinear-force-displacement-curves-of-rubber-metal-parts-ab7c48448e96)

2. C. Buzzi, A. Dutzler, T. Faethe, J. Lassacher, M. Leitner, and F.-J. Weber, "Development of a tool for estimating 
the characteristic curves of rubber-metal parts", in Proceedings of the [12th International Conference on Railway 
Bogies and Running Gears](https://transportation.bme.hu/2022/11/30/bogie22/), 2022 (ISBN 978-963-9058-46-0).

3. J. Torggler, A. Dutzler, B. Oberdorfer, T. Faethe, H. Müller, C. Buzzi, and M. Leitner, "Investigating Damage Mechanisms in Cord-Rubber Composite Air Spring Bellows of Rail Vehicles and Representative Specimen Design", [Applied Composite Materials](https://www.springer.com/journal/10443). Springer Science and Business Media LLC, Aug. 22, 2023. doi: [10.1007/s10443-023-10157-1](https://link.springer.com/article/10.1007/s10443-023-10157-1). Simulation-related Python scripts are available on GitHub at [adtzlr/fiberreinforcedrubber](https://github.com/adtzlr/fiberreinforcedrubber). <a target="_blank" href="https://colab.research.google.com/github/adtzlr/fiberreinforcedrubber/blob/main/docs/notebooks/ex01_specimen_amplitudes.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Changelog
All notable changes to this project will be documented in [this file](CHANGELOG.md). The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# License
FElupe - finite element analysis (C) 2021-2024 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
