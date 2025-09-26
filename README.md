<p align="center">
  <a href="https://felupe.readthedocs.io/en/latest/?badge=latest"><img src="https://github.com/user-attachments/assets/cd86b3fb-db8e-40ed-b8c5-f879f032a57c" height="80"></a>
  <p align="center"><i>Finite element analysis for continuum mechanics of solid bodies.</i></p>
</p>

[![FElupe](https://img.shields.io/badge/%F0%9F%94%8D-FElupe-white)](https://felupe.readthedocs.io/en/stable/) [![PyPI version shields.io](https://img.shields.io/pypi/v/felupe.svg)](https://pypi.python.org/pypi/felupe/) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/felupe.svg)](https://anaconda.org/conda-forge/felupe) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/felupe)
 [![Documentation Status](https://readthedocs.org/projects/felupe/badge/?version=stable)](https://felupe.readthedocs.io/en/stable/?badge=stable) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![codecov](https://codecov.io/gh/adtzlr/felupe/branch/main/graph/badge.svg?token=J2QP6Y6LVH)](https://codecov.io/gh/adtzlr/felupe) [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) [![pyOpenSci Peer-Reviewed](https://pyopensci.org/badges/peer-reviewed.svg)](https://github.com/pyOpenSci/software-review/issues/212) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![PyPI - Downloads](https://img.shields.io/pypi/dm/felupe) [![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://adtzlr.github.io/felupe-web/lab?path=01_Getting-Started.ipynb) <a target="_blank" href="https://colab.research.google.com/github/adtzlr/felupe-web/blob/main/notebooks/colab/01_Getting-Started.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://felupe-web.streamlit.app)

FElupe is a Python 3.9+ finite element analysis package focusing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. This package is intended for scientific research, but is also suitable for running nonlinear simulations in general. In addition to the transformation of general weak forms into sparse vectors and matrices, FElupe provides an efficient high-level abstraction layer for the simulation of the deformation of solid bodies.

<p align="center">
  <a href="https://felupe.readthedocs.io/en/latest/examples/ex01_beam.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex01_beam_thumb.png" height="100px"/></a> <a href="https://felupe.readthedocs.io/en/latest/examples/ex02_plate-with-hole.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex02_plate-with-hole_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex03_plasticity.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex03_plasticity_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex04_balloon.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex04_balloon_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex05_rubber-metal-bushing.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex05_rubber-metal-bushing_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex06_rubber-metal-spring.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex06_rubber-metal-spring_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex07_engine-mount.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex07_engine-mount_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex08_shear.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex08_shear_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex09_numeric-continuation.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex09_numeric-continuation_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex10_poisson-equation.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex10_poisson-equation_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex11_notch-stress.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex11_notch-stress_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex12_foot-bone.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex12_foot-bone_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex13_morph-rubber-wheel.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex13_morph-rubber-wheel_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex14_hyperelasticity.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex14_hyperelasticity_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex15_hexmesh-metacone.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex15_hexmesh-metacone_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex16_deeplearning-torch.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex16_deeplearning-torch_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex17_torsion-gif.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex17_torsion-gif_thumb.gif" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex18_nonlinear-viscoelasticity-newton.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex18_nonlinear-viscoelasticity-newton_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex19_taylor-hood.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex19_taylor-hood_thumb.png" height="100px"/></a> <a
  href="https://felupe.readthedocs.io/en/latest/examples/ex20_third-medium-contact.html"><img src="https://felupe.readthedocs.io/en/latest/_images/sphx_glr_ex20_third-medium-contact_thumb.gif" height="100px"/></a>
</p>

## ✨ Highlights
- ✅ 100% Python package built with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)
- ✅ easy to learn and productive high-level [API](https://felupe.readthedocs.io/en/latest/felupe.html)
- ✅ nonlinear deformation of [solid bodies](https://felupe.readthedocs.io/en/latest/felupe/mechanics.html#felupe.SolidBody)
- ✅ interactive views on meshes, fields and solid bodies (using [PyVista](https://pyvista.org/))
- ✅ typical [finite elements](https://felupe.readthedocs.io/en/latest/felupe/element.html)
- ✅ cartesian, axisymmetric, plane strain and mixed fields
- ✅ [hyperelastic material models](https://felupe.readthedocs.io/en/latest/felupe/constitution.html) with automatic differentiation

Efficient [NumPy](https://numpy.org/)-based math is realized by element-wise operating *trailing axes* [[1]](https://doi.org/10.21105/joss.02369). The finite element method, as used in FElupe, is based on [[2]](https://doi.org/10.1017/cbo9780511755446), [[3]]() and [[4]](https://doi.org/10.1016/c2009-0-24909-9). Related scientific articles are listed in the sections of the [API reference](https://felupe.readthedocs.io/en/latest/felupe.html).

> [!NOTE]
> The name *FElupe* is a combination of FE (finite element) and the german word *Lupe* (magnifying glass) as a synonym for getting an insight how a finite element analysis code looks like *under the hood*.

## 📦 Installation
Install Python, open a terminal and run

```shell
pip install felupe[all]
```

The [documentation](https://felupe.readthedocs.io/en/stable/) covers more details, like required and optional dependencies and how to install the latest development version.

## 🚀 Getting Started
This minimal code-block demonstrates a nonlinear simulation of a hyperelastic cube under compression.

```python
import felupe as fem

mesh = fem.Cube(n=8)
region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer(fields=[fem.Field(region, dim=3)])

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, move=-0.3)
solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=5), field=field)

step = fem.Step(items=[solid], boundaries=boundaries)
job = fem.Job(steps=[step]).evaluate()

solid.plot("Principal Values of Cauchy Stress").show()
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/9f6e1267-b624-44ae-a79c-0b337369591a" alt="Solid Body" height="200px">
</p>

## 📖 Documentation
The documentation is located [here](https://felupe.readthedocs.io/en/stable/).

## 🧩 Extension Packages
The capabilities of FElupe may be enhanced with extension packages created by the community.

|                    **Package**                          |                     **Description**                     |
|:-------------------------------------------------------:|:-------------------------------------------------------:|
|  [hyperelastic](https://github.com/adtzlr/hyperelastic) |     Constitutive hyperelastic material formulations     |
|    [matadi](https://github.com/adtzlr/matadi)           | Material Definition with Automatic Differentiation (AD) |
|  [tensortrax](https://github.com/adtzlr/tensortrax)     |      Differentiable Tensors based on NumPy Arrays       |
|    [feplot](https://github.com/ZAARAOUI999/feplot)      |             A visualization tool for FElupe             |

## 🛠️ Testing

To run the FElupe unit tests, check out this repository and type

```
tox
```

## 📝 Scientific Publications
A list of articles in which [![FElupe](https://img.shields.io/badge/%F0%9F%94%8D-FElupe-white)](https://felupe.readthedocs.io/en/stable) is involved. If you use FElupe in your scientific work, please star this repository, cite it [![DOI](https://zenodo.org/badge/360657894.svg)](https://zenodo.org/badge/latestdoi/360657894) and add your publication to this list.

<details><summary>Expand the list...</summary>

- A. Dutzler, C. Buzzi, and M. Leitner, "Nondimensional translational characteristics of elastomer components", [Journal of Applied Engineering Design and Simulation](https://jaeds.uitm.edu.my/index.php/jaeds), vol. 1, no. 1. UiTM Press, Universiti Teknologi MARA, Sep. 21, 2021. doi: [10.24191/jaeds.v1i1.20](https://doi.org/10.24191/jaeds.v1i1.20). [![medium-story](https://img.shields.io/badge/medium-story-white)](https://medium.com/@adtzlr/nonlinear-force-displacement-curves-of-rubber-metal-parts-ab7c48448e96)

- C. Buzzi, A. Dutzler, T. Faethe, J. Lassacher, M. Leitner, and F.-J. Weber, "Development of a tool for estimating
the characteristic curves of rubber-metal parts", in Proceedings of the [12th International Conference on Railway 
Bogies and Running Gears](https://transportation.bme.hu/2022/11/30/bogie22/), 2022 (ISBN 978-963-9058-46-0).

- J. Torggler, A. Dutzler, B. Oberdorfer, T. Faethe, H. Müller, C. Buzzi, and M. Leitner, "Investigating Damage Mechanisms in Cord-Rubber Composite Air Spring Bellows of Rail Vehicles and Representative Specimen Design", [Applied Composite Materials](https://www.springer.com/journal/10443). Springer Science and Business Media LLC, Aug. 22, 2023. doi: [10.1007/s10443-023-10157-1](https://link.springer.com/article/10.1007/s10443-023-10157-1). Simulation-related Python scripts are available on GitHub at [adtzlr/fiberreinforcedrubber](https://github.com/adtzlr/fiberreinforcedrubber). <a target="_blank" href="https://colab.research.google.com/github/adtzlr/fiberreinforcedrubber/blob/main/docs/notebooks/ex01_specimen_amplitudes.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

</details>

## 📄 Changelog
All notable changes to this project will be documented in [this file](CHANGELOG.md). The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 📚 References
[[1]](https://doi.org/10.21105/joss.02369) T. Gustafsson and G. McBain, "*[scikit-fem](https://github.com/kinnala/scikit-fem): A Python package for finite element assembly*", Journal of Open Source Software, vol. 5, no. 52. The Open Journal, p. 2369, Aug. 21, 2020. [![DOI:10.21105/joss.02369](https://zenodo.org/badge/DOI/10.21105/joss.02369.svg)](https://doi.org/10.21105/joss.02369).

[[2]](https://doi.org/10.1017/cbo9780511755446) J. Bonet and R. D. Wood, "*Nonlinear Continuum Mechanics for Finite Element Analysis*". Cambridge University Press, Mar. 13, 2008. [![DOI:10.1017/cbo9780511755446](https://zenodo.org/badge/DOI/10.1017/cbo9780511755446.svg)](https://doi.org/10.1017/cbo9780511755446).

[[3]]() K. J. Bathe, "*Finite Element Procedures*". 2006, isbn: 978-0-9790049-0-2.

[[4]](https://doi.org/10.1016/c2009-0-24909-9) O. C. Zienkiewicz, R. L. Taylor and J. Z. Zhu, "*The Finite Element Method: its Basis and Fundamentals*". Elsevier, 2013. [![DOI:10.1016/c2009-0-24909-9](https://zenodo.org/badge/DOI/10.1016/c2009-0-24909-9.svg)](https://doi.org/10.1016/c2009-0-24909-9).

## 🔓 License
FElupe - finite element analysis (C) 2021-2025 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
