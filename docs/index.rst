FElupe
======

.. image:: _static/logo_dark.svg
   :align: right
   :class: only-dark
   :width: 100px

.. image:: _static/logo_light.svg
   :align: right
   :class: only-light
   :width: 100px

.. admonition:: FElupe - Finite Element Analysis
   :class: tip

   💯 Python ➡️ easy install and intuitive usage 👌🏻


.. image:: https://img.shields.io/pypi/v/felupe.svg
   :target: https://pypi.python.org/pypi/felupe/
   
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0

.. image:: https://madewithlove.now.sh/at?heart=true&colorB=%231f744f&text=Graz+%28Austria%29

.. image:: https://zenodo.org/badge/360657894.svg
   :target: https://zenodo.org/badge/latestdoi/360657894

Introduction
------------

FElupe is a Python 3.7+ finite element analysis package focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics of solid bodies. Its name is a combination of FE (finite element) and the german word Lupe (magnifying glass) as a synonym for getting an insight how a finite element analysis code looks like under the hood. The user code for defining the integral form of equilibrium equations as well as their linearizations over a region are kept as close as possible to the analytical expressions. FElupe is both written in Python and fast in execution times thanks to NumPy and (optional) Numba. 

*No complicated installation, just pure Python.*

Another key feature is the easy and straightforward definition of mixed field formulations for the treatment of nearly incompressible material behavior. In combination with matADi isotropic hyperelastic material formulations are defined in terms of their strain energy density function - gradients (stress) and hessians (elasticity tensor) are evaluated with the help of automatic differentiation. Several useful utilities are available, e.g. the calculation of reaction forces and moments on given boundaries. Finally, results are ready to be exported to VTK or XDMF files (using meshio).

.. admonition:: Highlights
   :class: admonition
   
   + basic high-level finite-element-analysis interface

   + flexible building blocks for finite element assembly
   
   + fast assembly of hyperelastic integral (weak) forms
   
   + straight-forward definition of mixed-field problems
   
   + nearly-incompressible hyperelastic problems of solid mechanics
   
   
   
   

Installation
------------

Install Python, fire up a terminal and run ``pip install felupe[all]``, where ``[all]`` installs all optional dependencies. By default, FElupe only depends on ``numpy`` and ``scipy``. However,  ``meshio``, ``h5py``, ``numba``, ``einsumt`` and ``sparse`` are highly recommended. In order to make use of all features of FElupe, it is suggested to install all optional dependencies. For constitutive material definitions using Automatic Differentation consider also installing `matADi <https://github.com/adtzlr/matadi>`_.

.. code-block:: shell

   pip install felupe[all]

*optional:*

.. code-block:: shell

   pip install matadi

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   tutorial
   examples
   howto
   explanation

.. toctree::
   :maxdepth: 2
   :caption: Reference:
   
   felupe

License
-------

FElupe - finite element analysis (C) 2022 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see `<https://www.gnu.org/licenses/>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`