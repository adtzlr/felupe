r"""
Building Blocks
---------------

.. topic:: Learn the building blocks of FElupe.

   * create a meshed cube with hexahedron elements

   * setup your own numeric region with a mesh, an element and a quadrature

   * add a displacement field to a field container

   * define your own Neo-Hookean material formulation

   * apply your own boundary conditions

   * solve the problem (create your own Newton-Rhapson iteration loop)

   * export the displaced mesh along with cauchy stress projected to mesh-points

Start setting up a problem in FElupe by the creation of a numeric :class:`~felupe.Region` with a geometry :class:`~felupe.Mesh`, a finite **Element** formulation :class:`~felupe.Hexahedron` and a **Quadrature** rule :class:`~felupe.GaussLegendre`.

.. image:: examples/extut03_building_blocks_sketch.svg
   :width: 600px
"""

# sphinx_gallery_thumbnail_number = -1
import felupe as fem

mesh = fem.Cube(n=6)
element = fem.Hexahedron()
quadrature = fem.GaussLegendre(order=1, dim=3)

mesh.plot().show()

# %%
# An overview of the mesh-metadata is available in the console.
print(mesh)

# %%
# Region
# ~~~~~~
# A :class:`~felupe.Region` essentially pre-calculates element shape/ansatz/basis
# functions and derivatives evaluated at every quadrature point of every cell w.r.t. the
# undeformed coordinates (as attribute :attr:`~felupe.Region.dhdX`).
#
# ..  note::
#     By using a template region like :class:`~felupe.RegionHexahedron`, only the mesh
#     has to be created. The element formulation and the appropriate quadrature scheme
#     are chosen automatically.
region = fem.Region(mesh, element, quadrature)
# region = fem.RegionHexahedron(mesh)
print(region)

# %%
# The scheme of the region, i.e. the element with its point labels and the integration
# points of the quadrature rule may be plotted.
region.plot().show()

# %%
# An array containing products of quadrature weights multiplied by the determinants of
# the (geometric) jacobians is stored as the array of (undeformed) differential volumes
# :attr:`~felupe.Region.dV`. The sum of all differential volumes gives the total
# (undeformed) volume of the region.
dV = region.dV
V = dV.sum()

# %%
# Field
# ~~~~~
# In a next step, a displacement :class:`~felupe.Field` is added to the
# :class:`~felupe.Region`. This may be either a scalar- or a vector-valued field. The
# values at mesh-points are obtained with the attribute :attr:`~felupe.Field.values`.
# Interpolated field values at quadrature points are calculated with the
# :meth:`~felupe.Field.interpolate` method. Additionally, the displacement gradient
# w.r.t. the undeformed coordinates is calculated for every quadrature point of every
# cell in the region with the field method :meth:`~felupe.Field.grad`. A generalized
# extraction method
# :meth:`extract(grad=True, add_identity=True, sym=False) <felupe.Field.extract>` allows
# several arguments to be passed. This involves whether the gradient or the values are
# extracted. If the gradient is extracted, the identity matrix may be added to the
# gradient (useful for the calculation of the deformation gradient). Optionally, the
# symmetric part is returned (small strain tensor).
displacement = fem.Field(region, dim=3)

u = displacement.values
v = displacement.interpolate()
dudX = displacement.grad()


# %%
# Next, the field is added to a :class:`~felupe.FieldContainer`, which holds one or more
# fields. Like a :class:`~felupe.Field`, the :class:`~felupe.FieldContainer` also
# provides the
# :meth:`extract(grad=True, add_identity=True, sym=False) <felupe.FieldContainer.extract>`
# method, returning a list of interpolated field values or gradients.
field = fem.FieldContainer([displacement])
print(field)

# %%
# The deformation gradient is obtained by a sum of the identity and the displacement
# gradient. The :meth:`~felupe.FieldContainer.extract`-method generates a list with
# extracted values at the integration points for each field in the field container.
# To evaluate the deformation gradient tensor of the first field of a container, the
# identity matrix is added to the gradient of the field.
F = field.extract(grad=True, sym=False, add_identity=True)

# %%
# Methods to evaluate the deformation gradient as well as strain measures are
# provided in :class:`FieldContainer.evaluate <felupe.field.EvaluateFieldContainer>`.
import numpy as np

log_strain = field.evaluate.strain(fun=lambda stretch: np.log(stretch), tensor=True)
principal_stretches = field.evaluate.strain(fun=lambda stretch: stretch, tensor=False)

# %%
# Constitution
# ~~~~~~~~~~~~
# The material behavior has to be provided by the first Piola-Kirchhoff stress tensor as
# a function of the deformation gradient. FElupe provides a comprehensive constitutive
# library (see :ref:`felupe-api-constitution`) including :class:`~felupe.NeoHooke`,
# :class:`~felupe.LinearElastic` and a generalized Hu-Washizu (u,p,J)
# :class:`~felupe.ThreeFieldVariation`. By alternative, an isotropic material
# formulation :class:`~felupe.constitution.tensortrax.Hyperelastic` is defined by a
# strain energy density function where both variation (stress) and linearization
# (elasticity) are carried out by automatic differentiation. The latter one is
# demonstrated here with a nearly-incompressible version of the Neo-Hookean material
# model formulation.
#
# .. note::
#    FElupe supports different backends for automatic differentiation and uses
#    `tensortrax <https://github.com/adtzlr/tensortrax>`_ by default. It is important to
#    use only differentiable math-functions from the backend, e.g.
#    NumPy-like modules `tensortrax.math <https://github.com/adtzlr/tensortrax>`_,
#    `tensortrax.math.linalg <https://github.com/adtzlr/tensortrax>`_ or
#    `tensortrax.math.special <https://github.com/adtzlr/tensortrax>`_.
#    `tensortrax <https://github.com/adtzlr/tensortrax>`_ is a dependency of FElupe and
#    hence, is installed along with `FElupe <https://pypi.org/project/felupe/>`_.
#
# .. math::
#
#    \psi = \frac{\mu}{2} \left( J^{-2/3}\ \text{tr}(\boldsymbol{C}) - 3 \right)
#         + \frac{K}{2} \left( J - 1 \right)^2
import tensortrax.math as tm

import felupe.constitution.tensortrax as mat


def W(C, mu, bulk):
    "Isotropic hyperelastic Neo-Hookean material formulation."

    J = tm.sqrt(tm.linalg.det(C))

    return mu / 2 * (J ** (-2 / 3) * tm.trace(C) - 3) + bulk * (J - 1) ** 2 / 2


umat = mat.Hyperelastic(W, mu=1.0, bulk=2.0)

P = umat.gradient
A = umat.hessian

# %%
# Boundary Conditions
# ~~~~~~~~~~~~~~~~~~~
# Next we enforce boundary conditions on the displacement field. Boundary conditions are
# stored as a dictionary of multiple :class:`~felupe.Boundary` instances. First, the
# left end of the cube is fixed. Displacements on the right end are fixed in directions
# y and z whereas displacements in direction x are prescribed with a user-defined
# ``value=0.5``. A :class:`~felupe.Boundary` holds useful attributes like
# :attr:`~felupe.Boundary.points` or :attr:`~felupe.Boundary.dof`.
f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

boundaries = fem.BoundaryDict()

boundaries["left"] = fem.Boundary(displacement, fx=f0)
boundaries["right"] = fem.Boundary(displacement, fx=f1, skip=(1, 0, 0))
boundaries["move"] = fem.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=0.5)

boundaries.plot().show()

# %%
# Partition of deegrees of freedom
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The separation of active and inactive degrees of freedom is performed by a so-called
# :func:`~felupe.dof.partition`. External values of prescribed displacement degrees of
# freedom are obtained by the application of the boundary values on the displacement
# field.
dof0, dof1 = fem.dof.partition(field, boundaries)
ext0 = fem.dof.apply(field, boundaries, dof0)

# %%
# Integral forms of equilibrium equations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The integral (or weak) forms of equilibrium equations are defined by the
# :class:`~felupe.IntegralForm` class, see Eq. :eq:`eq_integral_form`. The pre-evaluated
# function of interest has to be passed as the ``fun`` argument and the test field as
# the ``v`` argument. By setting ``grad_v=[True]`` (default), FElupe passes the gradient
# of the first test field to the integral form. FElupe assumes a linear form if
# ``u=None`` (default) and creates a bilinear form if a field is passed to the trial
# field argument ``u``.
#
# .. math::
#    :label: eq_integral_form
#
#    \int_V P_{iJ} : \frac{\partial \delta u_i}{\partial X_J} \ dV
#        \qquad \text{and} \qquad \int_V \frac{\partial \delta u_i} {\partial X_J} :
#            \mathbb{A}_{iJkL} : \frac{\partial u_k}{\partial X_L} \ dV
linearform = fem.IntegralForm(P(F)[:-1], field, dV, grad_v=[True])
bilinearform = fem.IntegralForm(A(F), field, dV, u=field, grad_v=[True], grad_u=[True])

# %%
# The assembly of both forms lead to the (point-based) internal force vector and the
# (sparse) stiffness matrix.
r = linearform.assemble()
K = bilinearform.assemble()

# %%
# Prepare (partition) and solve the linearized equation system
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In order to solve the linearized equation system a :func:`~felupe.solve.partition`
# into active and inactive degrees of freedom has to be performed, see Eqs.
# :eq:`eq_partition`. This system may then be passed to the (sparse direct) solver.
# Given a set of nonlinear equilibrium equations :math:`\boldsymbol{g}` the unknowns
# :math:`\boldsymbol{u}` are found by linearization at a valid initial state of
# equilibrium and an iterative Newton-Rhapson solution prodecure. The incremental values
# of inactive degrees of freedom are given as the difference of external prescribed and
# current values of unknowns. The (linear) solution is equal to the first result of a
# Newton-Rhapson iterative solution procedure. The solution ``du`` is finally added to
# the displacement field.
#
# .. math::
#    :label: eq_partition
#
#    \boldsymbol{g}_1(\boldsymbol{u}) &= -\boldsymbol{r}_1(\boldsymbol{u})
#        + \boldsymbol{f}_1
#
#    \boldsymbol{g}_1(\boldsymbol{u} + d\boldsymbol{u}) &\approx -\boldsymbol{r}_1 +
#        \boldsymbol{f}_1 - \frac{\partial \boldsymbol{r}_1}{\partial \boldsymbol{u}_1}
#        \ d\boldsymbol{u}_1 - \frac{
#            \partial \boldsymbol{r}_1}{\partial \boldsymbol{u}_0} \ d\boldsymbol{u}_0
#        = \boldsymbol{0}
#
#    d\boldsymbol{u}_0 &= \boldsymbol{u}_0^{(ext)} - \boldsymbol{u}_0
#
#    \text{solve} \qquad \boldsymbol{K}_{11}\ d\boldsymbol{u}_1 &=
#        \boldsymbol{g}_1 - \boldsymbol{K}_{10}\ d\boldsymbol{u}_{0}
#
#    \boldsymbol{u}_0 &+= d\boldsymbol{u}_0
#
#    \boldsymbol{u}_1 &+= d\boldsymbol{u}_1
#
# The default solver of FElupe is
# `SuperLU <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve>`_
# provided by the sparse package of `SciPy <https://docs.scipy.org>`_. A significantly
# faster alternative is `pypardiso <https://pypi.org/project/pypardiso/>`_ which may be
# installed from PyPI with ``pip install pypardiso`` (not included with FElupe). The
# optional argument ``solver`` of :func:`~felupe.solve.solve` accepts any user-defined
# solver.
from scipy.sparse.linalg import spsolve  # default

# from pypardiso import spsolve

system = fem.solve.partition(field, K, dof1, dof0, r)
dfield = fem.solve.solve(*system, ext0, solver=spsolve)  # .reshape(*u.shape)
# field += dfield

# %%
# A very simple Newton-Rhapson code looks like this:
for iteration in range(8):
    F = field.extract()

    linearform = fem.IntegralForm(P(F)[:-1], field, dV)
    bilinearform = fem.IntegralForm(A(F), field, dV, field)

    r = linearform.assemble()
    K = bilinearform.assemble()

    system = fem.solve.partition(field, K, dof1, dof0, r)
    dfield = fem.solve.solve(*system, ext0, solver=spsolve)

    norm = np.linalg.norm(dfield)
    print(iteration, norm)
    field += dfield

    if norm < 1e-12:
        break

# %%
# By alternative, one may also use the :func:`Newton-Rhapson <felupe.newtonrhapson>`
# function of FElupe.
field[0].fill(0)
solid = fem.SolidBody(umat, field)
loadcase = {"dof1": dof1, "dof0": dof0, "ext0": ext0}
res = fem.newtonrhapson(items=[solid], verbose=2, tol=1e-12, **loadcase)
field = res.x


# %%
# All 3x3 components of the deformation gradient of integration point 2 of cell 1
# (Python is 0-indexed) are obtained with
defgrad = field.evaluate.deformation_gradient()
print(defgrad[:, :, 1, 0])

# %%
# Export and plot of results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Results are exported as VTK or XDMF files using
# `meshio <https://pypi.org/project/meshio/>`_.
fem.save(region, field, filename="result.vtk")

# %%
# Any tensor at quadrature points, shifted or projected to and averaged at mesh-points,
# is evaluated for ``quad`` and ``hexahedron`` cell types by :func:`~felupe.topoints`
# or :func:`~felupe.project`, respectively. For example, the calculation of the Cauchy
# stress involves the conversion from the first Piola-Kirchhoff stress to the Cauchy
# stress followed by the shift or the projection. The stress results at mesh points are
# passed as a dictionary to the ``point_data`` argument.
from felupe.math import det, dot, tovoigt, transpose

s = dot(P(F)[0], transpose(F[0])) / det(F[0])
# s = solid.evaluate.cauchy_stress()

fem.save(
    region,
    field,
    filename="result_with_cauchy.vtk",
    point_data={
        "CauchyStressShifted": fem.topoints(s, region),
        "CauchyStressProjected": fem.project(tovoigt(s), region),
    },
)

# %%
# Instead of writing the simulation results to a VTK-file, point- and cell-data related
# to a solid body may also be viewed with PyVista. Once again, additional point-data
# like our stress results at mesh points are passed as a dictionary to the
# ``point_data`` argument of the :meth:`~felupe.SolidBody.view`-method. Cauchy stresses
# on solid bodies are also pre-defined in FElupe and may be plotted as cell- or
# (shifted / projected) point-data.
view = solid.view(
    point_data={
        "Cauchy Stress (Shifted)": fem.topoints(s, region),
        "Cauchy Stress (Projected)": fem.project(tovoigt(s), region),
    }
)

view.plot("Cauchy Stress (Shifted)", component=0).show()
solid.plot("Cauchy Stress", component=0).show()
# solid.plot("Cauchy Stress", component=0, project=fem.topoints).show()
# solid.plot("Cauchy Stress", component=0, project=fem.project).show()
