r"""
Best-fit Hyperelastic Material Parameters
-----------------------------------------

.. topic:: Best-fit of hyperelastic material parameters on given experimental data.

   * define a strain-energy function for a hyperelastic material formulation
   
   * best-fit the parameters to uniaxial and biaxial tension experiments
   
   * plot the standard deviations of the material parameters

The :func:`Extended Tube <felupe.extended_tube>` material model formulation [1]_ is
best-fitted on Treloar's uniaxial and biaxial tension data [2]_.
"""
import numpy as np
import tensortrax.math as tm

import felupe as fem

λ_ux, P_ux = np.array(
    [
        [1.000, 0.00],
        [1.020, 0.26],
        [1.125, 1.37],
        [1.240, 2.30],
        [1.390, 3.23],
        [1.585, 4.16],
        [1.900, 5.10],
        [2.180, 6.00],
        [2.420, 6.90],
        [3.020, 8.80],
        [3.570, 10.7],
        [4.030, 12.5],
        [4.760, 16.2],
        [5.360, 19.9],
        [5.750, 23.6],
        [6.150, 27.4],
        [6.400, 31.0],
        [6.600, 34.8],
        [6.850, 38.5],
        [7.050, 42.1],
        [7.150, 45.8],
        [7.250, 49.6],
        [7.400, 53.3],
        [7.500, 57.0],
        [7.600, 64.4],
    ]
).T * np.array([[1.0], [0.0980665]])

λ_bx, P_bx = np.array(
    [
        [1.00, 0.00],
        [1.03, 0.95],
        [1.07, 1.60],
        [1.12, 2.42],
        [1.14, 2.62],
        [1.20, 3.32],
        [1.31, 4.43],
        [1.42, 5.18],
        [1.68, 6.60],
        [1.94, 7.78],
        [2.49, 9.79],
        [3.03, 12.6],
        [3.43, 14.7],
        [3.75, 17.4],
        [4.07, 20.1],
        [4.26, 22.5],
        [4.45, 24.7],
    ]
).T * np.array([[1.0], [0.0980665]])


# %%
# An isotropic-hyperelastic material formulation is defined by a strain energy density
# function. All math-operations must support automatic differentiation and hence, they
# must be taken from :mod:`tensortrax.math <https://github.com/adtzlr/tensortrax>`. The
# strain energy function of the Extended Tube model is given in Eq. :eq:`ex-psi-et`
#
# ..  math::
#     :label: ex-psi-et
#
#     \psi = \frac{G_c}{2} \left[ \frac{\left( 1 - \delta^2 \right)
#         \left( \hat{I}_1 - 3 \right)}{1 - \delta^2 \left( \hat{I}_1 - 3 \right)} +
#         \ln \left( 1 - \delta^2 \left( \hat{I}_1 - 3 \right) \right) \right] +
#         \frac{2 G_e}{\beta^2} \left( \hat{\lambda}_1^{-\beta} +
#         \hat{\lambda}_2^{-\beta} + \hat{\lambda}_3^{-\beta} - 3 \right)
#
# with the first main invariant of the distortional part of the right
# Cauchy-Green deformation tensor as given in Eq. :eq:`ex-invariants-et`
#
# ..  math::
#     :label: ex-invariants-et
#
#     D = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)
#
# and the principal stretches, obtained from the distortional part of the right
# Cauchy-Green deformation tensor, see Eq. :eq:`ex-stretches-et`.
#
# ..  math::
#     :label: ex-stretches-et
#
#     \lambda^2_\alpha &= \text{eigvals}\left( \boldsymbol{C} \right)
#
#     \hat{\lambda}_\alpha &= J^{-1/3} \lambda_\alpha
#
# ..  note::
#     This material formulation is also available in FElupe as
#     :func:`~felupe.extended_tube` among other
#     :ref:`hyperelastic material models <felupe-api-constitution-hyperelasticity>`.
def extended_tube(C, Gc, δ, Ge, β):
    "Strain energy density function of the Extended Tube material formulation."

    I3 = tm.linalg.det(C)
    D = I3 ** (-1 / 3) * tm.trace(C)
    λ = tm.sqrt(I3 ** (-1 / 3) * tm.linalg.eigvalsh(C))
    γ = (1 - δ**2) * (D - 3) / (1 - δ**2 * (D - 3))

    Wc = Gc / 2 * (γ + tm.log(1 - δ**2 * (D - 3)))
    We = 2 * Ge / β**2 * tm.sum(λ**-β - 1)

    return Wc + We


umat = fem.Hyperelastic(extended_tube, Gc=0, δ=0.1, Ge=0, β=1)

# %%
# The material parameters are best-fitted to the experimental data. Relative force-
# residuals are used. Lower and upper material parameters are used to plot the errors
# of the force-stretch curves due to the errors of the material parameters.
umat_new, res = umat.optimize(
    ux=[λ_ux, P_ux], bx=[λ_bx, P_bx], incompressible=True, relative=True
)

low = umat_new.copy()
for (key, value), dx in zip(low.kwargs.items(), res.dx):
    low.kwargs[key] -= dx

high = umat_new.copy()
for (key, value), dx in zip(high.kwargs.items(), res.dx):
    high.kwargs[key] += dx

ux = np.linspace(λ_ux.min(), λ_ux.max(), num=50)
bx = np.linspace(λ_bx.min(), λ_bx.max(), num=50)

ax = umat_new.plot(incompressible=True, ux=ux, bx=bx, ps=None)
ax.plot(λ_ux, P_ux, "C0x")
ax.plot(λ_bx, P_bx, "C1x")
ax.fill_between(
    x=λ_ux,
    y1=low.view(incompressible=True, ux=λ_ux, ps=None, bx=None).evaluate()[0][1],
    y2=high.view(incompressible=True, ux=λ_ux, ps=None, bx=None).evaluate()[0][1],
    color="C0",
    alpha=0.2,
)
ax.fill_between(
    x=λ_bx,
    y1=low.view(incompressible=True, ux=None, bx=λ_bx, ps=None).evaluate()[0][1],
    y2=high.view(incompressible=True, ux=None, bx=λ_bx, ps=None).evaluate()[0][1],
    color="C1",
    alpha=0.2,
)

# %%
# References
# ----------
# .. [1] M. Kaliske and G. Heinrich, "An Extended Tube-Model for Rubber Elasticity:
#    Statistical-Mechanical Theory and Finite Element Implementation", Rubber
#    Chemistry and Technology, vol. 72, no. 4. Rubber Division, ACS, pp. 602–632,
#    Sep. 01, 1999. doi:
#    `10.5254/1.3538822 <https://www.doi.org/10.5254/1.3538822>`_.
#
# .. [2] L. R. G. Treloar, "Stress-strain data for vulcanised rubber under various
#    types of deformation", Transactions of the Faraday Society, vol. 40. Royal
#    Society of Chemistry (RSC), p. 59, 1944. doi:
#    `10.1039/tf9444000059 <https://doi.org/10.1039/tf9444000059>`_. Data
#    available at https://www.uni-due.de/mathematik/ag_neff/neff_hencky.
