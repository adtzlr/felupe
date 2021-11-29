Mixed-field formulations
------------------------

FElupe supports mixed-field formulations in a similar way it can handle (default) single-field variations. The definition of a mixed-field variation is shown for the (Hu-Washizu) hydrostatic-volumetric selective three-field-variation with independend fields for displacements :math:`\boldsymbol{u}`, pressure :math:`p` and volume ratio :math:`J`. The total potential energy for nearly-incompressible hyperelasticity is formulated with a determinant-modified deformation gradient. Pressure and volume ratio fields should be kept one order lower than the interpolation order of the displacement field, i.e. linear displacement fields should be paired with cellwise-constant (mean) values of pressure and volume ratio.

Total potential energy: variation and linearization
***************************************************

The total potential energy of internal forces is defined with a strain energy density function in terms of a determinant-modified deformation gradient and an additional control equation.

..  math::

    \Pi &= \Pi_{int} + \Pi_{ext}

    \Pi_{int} &= \int_V \psi(\boldsymbol{F}) \ dV \qquad \rightarrow \qquad \Pi_{int}(\boldsymbol{u},p,J) = \int_V \psi(\overline{\boldsymbol{F}}) \ dV + \int_V p (J-\overline{J}) \ dV

    \overline{\boldsymbol{F}} &= \left(\frac{\overline{J}}{J}\right)^{1/3} \boldsymbol{F}

The variations of the total potential energy w.r.t. :math:`(\boldsymbol{u},p,J)` lead to the following expressions. We denote first partial derivatives as :math:`\boldsymbol{f}_{(\bullet)}` and second partial derivatives as :math:`\boldsymbol{A}_{(\bullet,\bullet)}`.
    
..  math::

    \delta_{\boldsymbol{u}} \Pi_{int} &= \int_V \boldsymbol{f}_{\boldsymbol{u}} : \delta \boldsymbol{F} \ dV = \int_V \left( \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} : \frac{\partial \overline{\boldsymbol{F}}}{\partial \boldsymbol{F}} + p J \boldsymbol{F}^{-T} \right) : \delta \boldsymbol{F} \ dV

    \delta_{p} \Pi_{int} &= \int_V f_{p} \ \delta p \ dV = \int_V (J - \overline{J}) \ \delta p \ dV

    \delta_{\overline{J}} \Pi_{int} &= \int_V f_{\overline{J}} \ \delta \overline{J} \ dV = \int_V \left( \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} : \frac{\partial \overline{\boldsymbol{F}}}{\partial \overline{J}} - p \right) : \delta \overline{J} \ dV

The projection tensors from the variations lead the following results.

..  math::

    \frac{\partial \overline{\boldsymbol{F}}}{\partial \boldsymbol{F}} &= \left(\frac{\overline{J}}{J}\right)^{1/3} \left( \boldsymbol{I} \overset{ik}{\odot} \boldsymbol{I} - \frac{1}{3} \boldsymbol{F} \otimes \boldsymbol{F}^{-T} \right)

    \frac{\partial \overline{\boldsymbol{F}}}{\partial \overline{J}} &= \frac{1}{3 \overline{J}} \overline{\boldsymbol{F}}

The double-dot products from the variations are now evaluated.

..  math::

    \overline{\boldsymbol{P}} &= \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} = \overline{\overline{\boldsymbol{P}}} - \frac{1}{3} \left(  \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \qquad \text{with} \qquad \overline{\overline{\boldsymbol{P}}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}}

    \frac{\partial \psi}{\partial \overline{\boldsymbol{F}}} : \frac{1}{3 \overline{J}} \overline{\boldsymbol{F}} &= \frac{1}{3 \overline{J}} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F}

We now have three formulas; one for the first Piola Kirchhoff stress and two additional control equations.

..  math::

    \boldsymbol{f}_{\boldsymbol{u}} (= \boldsymbol{P}) &= \overline{\overline{\boldsymbol{P}}} - \frac{1}{3} \left(  \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} 

    f_p &= J - \overline{J} 

    f_{\overline{J}} &=  \frac{1}{3 \overline{J}} \left( \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) - p

A linearization of the above formulas gives six equations (only results are given here).

..  math::

    \mathbb{A}_{\boldsymbol{u},\boldsymbol{u}} &=  \overline{\overline{\mathbb{A}}} + \frac{1}{9} \left(  \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \frac{1}{3} \left( \boldsymbol{F}^{-T} \otimes \left( \overline{\overline{\boldsymbol{P}}} + \boldsymbol{F} : \overline{\overline{\mathbb{A}}} \right) + \left( \overline{\overline{\boldsymbol{P}}} + \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) \otimes \boldsymbol{F}^{-T} \right)

    &+\left( p J + \frac{1}{9} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - \left( p J - \frac{1}{3} \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \overset{il}{\odot} \boldsymbol{F}^{-T} 

    A_{p,p} &= 0 

    A_{\overline{J},\overline{J}} &= \frac{1}{9 \overline{J}^2} \left( \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) - 2 \left( \overline{\overline{\boldsymbol{P}}} : \boldsymbol{F} \right)

    \boldsymbol{A}_{\boldsymbol{u},p} &= \boldsymbol{A}_{p, \boldsymbol{u}} = J \boldsymbol{F}^{-T}

    \boldsymbol{A}_{\boldsymbol{u},\overline{J}} &= \boldsymbol{A}_{\overline{J}, \boldsymbol{u}} = \frac{1}{3 \overline{J}} \left( \boldsymbol{P}' + \boldsymbol{F} : \overline{\overline{\mathbb{A}}} - \frac{1}{3} \left( \boldsymbol{F} : \overline{\overline{\mathbb{A}}} : \boldsymbol{F} \right) \boldsymbol{F}^{-T} \right)

    A_{p,\overline{J}} &= A_{\overline{J}, p} = -1

with 

..  math::

    \overline{\overline{\mathbb{A}}} = \left(\frac{\overline{J}}{J}\right)^{1/3} \frac{\partial^2 \psi}{\partial \overline{\boldsymbol{F}} \partial \overline{\boldsymbol{F}}} \left(\frac{\overline{J}}{J}\right)^{1/3}

as well as

..  math::

    \boldsymbol{P}' = \boldsymbol{P} - p J \boldsymbol{F}^{-T}

