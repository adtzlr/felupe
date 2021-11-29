Axisymmetric Analysis
---------------------

Axisymmetric scenarios are modeled with a 2D-mesh and consequently, a 2D element formulation. The rotation axis is chosen along the global X-axis :math:`(X,Y,Z) \widehat{=} (Z,R,\varphi)`. The 3x3 deformation gradient consists of an in-plane 2x2 sub-matrix and one additional entry for the out-of-plane stretch which is equal to the ratio of deformed and undeformed radius.

..  math::
    
    \boldsymbol{F} = \begin{bmatrix} \boldsymbol{F}_{(2D)} & \boldsymbol{0} \\ \boldsymbol{0}^T & \frac{r}{R} \end{bmatrix}

The variation of the deformation gradient consists of both in- and out-of-plane contributions.

..  math::
    
    \delta \boldsymbol{F}_{(2D)} = \delta \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} \qquad \text{and} \qquad \delta \left(\frac{r}{R}\right) = \frac{\delta u_r}{R}

Again, the internal virtual work leads to two seperate terms.

..  math::
    
    -\delta W_{int} = \int_V \boldsymbol{P} : \delta \boldsymbol{F} \ dV = \int_V \boldsymbol{P}_{(2D)} : \delta \boldsymbol{F}_{(2D)} \ dV + \int_V \frac{P_{33}}{R} : \delta u_r \ dV

The differential volume is further expressed as a product of the differential in-plane area and the differential arc length. The arc length integral is finally pre-evaluated.

..  math::

    \int_V dV = \int_{\varphi=0}^{2\pi} \int_A R\ dA\ d\varphi = 2\pi \int_A R\ dA

Inserting the differential volume integral into the expression of internal virtual work, this leads to:

..  math::
    
    -\delta W_{int} = 2\pi \int_A \boldsymbol{P}_{(2D)} : \delta \boldsymbol{F}_{(2D)} \ R \ dA + 2\pi \int_A P_{33} : \delta u_r \ dA

A Linearization of the internal virtual work expression gives four terms.

..  math::
    
    -\Delta \delta W_{int} &= \Delta_{(2D)} \delta_{(2D)} W_{int} + \Delta_{33} \delta_{(2D)} W_{int} + \Delta_{(2D)} \delta_{33} W_{int} + \Delta_{33} \delta_{33} W_{int}

    -\Delta_{(2D)} \delta_{(2D)} W_{int} &= 2\pi \int_A \delta \boldsymbol{F}_{(2D)} : \mathbb{A}_{(2D),(2D)} : \Delta \boldsymbol{F}_{(2D)} \ R \ dA

    -\Delta_{33} \delta_{(2D)} W_{int} &= 2\pi \int_A \delta \boldsymbol{F}_{(2D)} : \mathbb{A}_{(2D),33} : \Delta u_r \ dA

    -\Delta_{(2D)} \delta_{33} W_{int} &= 2\pi \int_A \delta u_r : \mathbb{A}_{33,(2D)} : \Delta \boldsymbol{F}_{(2D)} \ dA

    -\Delta_{33} \delta_{33} W_{int} &= 2\pi \int_A \delta u_r : \frac{\mathbb{A}_{33,33}}{R} : \Delta u_r \ dA

with 

..  math::
    
    \mathbb{A}_{(2D),(2D)} &= \frac{\partial \psi}{\partial \boldsymbol{F}_{(2D)} \partial \boldsymbol{F}_{(2D)}}

    \mathbb{A}_{(2D),33} &= \frac{\partial \psi}{\partial \boldsymbol{F}_{(2D)} \partial F^3_{\hphantom{3}3}} \left ( = \mathbb{A}_{33,(2D)} \right )

    \mathbb{A}_{33,33} &= \frac{\partial \psi}{F^3_{\hphantom{3}3} \partial F^3_{\hphantom{3}3}}