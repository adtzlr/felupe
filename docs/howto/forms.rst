Linear and Bilinear Forms
~~~~~~~~~~~~~~~~~~~~~~~~~

FElupe requires a pre-evaluated array (a fourth-order tensor) for the definition of a bilinear :class:`felupe.IntegralForm` object on gradients. While this has two benefits, namely a fast integration of the form is easy to code and the array may be computed in any programming language, sometimes numeric representations of analytic linear and bilinear form expressions may be easier in user-code and less error prone compared to the calculation of explicit fourth order tensors. Therefore, FElupe provides :class:`felupe.LinearForm` and :class:`felupe.BilinearForm` for single-field as well as :class:`felupe.LinearFormMixed` and :class:`felupe.BilinearFormMixed` for mixed-field problems. These linear and bilinear form classes are nearly identical in their usage compared to :class:`felupe.IntegralForm` as they require a callable function (with optional arguments and keyword arguments) instead of a pre-computed array to be passed. The bilinear form of linear elasticity serves as a reference example for the demonstration how to use this feature of FElupe. The stiffness matrix is assembled for a unit cube out of hexahedrons.

..  code-block:: python

    import felupe as fe
    
    mesh = fe.Cube(n=11)
    region = fe.RegionHexahedron(mesh)
    displacement = fe.Field(region, dim=3)
    basis = fe.Basis(displacement)

The bilinear form of linear elasticity is defined as

..  math::
    
    a(v, u) = \int_\Omega 2 \mu \ \delta\boldsymbol{\varepsilon} : \boldsymbol{\varepsilon} + \lambda \ \text{tr}(\delta\boldsymbol{\varepsilon}) \ \text{tr}(\boldsymbol{\varepsilon}) \ dV

with

..  math::

    \delta\boldsymbol{\varepsilon} &= \text{sym}(\text{grad}(\boldsymbol{v}))
    
    \boldsymbol{\varepsilon} &= \text{sym}(\text{grad}(\boldsymbol{u})) 
    
and implemented in FElupe very close to the analytic expression. The first two arguments for the callable *weak-form* function of a bilinear form are always basis objects of field (gradients) ``(v, u)`` followed by optional arguments and keyword arguments. Optionally, the integration/assembly may be performed in parallel (threaded). Please note that this is only faster for relatively large systems. Contrary to :class:`felupe.IntegralForm`, :class:`felupe.BilinearForm` does not utilize Numba for parallel integration/assembly but uses Python's threading instead.

..  code-block:: python

    from felupe.math import ddot, trace, sym
    
    def linear_elastic(grad_v, grad_u, mu, lmbda):
        "Linear elasticity."
        
        de, e = sym(grad_v), sym(grad_u)
        return 2 * mu * ddot(de, e) + lmbda * trace(de) * trace(e)
    
    a = fe.BilinearForm(v=basis, u=basis, grad_v=True, grad_u=True)
    K = a.assemble(
        linear_elastic, 
        kwargs={"mu": 1.0, "lmbda": 2.0}, 
        parallel=False
    )

A :class:`felupe.LinearForm` is initiated identical to :class:`felupe.BilinearForm` but without the argument ``u``. Mixed forms (:class:`felupe.LinearFormMixed` and :class:`felupe.BilinearFormMixed`) have to be used with :class:`felupe.FieldMixed` and :class:`felupe.BasisMixed`.