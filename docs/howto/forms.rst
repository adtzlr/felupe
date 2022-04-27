Linear and Bilinear Forms
~~~~~~~~~~~~~~~~~~~~~~~~~

FElupe requires a pre-evaluated array for the definition of a bilinear :class:`felupe.IntegralForm` object on interpolated field values or their gradients. While this has two benefits, namely a fast integration of the form is easy to code and the array may be computed in any programming language, sometimes numeric representations of analytic linear and bilinear form expressions may be easier in user-code and less error prone compared to the calculation of explicit second or fourth-order tensors. Therefore, FElupe provides a function decorator :func:`felupe.Form` as an easy-to-use high-level interface, similar to what `scikit-fem <https://github.com/kinnala/scikit-fem>`_ offers. While the :func:`felupe.Form` decorator handles both single and mixed fields, additional access to the underlying form objects is enabled by :class:`felupe.LinearForm` and :class:`felupe.BilinearForm` for single-field as well as :class:`felupe.LinearFormMixed` and :class:`felupe.BilinearFormMixed` for mixed-field problems. All these linear and bilinear form classes are similar, but not identical in their usage compared to :class:`felupe.IntegralForm`. They require a callable function (with optional arguments and keyword arguments) instead of a pre-computed array to be passed. The bilinear form of linear elasticity serves as a reference example for the demonstration on how to use this feature of FElupe. The stiffness matrix is assembled for a unit cube out of hexahedrons.

High-level code using the :func:`felupe.Form` decorator
-------------------------------------------------------

..  code-block:: python

    import felupe as fe
    
    mesh = fe.Cube(n=11)
    region = fe.RegionHexahedron(mesh)
    displacement = fe.Field(region, dim=3)

The bilinear form of linear elasticity is defined as

..  math::
    
    a(v, u) = \int_\Omega 2 \mu \ \delta\boldsymbol{\varepsilon} : \boldsymbol{\varepsilon} + \lambda \ \text{tr}(\delta\boldsymbol{\varepsilon}) \ \text{tr}(\boldsymbol{\varepsilon}) \ dV

with

..  math::

    \delta\boldsymbol{\varepsilon} &= \text{sym}(\text{grad}(\boldsymbol{v}))
    
    \boldsymbol{\varepsilon} &= \text{sym}(\text{grad}(\boldsymbol{u})) 
    
and implemented in FElupe closely to the analytic expression. The first two arguments for the callable *weak-form* function of a bilinear form are always arrays of field (gradients) ``(v, u)`` followed by arguments and keyword arguments. Optionally, the integration/assembly may be performed in parallel (threaded). Please note that this is only faster for relatively large systems. Contrary to :class:`felupe.IntegralForm`, :func:`felupe.Form` does not offer a Just-In-Time (JIT) compilation by Numba for integration/assembly. The weak-form function is decorated by :func:`felupe.Form` where the appropriate fields are linked to ``v`` and ``u`` along with the gradient flags for both fields. Arguments as well as keyword arguments of the weak-form may be defined inside the decorator or as part of the assembly arguments.

..  code-block:: python

    from felupe.math import ddot, trace, sym
    
    @fe.Form(v=displacement, u=displacement, grad_v=True, grad_u=True, kwargs={"mu": 1.0, "lmbda": 2.0})
    def linear_elasticity(gradv, gradu, mu, lmbda):
        "Linear elasticity."
        
        de, e = sym(gradv), sym(gradu)
        return 2 * mu * ddot(de, e) + lmbda * trace(de) * trace(e)

    K = linear_elasticity.assemble(v=displacement, u=displacement, parallel=False)


(Legacy) low-level code
-----------------------

..  code-block:: python

    import felupe as fe
    
    mesh = fe.Cube(n=11)
    region = fe.RegionHexahedron(mesh)
    displacement = fe.Field(region, dim=3)
    basis = fe.Basis(displacement)

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
