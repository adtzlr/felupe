# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from ._tensor import dot, eigh, eigvalsh, transpose
from ._tensor import tovoigt as tensor_to_vector


def displacement(field, dim=3, n=0):
    r"""Return the padded displacement values of the n-th field.

    Parameters
    ----------
    field : FieldContainer
        A field container containing displacement fields.
    dim : int, optional
        Target spatial dimension for padding with zeros along the second axis (default is 3).
    n : int, optional
        The index of the displacement field in the container (default is 0).

    Returns
    -------
    ndarray of shape (npoints, dim)
        The displacement values, padded with zeros if necessary.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Rectangle(n=2)
    >>> region = fem.RegionQuad(mesh)
    >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    >>> u = fem.math.displacement(field)
    >>> u.shape
    (4, 3)

    See Also
    --------
    felupe.math.values : Return concatenated values of fields.
    """

    u = field[n].values
    return np.pad(u, ((0, 0), (0, dim - u.shape[1])))


def deformation_gradient(field, n=0):
    r"""Return the deformation gradient tensors of the n-th field.

    Parameters
    ----------
    field : FieldContainer
        A field container with a displacement field.
    n : int, optional
        The index of the displacement field in the container (default is 0).

    Returns
    -------
    ndarray of shape (dim, dim, nquadraturepoints, ncells)
        The deformation gradient tensors at the quadrature points.

    Notes
    -----
    The deformation gradient tensor :math:`\boldsymbol{F}` is evaluated from the
    displacement gradient tensor :math:`\boldsymbol{H}` as:

    .. math::

       \boldsymbol{F} = \boldsymbol{1} + \boldsymbol{H} = \boldsymbol{1} + \nabla_{\boldsymbol{X}} \boldsymbol{u}

    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions (quadrature points and cells).

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> mesh = fem.Cube()
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> F = fem.math.deformation_gradient(field)
    >>> F.shape
    (3, 3, 8, 1)
    >>> np.allclose(F, fem.math.identity(dim=3, shape=(8, 1)))
    True

    See Also
    --------
    felupe.math.right_cauchy_green_deformation : Compute the right Cauchy-Green tensor.
    """
    return field[n].extract(grad=True, sym=False, add_identity=True)


def right_cauchy_green_deformation(field, n=0):
    r"""Return the right Cauchy-Green deformation tensor of the n-th field.

    Parameters
    ----------
    field : FieldContainer
        A field container with a displacement field.
    n : int, optional
        The index of the displacement field in the container (default is 0).

    Returns
    -------
    ndarray of shape (dim, dim, nquadraturepoints, ncells)
        The right Cauchy-Green deformation tensors at the quadrature points.

    Notes
    -----
    The right Cauchy-Green deformation tensor :math:`\boldsymbol{C}` is defined by:

    .. math::

       \boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}

    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions (quadrature points and cells).

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> mesh = fem.Cube()
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> C = fem.math.right_cauchy_green_deformation(field)
    >>> C.shape
    (3, 3, 8, 1)
    >>> np.allclose(C, fem.math.identity(dim=3, shape=(8, 1)))
    True

    See Also
    --------
    felupe.math.deformation_gradient : Return deformation gradient tensors.
    felupe.math.strain : Return Lagrangian strain tensor.
    """
    F = deformation_gradient(field, n=n)
    return dot(transpose(F), F)


def strain_stretch_1d(stretch, k=0):
    r"""Compute the Seth-Hill strains.

    Parameters
    ----------
    stretch : ndarray of shape (M, ...)
        The array of stretches.
    k : float, optional
        The strain-exponent of the Seth-Hill strain-stretch relation (default is 0).
        If 0, the logarithmic strains are computed.

    Returns
    -------
    ndarray of shape (M, ...)
        The computed strains.

    Notes
    -----
    .. math::

       E(\lambda, k) = \frac{1}{k} \left( \lambda^k - 1 \right)

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> stretch = np.array([1.0, 2.0])
    >>> fem.math.strain_stretch_1d(stretch, k=0)
    array([0.        , 0.69314718])

    >>> fem.math.strain_stretch_1d(stretch, k=1)
    array([0., 1.])

    >>> fem.math.strain_stretch_1d(stretch, k=2)
    array([0. , 1.5])

    See Also
    --------
    felupe.math.strain : Compute Lagrangian strain tensors.
    """

    if k == 0:
        strain = np.log(stretch)
    else:
        strain = (stretch**k - 1) / k

    return strain


def strain(
    field, C=None, fun=strain_stretch_1d, tensor=True, asvoigt=False, n=0, **kwargs
):
    r"""Return Lagrangian strain tensor or its principal values of the n-th field.

    Parameters
    ----------
    field : FieldContainer or None
        A field container with a displacement field from which the deformation gradient
        tensor is extracted if ``C`` is None.
    C : ndarray of shape (N, N, ...) or None, optional
        Optional array of right Cauchy-Green deformation tensors. If None, the
        right Cauchy-Green deformation tensor is obtained from the field. Default is
        None.
    fun : callable, optional
        A callable for the one-dimensional strain-stretch relation. Function signature
        must be ``lambda stretch, **kwargs: strain`` (default is the log. strain,
        :func:`~felupe.math.strain_stretch_1d` with ``k=0``).
    tensor : bool, optional
        Assemble and return the strain tensors if True or return their principal values
        only if False. Default is True.
    asvoigt : bool, optional
        Return the symmetric strain tensors in reduced vector (Voigt) storage. Default
        is False.
    n : int, optional
        The index of the displacement field (default is 0).
    **kwargs : dict, optional
        Optional keyword-arguments are passed to ``fun(stretch, **kwargs)``.

    Returns
    -------
    ndarray of shape (N, N, ...) tensor, (N!, ...) asvoigt or (N, ...) princ. values
        The strain tensors or their principal values.

    Notes
    -----
    The right Cauchy-Green deformation tensor is defined by the dot-products of the
    column vectors of the deformation gradient tensor and enables a quadratic length
    measure, see Eq. :eq:`right-cauchy-green-deformation-tensor`.

    .. math::
       :label: right-cauchy-green-deformation-tensor

       \boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}

    The principal stretches are obtained as the square-roots of the eigenvalues of the
    right Cauchy-Green deformation tensor, see Eq. :eq:`principal-stretches`.

    .. math::
       :label: principal-stretches

       \lambda^2_\alpha, \boldsymbol{N}_\alpha = \text{eig}\left( \boldsymbol{C} \right)

    The Lagrangian strain tensor is assembled with the one-dimensional strain-stretch
    relation from Eq. :eq:`principal-strain` and the eigenbases as dyadic products of
    the eigenvectors, see Eq. :eq:`strain-tensor`.

    .. math::
       :label: principal-strain

       E_\alpha = \text{f}\left( \lambda_\alpha \right)

    .. math::
       :label: strain-tensor

       \boldsymbol{E} = \sum_\alpha E_\alpha \
           \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube()
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> E = fem.math.strain(field)
    >>> E.shape
    (3, 3, 8, 1)

    Principal values can be computed by setting ``tensor=False``:

    >>> fem.math.strain(field, tensor=False).shape
    (3, 8, 1)

    See Also
    --------
    felupe.math.strain_stretch_1d : Compute the Seth-Hill strains.
    """

    if C is None:
        F = deformation_gradient(field, n=n)
        C = dot(transpose(F), F)

    if tensor:
        w, N = eigh(C)
        stretch = np.sqrt(w)
        tensor = np.einsum("a...,ia...,ja...->ij...", fun(stretch, **kwargs), N, N)
        if asvoigt:
            # double the off-diagonal items in Voigt-notation for strain tensors
            tensor = tensor_to_vector(tensor, strain=True)
        return tensor
    else:
        stretch = np.sqrt(eigvalsh(C))
        return fun(stretch, **kwargs)


def extract(field, grad=True, sym=False, add_identity=True):
    r"""Extract gradient or interpolated field values at quadrature points.

    Parameters
    ----------
    field : Field
        A field object.
    grad : bool, optional
        A flag to extract gradient values (default is True) or field values (False).
    sym : bool, optional
        A flag to return symmetric parts of second-order tensors (default is False).
    add_identity : bool, optional
        A flag to add identity matrices to second-order tensors (default is True).

    Returns
    -------
    ndarray
        Extracted field values or gradients at the quadrature points.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube()
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.Field(region, dim=3)
    >>> fem.math.extract(field, grad=True).shape
    (3, 3, 8, 1)
    >>> fem.math.extract(field, grad=False).shape
    (3, 8, 1)

    See Also
    --------
    felupe.math.interpolate : Interpolate field values at quadrature points.
    felupe.math.grad : Return the gradient of a field.
    """
    return field.extract(grad=grad, sym=sym, add_identity=add_identity)


def values(field):
    r"""Return flattened values of all fields in a field container.

    Parameters
    ----------
    field : FieldContainer
        A container containing one or more fields.

    Returns
    -------
    ndarray of shape (ndofs,)
        1D array of concatenated degrees of freedom values.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube()
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> u = fem.math.values(field)
    >>> u.shape
    (24,)

    See Also
    --------
    felupe.math.displacement : Return padded displacement values.
    """

    return np.concatenate([f.values.ravel() for f in field.fields])


def norm(array, axis=None):
    r"""Calculate the Euclidean norm of an array or a list of arrays.

    Parameters
    ----------
    array : ndarray or list of ndarray
        The input array or list of arrays.
    axis : int or tuple of int or None, optional
        Axis or axes along which to compute the norm (default is None).

    Returns
    -------
    float or ndarray
        Norm of the array, or array of norms if ``array`` is a list.

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> a = np.array([3.0, 4.0])
    >>> float(fem.math.norm(a))
    5.0

    Calculate the column-wise norms of a 2D array:

    >>> b = np.array([[3.0, 0.0], [4.0, 2.0]])
    >>> fem.math.norm(b, axis=0)
    array([5., 2.])

    Calculate the norms of a list of arrays:

    >>> fem.math.norm([np.array([3.0, 4.0]), np.array([1.0, 2.0, 2.0])])
    array([5., 3.])

    See Also
    --------
    numpy.linalg.norm : Matrix or vector norm.
    """
    if isinstance(array, list):
        return np.array([np.linalg.norm(arr, axis=axis) for arr in array])
    else:
        return np.linalg.norm(array, axis=axis)


def interpolate(field):
    r"""Interpolate field values at the quadrature points of its region.

    Parameters
    ----------
    field : Field
        The primary field object.

    Returns
    -------
    ndarray of shape (dim, nquadraturepoints, ncells)
        The interpolated field values.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube()
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.Field(region, dim=3)
    >>> u_quad = fem.math.interpolate(field)
    >>> u_quad.shape
    (3, 8, 1)

    See Also
    --------
    felupe.math.extract : Extract gradient or interpolated field values.
    """
    return field.interpolate()


def grad(x, **kwargs):
    r"""Return the gradient of a field or a basis-array.

    Parameters
    ----------
    x : Field or Basis
        An object providing a ``grad`` attribute or method.
    **kwargs : dict, optional
        Optional keyword arguments passed to ``x.grad(**kwargs)`` if callable.

    Returns
    -------
    ndarray
        The gradient array evaluated at the quadrature points.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube()
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.Field(region, dim=3)
    >>> g = fem.math.grad(field)
    >>> g.shape
    (3, 3, 8, 1)

    See Also
    --------
    felupe.math.hess : Return the hessian of a field or a basis-array.
    """
    if callable(x.grad):
        return x.grad(**kwargs)
    else:
        return x.grad


def hess(x, **kwargs):
    r"""Return the hessian of a field or a basis-array.

    Parameters
    ----------
    x : Field or Basis
        An object providing a ``hess`` attribute or method.
    **kwargs : dict, optional
        Optional keyword arguments passed to ``x.hess(**kwargs)`` if callable.

    Returns
    -------
    ndarray
        The hessian array evaluated at the quadrature points.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Rectangle()
    >>> region = fem.RegionQuad(mesh, hess=True)
    >>> field = fem.FieldPlaneStrain(region, dim=2)
    >>> h = fem.math.hess(field)
    >>> h.shape
    (3, 3, 3, 4, 1)

    See Also
    --------
    felupe.math.grad : Return the gradient of a field or a basis-array.
    """
    if callable(x.hess):
        return x.hess(**kwargs)
    else:
        return x.hess
