.. _single-precision:

Run a Model in Single Precision
-------------------------------

FElupe evaluates a model in the data types of its regions and fields: the items assemble
their sparse vectors and matrices in the precision of the field values and the linear
equation system, its factorization and the solution keep this precision. With ``float32``
regions and fields, the memory of the assembled system and of its factorization is halved
compared to ``float64``, e.g. in WebAssembly, where the memory is limited.

Both the region and the fields have to be casted: :meth:`~felupe.Region.astype` casts the
shape-function gradients and the differential volumes of a region and a
:class:`~felupe.Field` takes the data type of its values. A ``float32`` field on a
``float64`` region is assembled in ``float64``, because the values of an item are of the
common data type of the field values and of the differential volumes ``dV``, which are
evaluated by the region. A casted region keeps its data type: if the arrays of the region
are re-evaluated by :meth:`~felupe.Region.reload`, e.g. by :meth:`~felupe.Region.copy` or
by :func:`~felupe.field.merge` for a model with several bodies, they are casted again.

..  code-block:: python

    import numpy as np
    import felupe as fem

    # the region with its shape-function gradients and differential volumes is
    # evaluated in double precision and casted once
    region = fem.RegionHexahedron(fem.Cube(n=8)).astype(np.float32)
    field = fem.FieldContainer([fem.Field(region, dim=3, dtype=np.float32)])

    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field)
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, return_loadcase=True)

    move = fem.math.linsteps([0, 0.2], num=4)
    step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
    job = fem.Job([step]).evaluate()

The precision of the assembled system is checked by the data types of the objective
function and of the Jacobian:

..  code-block:: python

    assert fem.tools.fun([solid], field).dtype == np.float32
    assert fem.tools.jac([solid], field).dtype == np.float32

..  note::

    The default relative tolerance of :func:`~felupe.newtonraphson` (``tol=None``) is the
    square root of the machine epsilon of the lowest precision of the unknowns and of the
    linear equation system, i.e. about ``3.5e-4`` for a single-precision model instead of
    about ``1.5e-8`` for double precision. The absolute tolerance (``atol=None``) bounds
    the round-off errors of the objective function.

Some results are evaluated in double precision on purpose:

..  list-table::
    :header-rows: 1

    * - Result
      - Reason
    * - the points of a mesh
      - they are the undeformed configuration and they are not casted together with a
        region, e.g. ``mesh.points`` is reused by other regions
    * - the eigenvalues and eigenvectors of a :class:`~felupe.FreeVibration`
      - :func:`scipy.sparse.linalg.eigsh` is evaluated in double precision
    * - the values of a user-defined material or form which returns a higher precision
      - the values of an item are of the common data type of the field values and of the
        values of the material

An item with a higher precision than the unknowns promotes the linear equation system and
its factorization, i.e. a single item which returns ``float64`` is enough. For the same
reason, the fields of a mixed-field formulation with different precisions, e.g. a
``float32`` displacement field and a ``float64`` pressure field, are assembled in the
common precision. Cast the region, the fields and, if required, the material parameters to
keep the whole system in single precision.
