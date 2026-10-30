# SPDX-License-Identifier: GPL-3.0-or-later

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensortrax.math as tm

import felupe as fem
from felupe.constitution._material import takes_time_increment


def kinematics(seed=7, npoints=4, ncells=5):
    "Return a deformation gradient and the initial state of a viscoelastic material."

    rng = np.random.default_rng(seed)
    F = np.eye(3).reshape(3, 3, 1, 1) + rng.uniform(-0.1, 0.1, (3, 3, npoints, ncells))

    # the old state variables, the inelastic right Cauchy-Green deformation tensor is the
    # identity (all-zero deviation from the identity)
    Ci = np.zeros((6, npoints, ncells))

    return F, Ci


def viscoelastic_lagrange(F, statevars, mu, eta, dtime):
    "First Piola-Kirchhoff stress of a finite strain viscoelastic material formulation."
    C = tm.dot(tm.transpose(F), F)
    Cu = tm.linalg.det(C) ** (-1 / 3) * C
    eye = tm.base.eye(C)
    Cin = eye + tm.special.from_triu_1d(statevars, like=C)
    Ci = Cin + mu / eta * dtime * Cu
    Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci
    S = mu * tm.linalg.inv(C) @ tm.special.dev(Cu @ tm.linalg.inv(Ci))
    return F @ S, tm.special.triu_1d(Ci - eye)


def assert_equal_results(res, expected):
    "Assert equal lists of results (None or arrays)."
    assert len(res) == len(expected)
    for r, e in zip(res, expected):
        if e is None:
            assert r is None
        else:
            assert np.allclose(r, e, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    "Material, fun",
    [
        (fem.Hyperelastic, fem.constitution.finite_strain_viscoelastic),
        (fem.MaterialAD, viscoelastic_lagrange),
    ],
)
def test_time_increment_input(Material, fun):
    "The time increment as input equals the time increment as material parameter."

    F, Ci = kinematics()
    umat = Material(fun, mu=1.0, eta=2.0, nstatevars=6)

    assert umat.time_dependent

    for dtime in [0.0, 0.1, 2.0]:
        old = Material(fun, mu=1.0, eta=2.0, dtime=dtime, nstatevars=6)
        assert not old.time_dependent

        assert_equal_results(umat.gradient([F, Ci], dtime=dtime), old.gradient([F, Ci]))
        assert_equal_results(umat.hessian([F, Ci], dtime=dtime), old.hessian([F, Ci]))

    # the time increment is effective
    P1, Ci1 = umat.gradient([F, Ci], dtime=0.1)
    P2, Ci2 = umat.gradient([F, Ci], dtime=2.0)
    assert not np.allclose(Ci1, Ci2)
    assert not np.allclose(P1, P2)

    # the old state variables are not modified
    assert np.allclose(Ci, kinematics()[1])


def test_time_increment_chunks(monkeypatch):
    "The time increment is passed to all chunks of cells (also with threads)."

    import felupe.math._threads as threads

    monkeypatch.setattr(threads, "cpu_count", lambda: 2)

    F, Ci = kinematics(ncells=9)
    q = F.shape[-2]

    for Material, fun in [
        (fem.Hyperelastic, fem.constitution.finite_strain_viscoelastic),
        (fem.MaterialAD, viscoelastic_lagrange),
    ]:
        whole = Material(fun, mu=1.0, eta=2.0, dtime=0.3, nstatevars=6)
        whole.chunksize = None

        for parallel in [False, True]:
            umat = Material(fun, mu=1.0, eta=2.0, nstatevars=6, parallel=parallel)
            umat.chunksize = 2 * q  # chunks of two cells

            for method in ["gradient", "hessian"]:
                expected = getattr(whole, method)([F, Ci])
                res = getattr(umat, method)([F, Ci], dtime=0.3)
                assert_equal_results(res, expected)

    # the output array of the first gradient is supported
    umat = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, mu=1.0, eta=2.0, nstatevars=6
    )
    umat.chunksize = 2 * q
    out = np.zeros_like(F)
    P, statevars_new = umat.gradient([F, Ci], out=out, dtime=0.3)
    assert P is out

    old = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic,
        mu=1.0,
        eta=2.0,
        dtime=0.3,
        nstatevars=6,
    )
    assert np.allclose(out, old.gradient([F, Ci])[0])


def test_time_increment_material():
    "A user-defined material with a time increment argument is time-dependent."

    F, _ = kinematics()

    def stress(x, eta, dtime):
        F, statevars = x[0], x[-1]
        return [eta / (1 + dtime) * F, statevars]

    def elasticity(x, eta, dtime):
        F = x[0]
        eye = fem.math.identity(F)
        return [eta / (1 + dtime) * fem.math.cdya_ik(eye, eye)]

    umat = fem.Material(stress, elasticity, eta=2.0)
    assert umat.time_dependent

    P, statevars = umat.gradient([F, None], dtime=1.0)
    A = umat.hessian([F, None], dtime=1.0)[0]
    assert np.allclose(P, F)
    assert np.allclose(A[0, 0, 0, 0], 1.0)

    old = fem.Material(stress, elasticity, eta=2.0, dtime=1.0)
    assert not old.time_dependent
    assert np.allclose(old.gradient([F, None])[0], P)
    assert np.allclose(old.hessian([F, None])[0], A)


def test_time_increment_detection():
    "The time-dependency is detected by the signature of the material function."

    # a time increment with a default value is a material parameter (as before)
    def viscoelastic_default(C, Cin, mu, eta, dtime=1.0):
        return fem.constitution.finite_strain_viscoelastic(C, Cin, mu, eta, dtime)

    umat = fem.Hyperelastic(viscoelastic_default, mu=1.0, eta=1.0, nstatevars=6)
    assert not umat.time_dependent

    # the signature of a decorated function is taken from the original function
    umat = fem.Hyperelastic(
        fem.isochoric_volumetric_split(fem.constitution.finite_strain_viscoelastic),
        mu=1.0,
        eta=1.0,
        nstatevars=6,
    )
    assert umat.time_dependent

    @fem.total_lagrange
    def viscous_total_lagrange(F, statevars, eta, dtime):
        C = F.T @ F
        return eta * dtime * tm.special.dev(C) @ tm.linalg.inv(C), statevars

    umat = fem.MaterialAD(viscous_total_lagrange, eta=1.0, nstatevars=1)
    assert umat.time_dependent

    # a time increment given by the default material parameters of a model
    def visco(C, Cin, mu, eta, dtime):
        return fem.constitution.finite_strain_viscoelastic(C, Cin, mu, eta, dtime)

    visco.kwargs = dict(mu=0.0, eta=1.0, dtime=1.0)
    assert not fem.Hyperelastic(visco, nstatevars=6).time_dependent

    # materials without a time increment
    assert not fem.Hyperelastic(fem.neo_hooke, mu=1.0).time_dependent
    assert not fem.MaterialAD(
        viscoelastic_lagrange, mu=1, eta=1, dtime=1
    ).time_dependent
    assert not fem.NeoHooke(mu=1.0).time_dependent
    assert not fem.ConstitutiveMaterial.time_dependent

    # a function without a signature (e.g. some built-in functions)
    assert not takes_time_increment(max, {})

    # a function with a signature without a time increment
    assert not takes_time_increment(np.add, {})


def test_time_increment_errors():
    "A time-dependent material requires a time increment, other materials don't."

    F, Ci = kinematics()
    umat = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, mu=1.0, eta=1.0, nstatevars=6
    )

    for method in [umat.gradient, umat.hessian]:
        with pytest.raises(ValueError, match="time-dependent"):
            method([F, Ci])

        with pytest.raises(ValueError, match="time-dependent"):
            method([F, Ci], dtime=None)

    # a constant time increment as material parameter
    old = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic,
        mu=1.0,
        eta=1.0,
        dtime=1.0,
        nstatevars=6,
    )

    for method in [old.gradient, old.hessian]:
        with pytest.raises(TypeError, match="material parameter `dtime` is constant"):
            method([F, Ci], dtime=0.1)

    # materials which are not time-dependent
    def stress(x, mu):
        return [mu * x[0], x[-1]]

    def elasticity(x, mu):
        return [mu * fem.math.cdya_ik(fem.math.identity(x[0]), fem.math.identity(x[0]))]

    for umat in [
        fem.NeoHooke(mu=1.0, bulk=2.0),
        fem.Hyperelastic(fem.neo_hooke, mu=1.0),
        fem.Material(stress, elasticity, mu=1.0),
    ]:
        for method in [umat.gradient, umat.hessian]:
            with pytest.raises(TypeError, match="not time-dependent"):
                method([F, None], dtime=0.1)


def test_time_increment_custom():
    "A custom constitutive material takes the time increment in chunks of cells."

    class Viscous(fem.ConstitutiveMaterial):
        time_dependent = True

        def __init__(self, eta):
            self.eta = eta
            self.x = [np.eye(3), np.zeros(0)]
            self.chunksize = 8  # chunks of two cells

        def _gradient(self, x, dtime):
            F, statevars = x[0], x[-1]
            return [self.eta / dtime * (F - fem.math.identity(F)), statevars]

        def _hessian(self, x, dtime):
            eye = fem.math.identity(x[0])
            return [self.eta / dtime * fem.math.cdya_ik(eye, eye)]

    F, _ = kinematics()
    umat = Viscous(eta=2.0)

    P, statevars = umat.gradient([F, None], dtime=4.0)
    A = umat.hessian([F, None], dtime=4.0)[0]

    assert np.allclose(P, (F - np.eye(3).reshape(3, 3, 1, 1)) / 2)
    assert np.allclose(A[0, 0, 0, 0], 0.5)

    with pytest.raises(ValueError, match="time-dependent"):
        umat.gradient([F, None])


def wrapper_inputs():
    "Return the input lists of the mixed-field formulations and of a single field."

    rng = np.random.default_rng(3)
    region = fem.RegionHexahedron(fem.Cube(n=3))
    q, c = region.dV.shape

    # the old state variables, the inelastic right Cauchy-Green deformation tensor is the
    # identity (all-zero deviation from the identity)
    Ci = np.zeros((6, q, c))

    mixed = fem.FieldsMixed(region, n=3)
    mixed[0].values[:] = rng.uniform(-0.05, 0.05, mixed[0].values.shape)
    mixed[1].values[:] = rng.uniform(-0.1, 0.1, mixed[1].values.shape)
    mixed[2].values[:] = 1 + rng.uniform(-0.05, 0.05, mixed[2].values.shape)

    third = fem.FieldContainer(
        [fem.Field(region, dim=3), fem.Field(region, dim=9)], take=[0, 1, 1]
    )
    third[0].values[:] = mixed[0].values
    third[1].values[:] = rng.uniform(-0.05, 0.05, third[1].values.shape)

    return {
        "mixed": [*mixed.extract(), Ci],
        "third": [*third.extract(grad=[True, False, True]), Ci],
        "single": [mixed.extract()[0], Ci],
    }


def test_time_increment_wrappers():
    "The wrapper materials pass the time increment to the time-dependent materials."

    inputs = wrapper_inputs()
    visco = fem.constitution.finite_strain_viscoelastic

    wrappers = [
        (lambda m: fem.NearlyIncompressible(m, bulk=50.0), "mixed"),
        (fem.ThreeFieldVariation, "mixed"),
        (lambda m: fem.ThirdMediumContactMixed(m, 0.1, 1e-4, 1e-2), "third"),
        (lambda m: m & fem.Volumetric(bulk=50.0), "single"),
        (lambda m: fem.NeoHooke(mu=0.5) & m, "single"),
    ]

    for Wrapper, kind in wrappers:
        x = inputs[kind]
        umat = Wrapper(fem.Hyperelastic(visco, mu=1.0, eta=2.0, nstatevars=6))
        assert umat.time_dependent

        for dtime in [0.0, 0.3]:
            old = Wrapper(
                fem.Hyperelastic(visco, mu=1.0, eta=2.0, dtime=dtime, nstatevars=6)
            )
            assert not old.time_dependent

            assert_equal_results(umat.gradient(x, dtime=dtime), old.gradient(x))
            assert_equal_results(umat.hessian(x, dtime=dtime), old.hessian(x))

        # the time increment is effective
        assert not np.allclose(
            umat.gradient(x, dtime=0.1)[0], umat.gradient(x, dtime=2.0)[0]
        )

        for method in [umat.gradient, umat.hessian]:
            with pytest.raises(ValueError, match="time-dependent"):
                method(x)

            with pytest.raises(TypeError, match="not time-dependent"):
                Wrapper(fem.NeoHooke(mu=1.0)).gradient(x, dtime=0.1)

    # materials which are not time-dependent
    for umat in [
        fem.NearlyIncompressible(fem.NeoHooke(mu=1.0), bulk=50.0),
        fem.ThreeFieldVariation(fem.NeoHooke(mu=1.0)),
        fem.NeoHooke(mu=1.0) & fem.Volumetric(bulk=50.0),
    ]:
        assert not umat.time_dependent


def test_time_increment_external():
    "A material of an external package takes the time increment by duck typing."

    class External:
        "A time-dependent material which is not derived from ConstitutiveMaterial."

        time_dependent = True

        def __init__(self):
            self.x = [np.eye(3), np.zeros(0)]
            self.kwargs = {}
            self.dtimes = []

        def gradient(self, x, dtime):
            self.dtimes.append(dtime)
            F, statevars = x[0], x[-1]
            return [F / (1 + dtime), statevars]

        def hessian(self, x, dtime):
            self.dtimes.append(dtime)
            eye = fem.math.identity(x[0])
            ones = np.ones(x[0].shape[-2:])  # the full shape of the trailing axes
            return [fem.math.cdya_ik(eye, eye) / (1 + dtime) * ones]

    inputs = wrapper_inputs()
    x = inputs["mixed"]
    F = x[0]

    for Wrapper in [
        lambda m: fem.NearlyIncompressible(m, bulk=50.0),
        fem.ThreeFieldVariation,
        lambda m: fem.Volumetric(bulk=50.0) & m,
    ]:
        external = External()
        umat = Wrapper(external)
        assert umat.time_dependent

        if isinstance(umat, fem.constitution.CompositeMaterial):
            x = [F, None]

        umat.gradient(x, dtime=0.5)
        umat.hessian(x, dtime=0.5)

        assert len(external.dtimes) > 0
        assert set(external.dtimes) == {0.5}

    # without the attribute, the material is not time-dependent
    del External.time_dependent
    umat = fem.NearlyIncompressible(External(), bulk=50.0)
    assert not umat.time_dependent

    with pytest.raises(TypeError, match="not time-dependent"):
        umat.gradient(inputs["mixed"], dtime=0.5)


def jax_viscoelastic_lagrange(F, statevars, mu, eta, dtime):
    "Finite strain viscoelastic material formulation (template of jax.Material)."

    # unimodular part of the right Cauchy-Green deformation tensor
    C = F.T @ F
    Cu = jnp.linalg.det(C) ** (-1 / 3) * C

    # update of state variables by evolution equation
    from_triu = lambda C: C[jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])]
    Cin = jnp.eye(3) + from_triu(statevars)
    Ci = Cin + mu / eta * dtime * Cu
    Ci = jnp.linalg.det(Ci) ** (-1 / 3) * Ci

    # second Piola-Kirchhoff stress tensor
    dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)
    S = mu * jnp.linalg.inv(C) @ dev(Cu @ jnp.linalg.inv(Ci))

    # first Piola-Kirchhoff stress tensor and new state variables
    i, j = jnp.triu_indices(3)
    to_triu = lambda C: C[i, j]
    return F @ S, to_triu(Ci - jnp.eye(3))


def jax_viscoelastic(C, statevars, mu, eta, dtime):
    "Finite strain viscoelastic material formulation (template of jax.Hyperelastic)."

    # unimodular part of the right Cauchy-Green deformation tensor
    Cu = jnp.linalg.det(C) ** (-1 / 3) * C

    # update of state variables by evolution equation
    Cin = jnp.eye(3) + statevars.reshape(3, 3)
    Ci = Cin + mu / eta * dtime * Cu
    Ci = jnp.linalg.det(Ci) ** (-1 / 3) * Ci
    invCi = jnp.linalg.inv(Ci)

    # strain energy function
    ψ = mu / 2 * (jnp.trace(Cu @ invCi) - 3)

    # incremental dissipation potential (no evolution for a zero time increment)
    Δt = jnp.where(dtime > 0, dtime, 1.0)
    φ = eta / (2 * Δt) * (jnp.trace(Cin @ invCi) - 3)
    φ = jnp.where(dtime > 0, φ, 0.0)

    # incremental potential and new state variables
    return ψ + φ, (Ci - jnp.eye(3)).ravel()


def to_full(statevars):
    "Return all nine components of the six upper-triangle components."
    return statevars[[0, 1, 2, 1, 3, 4, 2, 4, 5]]


def to_triu(statevars):
    "Return the six upper-triangle components of all nine components."
    return statevars[[0, 1, 2, 4, 5, 8]]


@pytest.fixture
def x64():
    "Enable the double precision of JAX within a test."
    enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", enabled)


@pytest.mark.parametrize("kind", ["material", "hyperelastic"])
def test_time_increment_jax(x64, kind):
    """A time-dependent JAX material takes the time increment as input, it equals the
    finite strain viscoelastic material of tensortrax."""

    import felupe.constitution.jax as mat

    Material, fun, nstatevars, to_jax, from_jax = {
        "material": (mat.Material, jax_viscoelastic_lagrange, 6, np.copy, np.copy),
        "hyperelastic": (mat.Hyperelastic, jax_viscoelastic, 9, to_full, to_triu),
    }[kind]

    # non-zero old state variables of a previous increment
    reference = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, mu=1.0, eta=2.0, nstatevars=6
    )
    F, Ci = kinematics()
    Ci = reference.gradient([F, Ci], dtime=0.5)[-1]
    assert not np.allclose(Ci, 0)

    umat = Material(fun, mu=1.0, eta=2.0, nstatevars=nstatevars)
    assert umat.time_dependent

    # time increments of different types
    for dtime in [0.0, np.float64(0.1), np.float32(2.0), 3]:
        P, statevars_new = umat.gradient([F, to_jax(Ci)], dtime=dtime)
        A = umat.hessian([F, to_jax(Ci)], dtime=dtime)[0]

        P_ref, statevars_ref = reference.gradient([F, Ci], dtime=float(dtime))
        A_ref = reference.hessian([F, Ci], dtime=float(dtime))[0]

        assert np.allclose(P, P_ref, rtol=1e-10, atol=1e-12)
        assert np.allclose(from_jax(statevars_new), statevars_ref, atol=1e-12)
        assert np.allclose(A, A_ref, rtol=1e-10, atol=1e-12)

    # the compiled functions are re-used for all time increments
    assert umat._grad._cache_size() == 1
    assert umat._hess._cache_size() == 1

    # a constant time increment as material parameter
    old = Material(fun, mu=1.0, eta=2.0, dtime=0.1, nstatevars=nstatevars)
    assert not old.time_dependent

    x = [F, to_jax(Ci)]
    assert_equal_results(umat.gradient(x, dtime=0.1), old.gradient(x))
    assert_equal_results(umat.hessian(x, dtime=0.1), old.hessian(x))

    with pytest.raises(ValueError, match="time-dependent"):
        umat.gradient(x)

    with pytest.raises(TypeError, match="material parameter `dtime` is constant"):
        old.gradient(x, dtime=0.1)

    # the time increment is passed through a nearly-incompressible material
    F, p, J, Ci = wrapper_inputs()["mixed"]
    wrapped = fem.NearlyIncompressible(umat, bulk=50.0)
    assert wrapped.time_dependent

    x = [F, p, J, to_jax(Ci)]
    P = wrapped.gradient(x, dtime=0.1)[0]
    P_ref = fem.NearlyIncompressible(reference, bulk=50.0).gradient(
        [F, p, J, Ci], dtime=0.1
    )[0]
    assert np.allclose(P, P_ref, rtol=1e-10, atol=1e-12)


def test_time_increment_jax_detection():
    "The time-dependency of a JAX material is detected by the signature."

    import felupe.constitution.jax as mat

    kwargs = dict(mu=1.0, eta=1.0)

    assert mat.Material(
        jax_viscoelastic_lagrange, **kwargs, nstatevars=6
    ).time_dependent
    assert mat.Hyperelastic(jax_viscoelastic, **kwargs, nstatevars=9).time_dependent

    # parallel evaluations and without just-in-time compilation
    for options in [dict(parallel=True), dict(jit=False)]:
        umat = mat.Material(
            jax_viscoelastic_lagrange, **kwargs, nstatevars=6, **options
        )
        assert umat.time_dependent

        umat = mat.Hyperelastic(jax_viscoelastic, **kwargs, nstatevars=9, **options)
        assert umat.time_dependent

    # the signature of a decorated function is taken from the original function
    def viscous(F, statevars, eta, dtime):
        return eta * dtime * jnp.eye(3), statevars

    for decorator in [mat.total_lagrange, mat.updated_lagrange]:
        umat = mat.Material(decorator(viscous), eta=1.0, nstatevars=1)
        assert umat.time_dependent

    # a time increment with a default value or by the default parameters of a model
    def viscoelastic_default(C, statevars, mu, eta, dtime=1.0):
        return jax_viscoelastic(C, statevars, mu, eta, dtime)

    def viscoelastic_kwargs(C, statevars, mu, eta, dtime):
        return jax_viscoelastic(C, statevars, mu, eta, dtime)

    viscoelastic_kwargs.kwargs = dict(mu=0.0, eta=1.0, dtime=1.0)

    for fun in [viscoelastic_default, viscoelastic_kwargs]:
        assert not mat.Hyperelastic(fun, **kwargs, nstatevars=9).time_dependent

    # materials without a time increment
    neo_hooke = mat.models.hyperelastic.neo_hooke
    assert not mat.Hyperelastic(neo_hooke, mu=1.0).time_dependent

    umat = mat.Material(jax_viscoelastic_lagrange, **kwargs, dtime=1.0, nstatevars=6)
    assert not umat.time_dependent


def test_time_increment_jax_job(x64):
    "A time-dependent JAX material in a solid body with a time step item."

    import felupe.constitution.jax as mat

    def evaluate(dtime=None):
        mesh = fem.Cube(n=2)
        region = fem.RegionHexahedron(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        boundaries = fem.dof.uniaxial(field, clamped=True)

        kwargs = {} if dtime is None else {"dtime": dtime}
        umat = mat.Hyperelastic(
            jax_viscoelastic, mu=1.0, eta=1.0, nstatevars=9, **kwargs
        )
        solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=20.0)

        move = fem.math.linsteps([0, 0.3], num=3)
        ramp = {boundaries["move"]: move}

        if dtime is None:
            # the first substep is undeformed, i.e. its time increment doesn't matter
            time = fem.TimeStep(items=[solid])
            ramp[time] = 0.5 * np.arange(len(move))

        step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
        fem.Job(steps=[step]).evaluate(verbose=0)

        return field[0].values, solid.results.statevars, umat

    u, statevars, umat = evaluate()
    u_old, statevars_old, _ = evaluate(dtime=0.5)

    assert umat.time_dependent
    assert np.allclose(u, u_old, rtol=1e-10, atol=1e-12)
    assert np.allclose(statevars, statevars_old, rtol=1e-10, atol=1e-12)
    assert not np.allclose(statevars, 0)

    # the compiled functions are re-used for all substeps
    assert umat._grad._cache_size() == 1
    assert umat._hess._cache_size() == 1


def strain_viscoelastic(**kwargs):
    "Return a strain-based linear-elastic viscoelastic material."
    return fem.MaterialStrain(
        material=fem.linear_elastic_viscoelastic,
        λ=2.0,
        μ=1.0,
        G=[3.0, 2.0],
        τ=[10.0, 100.0],
        statevars=((2, 3, 3),),
        **kwargs,
    )


@pytest.mark.parametrize("framework", ["small-strain", "total-lagrange"])
def test_time_increment_strain(framework):
    """A strain-based time-dependent material takes the time increment as input, it
    equals the time increment as material parameter (also by its former name Δt)."""

    F, _ = kinematics()
    umat = strain_viscoelastic(framework=framework)
    assert umat.time_dependent

    # non-zero old state variables (viscoelastic stresses) of a previous increment
    statevars = np.zeros((*umat.x[-1].shape, *F.shape[-2:]))
    statevars = umat.gradient([F, statevars], dtime=1.0)[-1]
    F = F + 0.05 * kinematics(seed=8)[0]

    for dtime in [0.1, 5.0]:
        for old in [
            strain_viscoelastic(framework=framework, dtime=dtime),
            strain_viscoelastic(framework=framework, Δt=dtime),
        ]:
            assert not old.time_dependent
            assert "Δt" not in old.kwargs and old.kwargs["dtime"] == dtime

            x = [F, statevars]
            assert_equal_results(umat.gradient(x, dtime=dtime), old.gradient(x))
            assert_equal_results(umat.hessian(x, dtime=dtime), old.hessian(x))

    # the time increment is effective
    x = [F, statevars]
    assert not np.allclose(
        umat.gradient(x, dtime=0.1)[0], umat.gradient(x, dtime=5.0)[0]
    )

    # the limit for a zero time increment (no relaxation, instantaneous shear moduli)
    for method in [umat.gradient, umat.hessian]:
        res = method(x, dtime=0.0)
        assert np.all(np.isfinite(res[0]))
        assert np.allclose(res[0], method(x, dtime=1e-10)[0], rtol=1e-8, atol=1e-10)

    # the time increment is passed to all chunks of cells
    for parallel in [False, True]:
        chunked = strain_viscoelastic(framework=framework)
        chunked.chunksize = 2 * F.shape[-2]
        chunked.parallel = parallel

        for method in ["gradient", "hessian"]:
            assert_equal_results(
                getattr(chunked, method)(x, dtime=0.3),
                getattr(umat, method)(x, dtime=0.3),
            )

    with pytest.raises(ValueError, match="time-dependent"):
        umat.gradient(x)

    with pytest.raises(TypeError, match="material parameter `dtime` is constant"):
        strain_viscoelastic(dtime=1.0).gradient(x, dtime=0.1)


def test_time_increment_strain_detection():
    "The time-dependency of a strain-based material is detected by the signature."

    # a material formulation with an argument Δt (not renamed to dtime)
    def fun(dε, εn, σn, ζn, Δt, **kwargs):
        return None, σn, ζn

    umat = fem.MaterialStrain(material=fun, Δt=1.0)
    assert not umat.time_dependent
    assert umat.kwargs == {"Δt": 1.0}

    # a material formulation without a signature (not evaluated here)
    umat = fem.MaterialStrain(material=max, Δt=1.0)
    assert not umat.time_dependent
    assert umat.kwargs == {"Δt": 1.0}

    # a material formulation with a time increment
    def viscous(dε, εn, σn, ζn, η, dtime, **kwargs):
        I = fem.math.identity(dε)
        dσdε = η / dtime * fem.math.cdya_ik(I, I) if kwargs["tangent"] else None
        return dσdε, η / dtime * dε, ζn

    umat = fem.MaterialStrain(material=viscous, η=2.0)
    assert umat.time_dependent

    F, _ = kinematics()
    statevars = np.zeros((*umat.x[-1].shape, *F.shape[-2:]))
    σ, statevars_new = umat.gradient([F, statevars], dtime=4.0)
    assert np.allclose(σ, fem.math.sym(F - fem.math.identity(F)) / 2)

    # materials without a time increment
    assert not fem.MaterialStrain(material=fem.linear_elastic, λ=2, μ=1).time_dependent
    assert not fem.LinearElasticPlasticIsotropicHardening(
        E=1, nu=0.3, sy=1, K=0.1
    ).time_dependent


def test_time_increment_strain_job():
    "A strain-based time-dependent material in a solid body with a time step item."

    def evaluate(dtime=None):
        mesh = fem.Cube(n=2)
        region = fem.RegionHexahedron(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        boundaries = fem.dof.uniaxial(field, clamped=True)

        kwargs = {} if dtime is None else {"dtime": dtime}
        umat = strain_viscoelastic(**kwargs)
        solid = fem.SolidBody(umat, field)

        move = fem.math.linsteps([0, 0.05], num=3)
        ramp = {boundaries["move"]: move}

        if dtime is None:
            # the first substep is undeformed, i.e. its time increment doesn't matter
            time = fem.TimeStep(items=[solid])
            ramp[time] = 5.0 * np.arange(len(move))

        step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
        fem.Job(steps=[step]).evaluate(verbose=0)

        return field[0].values, solid.results.statevars

    u, statevars = evaluate()
    u_old, statevars_old = evaluate(dtime=5.0)

    assert np.allclose(u, u_old, rtol=1e-10, atol=1e-12)
    assert np.allclose(statevars, statevars_old, rtol=1e-10, atol=1e-12)
    assert not np.allclose(statevars[:18], 0)


@pytest.mark.parametrize("incompressible", [False, True])
def test_time_increment_view(incompressible):
    """The load cases of a time-dependent material include the times, a constant time
    increment equals the time increment as material parameter."""

    stretches = {
        "ux": fem.math.linsteps([1, 1.5, 1, 2], num=[5, 5, 10]),
        "ps": fem.math.linsteps([1, 1.5], num=5),
        "bx": fem.math.linsteps([1, 1.25, 1], num=[4, 4]),
    }

    def with_volumetric(umat):
        "Add a volumetric part for compressible load cases (a composite material)."
        return umat if incompressible else umat & fem.Volumetric(bulk=5.0)

    for Material, fun in [
        (fem.Hyperelastic, fem.constitution.finite_strain_viscoelastic),
        (fem.MaterialAD, viscoelastic_lagrange),
    ]:
        umat = with_volumetric(Material(fun, mu=1.0, eta=2.0, nstatevars=6))
        old = with_volumetric(Material(fun, mu=1.0, eta=2.0, dtime=0.2, nstatevars=6))
        assert umat.time_dependent and not old.time_dependent

        # a constant time increment (the first stretch is one, i.e. its time increment
        # of zero doesn't matter)
        cases = {key: [λ, 0.2 * np.arange(len(λ))] for key, λ in stretches.items()}

        data = umat.view(incompressible=incompressible, **cases).evaluate()
        expected = old.view(incompressible=incompressible, **stretches).evaluate()

        assert len(data) == 3
        for (λ, force, label), (λ_old, force_old, label_old) in zip(data, expected):
            assert label == label_old
            assert np.allclose(λ[0], λ_old[0])
            assert np.allclose(force, force_old, rtol=1e-10, atol=1e-12)

    # a hold time relaxes the stress
    λ = np.array([1.0, 1.5, 1.5, 1.5])
    t = np.array([0.0, 0.1, 1.0, 10.0])
    λ_, force, label = umat.view(
        incompressible=incompressible, ux=[λ, t], ps=None, bx=None
    ).evaluate()[0]

    assert force[1] > force[2] > force[3] > 0

    # a shorter holding time relaxes less (the same loading)
    t_short = np.array([0.0, 0.1, 0.19, 1.09])
    λ_, force_fast, label = umat.view(
        incompressible=incompressible, ux=[λ, t_short], ps=None, bx=None
    ).evaluate()[0]

    assert np.isclose(force_fast[1], force[1])
    assert force_fast[3] > force[3]


def test_time_increment_view_plot_and_errors():
    "Plots and errors of the load cases of time-dependent materials."

    import matplotlib.pyplot as plt

    umat = strain_viscoelastic()
    stretches = fem.math.linsteps([1, 1.1, 1.1, 1], num=[5, 5, 5])
    times = fem.math.linsteps([0, 1, 11, 31], num=[5, 5, 5])

    ax = umat.plot(ux=[stretches, times], ps=None, bx=None)
    plt.close(ax.get_figure())

    ax = umat.plot(ux=[stretches, times], ps=None, bx=None, incompressible=True)
    plt.close(ax.get_figure())

    # a time-dependent material without times (also the default load cases)
    with pytest.raises(ValueError, match="requires the \\(absolute\\) times"):
        umat.view(ux=stretches, ps=None, bx=None).evaluate()

    with pytest.raises(ValueError, match="requires the \\(absolute\\) times"):
        umat.plot(incompressible=True)

    # decreasing times and other shapes of the load cases
    with pytest.raises(ValueError, match="must not decrease"):
        umat.view(ux=[stretches, times[::-1]], ps=None, bx=None).evaluate()

    with pytest.raises(ValueError, match="stretches and times"):
        umat.view(ux=[stretches, times, times], ps=None, bx=None).evaluate()

    # a material which is not time-dependent
    with pytest.raises(TypeError, match="not time-dependent"):
        fem.NeoHooke(mu=1.0).view(
            incompressible=True, ux=[stretches, times], ps=None, bx=None
        ).evaluate()

    # a time-dependent material without state variables
    class Viscous(fem.ConstitutiveMaterial):
        time_dependent = True

        def __init__(self, eta):
            self.eta = eta
            self.x = [np.eye(3), np.zeros(0)]
            self.kwargs = {"eta": eta}

        def _gradient(self, x, dtime):
            F, statevars = x[0], x[-1]
            return [self.eta / (1 + dtime) * (F - fem.math.identity(F)), statevars]

    λ = np.array([1.0, 1.5, 1.5])
    λ_, force, label = (
        Viscous(eta=2.0)
        .view(incompressible=True, ux=[λ, [0.0, 0.0, 1.0]], ps=None, bx=None)
        .evaluate()[0]
    )

    assert np.isclose(force[2], force[1] / 2)


def test_time_increment_optimize():
    "The material parameters of a time-dependent material are fitted with times."

    umat = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, mu=1.5, eta=3.0, nstatevars=6
    )

    # synthetic experimental data: loading, hold, unloading
    λ = fem.math.linsteps([1, 1.5, 1.5, 1], num=[10, 10, 10])
    t = fem.math.linsteps([0, 0.5, 5.5, 6.0], num=[10, 10, 10])
    λ_, P, label = umat.view(
        incompressible=True, ux=[λ, t], ps=None, bx=None
    ).evaluate()[0]

    start = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, mu=1.0, eta=1.0, nstatevars=6
    )
    umat_new, res = start.optimize(ux=[λ, P, t], incompressible=True)

    # the time increment is not a material parameter
    assert list(umat_new.kwargs.keys()) == ["mu", "eta"]
    assert umat_new.time_dependent

    assert np.isclose(umat_new.kwargs["mu"], 1.5, rtol=1e-4)
    assert np.isclose(umat_new.kwargs["eta"], 3.0, rtol=1e-4)
    assert len(res.fun) == len(λ)

    # the data without times
    with pytest.raises(ValueError, match="requires the \\(absolute\\) times"):
        start.optimize(ux=[λ, P], incompressible=True)


def test_time_increment_optimize_constant():
    "A constant time increment is not a material parameter, i.e. it is not fitted."

    kwargs = dict(nstatevars=6, dtime=0.05)
    umat = fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic, mu=1.5, eta=3.0, **kwargs
    )
    assert not umat.time_dependent

    # synthetic experimental data with the constant time increment: loading, unloading
    λ = fem.math.linsteps([1, 1.5, 1], num=[10, 10])
    λ_, P, label = umat.view(incompressible=True, ux=λ, ps=None, bx=None).evaluate()[0]

    for start in [
        fem.Hyperelastic(
            fem.constitution.finite_strain_viscoelastic, mu=1.0, eta=1.0, **kwargs
        ),
        fem.Hyperelastic(
            fem.constitution.finite_strain_viscoelastic, mu=1.0, eta=1.0, **kwargs
        )
        & fem.Volumetric(bulk=5000.0),
    ]:
        umat_new, res = start.optimize(ux=[λ, P], incompressible=True)

        # the constant time increment is kept
        assert umat_new.kwargs["dtime"] == 0.05
        assert len(res.x) == len(res.dx) == len(start.kwargs) - 1

        assert np.isclose(umat_new.kwargs["mu"], 1.5, rtol=1e-4)
        assert np.isclose(umat_new.kwargs["eta"], 3.0, rtol=1e-4)


if __name__ == "__main__":
    for Material, fun in [
        (fem.Hyperelastic, fem.constitution.finite_strain_viscoelastic),
        (fem.MaterialAD, viscoelastic_lagrange),
    ]:
        test_time_increment_input(Material, fun)

    test_time_increment_material()
    test_time_increment_detection()
    test_time_increment_errors()
    test_time_increment_custom()
    test_time_increment_wrappers()
    test_time_increment_external()
    test_time_increment_jax_detection()

    for framework in ["small-strain", "total-lagrange"]:
        test_time_increment_strain(framework)

    test_time_increment_strain_detection()
    test_time_increment_strain_job()

    for incompressible in [False, True]:
        test_time_increment_view(incompressible)

    test_time_increment_view_plot_and_errors()
    test_time_increment_optimize()
