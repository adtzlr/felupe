# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
import tensortrax.math as tm

import felupe as fem


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
    takes_time_increment = fem.constitution._material.takes_time_increment
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
