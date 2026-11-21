"""Tests for the mattisbardeen module."""

import mattisbardeen as mb
import numpy as np
import pytest
import scipy.constants as sc
import warnings
from scipy.special import ellipe, ellipk, i0e, k0e


# Constants (all energies in eV)
kb = sc.k / sc.e
h = sc.h / sc.e
hbar = sc.h / (2 * sc.pi) / sc.e


def test_fermi():
    """Test the fermi function."""

    # Temperature [K]
    T = 4.

    # Energy [eV]
    e = np.linspace(-0.01, 0.01, 101)

    # Using standard formula
    fe1 = 1 / (1 + np.exp(e / kb / T))

    # Using optimized formula
    fe2 = mb.fermi(e, T)

    np.testing.assert_almost_equal(fe1, fe2)


def test_gap_voltage():
    """Check the BCS gap against its analytic limits.

    Near Tc:  Delta/Delta0 -> e^gamma sqrt(8/(7 zeta(3))) sqrt(1 - t)
    Low T:    1 - Delta/Delta0 -> sqrt(2 pi tau) exp(-1/tau),
              with tau = k_B T / Delta0 = t e^gamma / pi

    """

    Vgap0, Tc = 2.8e-3, 9.0

    # Endpoints
    assert mb.gap_voltage(0., Vgap0, Tc) == Vgap0
    assert mb.gap_voltage(Tc, Vgap0, Tc) == 0.
    assert mb.gap_voltage(1.5 * Tc, Vgap0, Tc) == 0.

    # Monotonically decreasing between 0 and Tc
    t = np.linspace(0.05, 0.995, 50)
    ratio = np.array([mb.gap_voltage(ti * Tc, Vgap0, Tc) for ti in t]) / Vgap0
    assert (np.diff(ratio) < 0).all()

    # Ginzburg-Landau limit near Tc (relative error is O(1 - t))
    zeta3 = 1.2020569031595942
    coeff = np.exp(np.euler_gamma) * np.sqrt(8 / (7 * zeta3))
    for ti in [0.999, 0.9999]:
        expected = coeff * np.sqrt(1 - ti)
        assert mb.gap_voltage(ti * Tc, 1., Tc) == pytest.approx(expected, rel=2 * (1 - ti))

    # Exponentially small correction at low temperature
    for ti in [0.15, 0.2]:
        tau = ti * np.exp(np.euler_gamma) / np.pi
        expected = np.sqrt(2 * np.pi * tau) * np.exp(-1 / tau)
        assert 1 - mb.gap_voltage(ti * Tc, 1., Tc) == pytest.approx(expected, rel=0.05)

    # Defaults come from PARAM
    assert mb.gap_voltage(4.) == mb.gap_voltage(4., mb.PARAM['Vgap0'], mb.PARAM['Tc'])


def test_vgap_default_uses_bcs():
    """Omitting Vgap should be equivalent to passing gap_voltage(T)."""

    f = np.array([100e9, 500e9, 900e9])
    T, d = 4.2, 200e-9
    param = dict(sigma_n=1.74e7, Tc=8.1, Vgap0=2.65e-3, lambda0=86*sc.nano)
    vgap = mb.gap_voltage(T, param['Vgap0'], param['Tc'])

    np.testing.assert_array_equal(mb.conductance(f, T, **param),
                                  mb.conductance(f, T, vgap, **param))
    np.testing.assert_array_equal(mb.surface_impedance(f, d, T, **param),
                                  mb.surface_impedance(f, d, T, vgap, **param))

    # Above Tc there is no gap: the normal state
    np.testing.assert_array_equal(mb.conductance(f, 9.0, **param), param['sigma_n'] * np.ones(3))
    with pytest.raises(ValueError):
        mb.conductance(f, 4.2, -1e-3, **param)


def test_zero_temperature_closed_form():
    """At T=0 the Mattis-Bardeen integrals reduce to complete elliptic
    integrals (D. C. Mattis and J. Bardeen, Phys. Rev. 111, 412 (1958),
    Eqs. 3.11-3.13; also Tinkham, Introduction to Superconductivity,
    Eq. 3.125). With x = hbar*omega / 2*Delta:

        sigma1/sigma_n = 0                                  (x < 1)
                       = (1 + 1/x) E(k) - (2/x) K(k)        (x > 1)
        sigma2/sigma_n = (1/2)(1 + 1/x) E(k') + (1/2)(1/x - 1) K(k')

    where k = |1 - x| / (1 + x) and k' = sqrt(1 - k^2). (scipy's ellipk and
    ellipe take the parameter m = k^2.)

    This is an independent check of the numerical integration, including
    the near-gap region x -> 1.

    """

    Vgap = 2.8e-3
    sigma_n = mb.PARAM['sigma_n']

    for x in [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.001, 1.01, 1.5, 3., 10.]:
        f = x * Vgap / mb.h
        sigma = mb.conductance(float(f), 0., Vgap) / sigma_n

        m = ((1 - x) / (1 + x)) ** 2
        sigma1 = (1 + 1/x) * ellipe(m) - (2/x) * ellipk(m) if x > 1 else 0.
        sigma2 = 0.5 * (1 + 1/x) * ellipe(1 - m) + 0.5 * (1/x - 1) * ellipk(1 - m)

        assert sigma.real == pytest.approx(sigma1, rel=1e-6, abs=1e-9)
        assert -sigma.imag == pytest.approx(sigma2, rel=1e-6)

    # Low-frequency limit: sigma2/sigma_n -> pi*Delta/(hbar*omega) = pi/(2x)
    x = 1e-4
    sigma = mb.conductance(x * Vgap / mb.h, 0., Vgap) / sigma_n
    assert -sigma.imag == pytest.approx(np.pi / (2 * x), rel=1e-4)


def test_mpmath_reference():
    """Compare against 40-digit mpmath values of the original integrals
    (validation-data/reference/generate.py), on a grid that includes T=0,
    near Tc, very small gaps, very low frequencies and frequencies within
    1e-4 of the gap frequency."""

    filename = 'validation-data/reference/mpmath-conductance.txt'
    data = np.genfromtxt(filename, delimiter=',')
    sigma_n = mb.PARAM['sigma_n']

    for Vgap, T in np.unique(data[:, :2], axis=0):
        rows = data[(data[:, 0] == Vgap) & (data[:, 1] == T)]
        x, ref1, ref2 = rows[:, 2:].T
        sigma = mb.conductance(x * Vgap / mb.h, T, Vgap) / sigma_n
        nonzero = ref1 != 0
        np.testing.assert_allclose(sigma.real[nonzero], ref1[nonzero], rtol=1e-9)
        np.testing.assert_array_equal(sigma.real[~nonzero], 0.)
        np.testing.assert_allclose(-sigma.imag, ref2, rtol=1e-9)


def test_low_temperature_limit():
    """For kT, hbar*omega << Delta (e.g., Gao, PhD thesis, Caltech 2008):

        sigma1/sigma_n = (4 Delta / hw) exp(-Delta/kT) sinh(xi) K0(xi)
        sigma2/sigma_n = (pi Delta / hw) [1 - 2 exp(-Delta/kT) exp(-xi) I0(xi)]

    with xi = hw / 2kT. Corrections are O(kT/Delta, hw/Delta)."""

    delta = 1.4e-3
    for T in [0.3, 0.5]:
        kT = mb.kb * T
        for x in [1e-4, 1e-3, 1e-2]:
            hw = 2 * delta * x
            xi = hw / (2 * kT)
            # sinh(xi) K0(xi) and exp(-xi) I0(xi) via exponentially scaled Bessel functions
            sigma1 = 4 * delta / hw * np.exp(-delta / kT) * -np.expm1(-2 * xi) / 2 * k0e(xi)
            sigma2 = np.pi * delta / hw * (1 - 2 * np.exp(-delta / kT) * i0e(xi))

            sigma = mb.conductance(hw / mb.h, T, 2 * delta) / mb.PARAM['sigma_n']
            tol = kT / delta + hw / delta
            assert sigma.real == pytest.approx(sigma1, rel=tol)
            assert -sigma.imag == pytest.approx(sigma2, rel=tol)


def test_zero_frequency():
    """At f=0 the surface impedance of a superconductor is zero (not NaN)."""

    f = np.array([0., 1e9, 1e11])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        zs = mb.surface_impedance(f, 100e-9, 4., 2.8e-3)
    assert zs[0] == 0
    assert np.isfinite(zs).all()
    assert mb.surface_impedance(0., 100e-9, 4., 2.8e-3) == 0


def test_input_shapes():
    """Scalars give scalars, lists are accepted, and array shapes are kept.
    (Values can differ in the last bit with array size, since the summation
    order of the quadrature's matrix product can change.)"""

    T, Vgap, d = 4., 2.8e-3, 100e-9
    f = np.array([[100e9, 300e9], [700e9, 900e9]])

    sigma = mb.conductance(f, T, Vgap)
    assert sigma.shape == f.shape
    for idx in np.ndindex(f.shape):
        sigma_i = mb.conductance(f[idx], T, Vgap)
        assert isinstance(sigma_i, complex)
        assert sigma_i == pytest.approx(sigma[idx], rel=1e-13)
    np.testing.assert_allclose(mb.conductance(list(f.ravel()), T, Vgap), sigma.ravel(), rtol=1e-13)

    zs = mb.surface_impedance(f, d, T, Vgap)
    assert zs.shape == f.shape
    assert isinstance(mb.surface_impedance(int(300e9), d, T, Vgap), complex)

    with pytest.raises(ValueError):
        mb.conductance(-1e9, T, Vgap)


def test_chunking(monkeypatch):
    """Results must not depend on how frequencies are split into chunks."""

    f = np.linspace(0, 1500, 301) * 1e9
    expected = mb.conductance(f, 4., 2.8e-3)
    monkeypatch.setattr(mb, '_CHUNK', 7)
    np.testing.assert_allclose(mb.conductance(f, 4., 2.8e-3), expected, rtol=1e-13)


def test_belitsky1995_fig6():
    """Recreate Figure 6 from:

        V. Y. Belitsky, S. W. Jacobsson, L. V Filippenko, and E. L. Kollberg,
        “Theoretical and Experimental Studies of Nb-Based Tuning Circuits for
        THz SIS Mixers,” in Proceedings of the Sixth International Symposium
        on Space Terahertz Technology (ISSTT), 1995, pp. 87–102.

    """

    # Tc = 8.1 K -------------------------------------------------------------

    # Parameters from Table 3
    T = 4.2
    Vgap = 2.53e-3
    param = dict(sigma_n=1.74e7, Tc=8.1, Vgap0=2.65e-3, lambda0=86*sc.nano)

    # Calculate sigma_1
    filename = 'validation-data/belitsky1995/belitsky1995-fig6-sigma1-temp8.1.txt'
    fghz, lit_sigma1 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)

    max_difference = np.abs(lit_sigma1 - sigma.real / param['sigma_n']).max()
    assert max_difference < 0.01

    # Calculate sigma_2
    filename = 'validation-data/belitsky1995/belitsky1995-fig6-sigma2-temp8.1.txt'
    fghz, lit_sigma2 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)

    max_difference = np.abs(lit_sigma2 + sigma.imag / param['sigma_n']).max()
    assert max_difference < 0.05

    # Tc = 9.0 K -------------------------------------------------------------

    # Parameters from Table 3
    T = 4.2
    Vgap = 2.85e-3
    param = dict(sigma_n=1.57e7, Tc=9.0, Vgap0=2.95e-3, lambda0=86*sc.nano)

    # Calculate sigma_1
    filename = 'validation-data/belitsky1995/belitsky1995-fig6-sigma1-temp9.0.txt'
    fghz, lit_sigma1 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)

    max_difference = np.abs(lit_sigma1 - sigma.real / param['sigma_n']).max()
    assert max_difference < 0.02

    # Calculate sigma_2
    filename = 'validation-data/belitsky1995/belitsky1995-fig6-sigma2-temp9.0.txt'
    fghz, lit_sigma2 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)

    max_difference = np.abs(lit_sigma2 + sigma.imag / param['sigma_n']).max()
    assert max_difference < 0.05


def test_belitsky2006_fig2():
    """Recreate Figure 2 from:

        V. Belitsky, C. Risacher, M. Pantaleev, and V. Vassilev,
        “Superconducting microstrip line model studies at millimetre and
        sub-millimetre waves,” Int. J. Infrared Millimeter Waves, vol. 27,
        no. 6, pp. 809–834, 2006.

    The caption gives sigma_n = 1.739e7 S/m, Tc = 8.1 K, lambda0 = 85 nm and
    T = 4 K, but not the gap. The gap at T=0 follows from the paper's Eq. 2,
    lambda0 = sqrt(hbar / (pi mu0 sigma_n Delta0)), and the gap at 4 K from
    BCS theory (2.546 mV; the gap frequency, 616 GHz, is close to the
    "about 650 GHz" in the text).

    """

    # Parameters (Fig. 2 caption)
    T = 4.0
    sigma_n, Tc, lambda0 = 1.739e7, 8.1, 85*sc.nano

    # Gap voltage at T=0 from Eq. 2, and at T from BCS theory
    Vgap0 = 2 * mb.hbar / (np.pi * sc.mu_0 * sigma_n * lambda0**2)
    param = dict(sigma_n=sigma_n, Tc=Tc, Vgap0=Vgap0, lambda0=lambda0)
    assert mb._lambda0(**param) == pytest.approx(lambda0, rel=1e-12)
    Vgap = mb.gap_voltage(T, Vgap0, Tc)

    # Calculate sigma_1
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-sigma1.txt'
    fghz, lit_sigma1 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)
    max_difference = np.abs(lit_sigma1 - sigma.real / param['sigma_n']).max()
    assert max_difference < 0.01

    # Calculate sigma_2
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-sigma2.txt'
    fghz, lit_sigma2 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)
    max_difference = np.abs(lit_sigma2 + sigma.imag / param['sigma_n']).max()
    assert max_difference < 0.15  # sigma2/sigma_n is up to ~5 here: ~3% (digitizing)

    # Calculate E-field penetration
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-lambda.txt'
    fghz, lit_lambda = np.genfromtxt(filename, delimiter=',').T
    d = 500e-9  # The thickness is not specified in the paper!!!
    zs = mb.surface_impedance(fghz * 1e9, d, T, Vgap, **param)
    _lambda = mb.efield_penetration(zs, fghz * 1e9) / mb._lambda0(**param)
    max_difference = np.abs(_lambda - lit_lambda).max()
    assert max_difference < 0.07

    # Calculate H-field penetration
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-delta.txt'
    fghz, lit_delta = np.genfromtxt(filename, delimiter=',').T
    d = 500e-9  # The thickness is not specified in the paper!!!
    zs = mb.surface_impedance(fghz * 1e9, d, T, Vgap, **param)
    _delta = mb.hfield_penetration(zs, fghz * 1e9) / mb._lambda0(**param)
    max_difference = np.abs(_delta - lit_delta).max()
    assert max_difference < 0.08


def test_kerr1996_finite_thickness():
    """Compare the film model with the closed-form surface impedance of a
    film of thickness t backed by free space, from:

        A. R. Kerr, “Surface impedance of superconductors and normal
        conductors in EM simulators,” NRAO Electronics Division Internal
        Report 302, 1996 (p. 5, and Appendix A4, A5).

    Normal conductor (k = (1 + j) / delta, delta the skin depth):

        Zs = (k/sigma) [e^kt + r e^-kt] / [e^kt - r e^-kt],
        r = (Zn - k/sigma) / (Zn + k/sigma)

    Superconductor (London, penetration depth lambda):

        Zs = j w mu0 lambda [e^(t/lambda) + r e^(-t/lambda)] /
                            [e^(t/lambda) - r e^(-t/lambda)],
        r = (Zn - j w mu0 lambda) / (Zn + j w mu0 lambda)

    with Zn the impedance of free space. Kerr's circuit model has no
    displacement current; in film_impedance it changes Zs by O((k0 delta)^2)
    or O((k0 lambda)^2), < 1e-5 here. For Zn >> |Zs| the superconductor
    reduces to j w mu0 lambda coth(t / lambda), which is method='simple'.

    """

    z_n = sc.mu_0 * sc.c
    f = np.array([10e9, 100e9, 1e12])
    w = 2 * np.pi * f

    def kerr(z1, kt):
        r = (z_n - z1) / (z_n + z1)
        return z1 * (np.exp(kt) + r * np.exp(-kt)) / (np.exp(kt) - r * np.exp(-kt))

    # Normal conductor (copper-like), thin to thick compared with the skin depth
    rho = 2e-8
    skin = np.sqrt(2 * rho / (w * sc.mu_0))
    k = (1 + 1j) / skin
    for t_over_skin in [0.01, 0.1, 0.5, 1., 3.]:
        t = t_over_skin * skin
        zs = np.array([mb.film_impedance(fi, [(1 / rho, ti)]) for fi, ti in zip(f, t)])
        np.testing.assert_allclose(zs, kerr(k * rho, k * t), rtol=1e-5)

    # Superconductor in the London limit: sigma = 1 / (j w mu0 lambda^2)
    lam = 100e-9
    sigma = 1 / (1j * w * sc.mu_0 * lam**2)
    for t_over_lam in [0.1, 0.5, 1., 2., 5.]:
        t = t_over_lam * lam
        zs = mb.film_impedance(f, [(sigma, t)])
        np.testing.assert_allclose(zs, kerr(1j * w * sc.mu_0 * lam, t / lam), rtol=1e-5)

    # Zn >> |Zs|: j w mu0 lambda coth(t / lambda), i.e., method='simple'
    t = 300e-9
    zs = mb.film_impedance(f, [(sigma, t)], z_back=np.inf)
    np.testing.assert_allclose(zs, mb._surface_impedance_simple(f, t, lam), rtol=1e-5)


def test_compare_simple_and_MB_models():
    """Compare the surface impedance that is calculated from Mattis-Bardeen
    theory to the results from [1]."""

    # Frequency
    f = np.arange(100, 1000, 100) * 1e9

    # Surface impedance (using both models)
    zs_simple = mb.surface_impedance(f, 250e-9, 4., 2.75e-3, method='simple')
    zs_mb = mb.surface_impedance(f, 250e-9, 4., 2.75e-3, method='MB')

    # Below ~fgap/2, both models should be roughly the same
    mask = f < 350e9

    # Check real values
    assert (zs_simple.real == 0.).all()
    assert (zs_mb[mask].real < 0.01).all()

    # Check imaginary values
    error_imag = (zs_simple.imag - zs_mb.imag) / zs_mb.imag * 100.
    max_error = np.abs(error_imag[mask]).max()
    assert max_error < 5.  # below 5% relative error

    # DEBUG
    # plt.figure()
    # plt.plot(f/1e9, zs_simple.real)
    # plt.plot(f/1e9, zs_mb.real)
    # plt.figure()
    # plt.plot(f/1e9, zs_simple.imag)
    # plt.plot(f/1e9, zs_mb.imag)
    # plt.show()


# Films of finite thickness ---------------------------------------------------

def test_normal_state():
    """Above Tc: the conductance is sigma_n, the surface impedance that of a
    normal metal (thick: skin effect; DC: the sheet resistance), the
    penetration depth is infinite, and method='simple' is not defined."""

    f = np.array([0., 10e9, 100e9])
    sigma_n, d = 1.74e7, 100e-6
    zs = mb.surface_impedance(f, d, 10., sigma_n=sigma_n)
    assert zs[0] == pytest.approx(1 / (sigma_n * d))
    np.testing.assert_allclose(zs[1:], (1 + 1j) * np.sqrt(np.pi * f[1:] * sc.mu_0 / sigma_n), rtol=1e-12)
    np.testing.assert_allclose(mb.surface_impedance(f[1:], d, 4.2, 0., sigma_n=sigma_n), zs[1:], rtol=1e-14)
    assert mb.penetration_depth(10.) == np.inf
    with pytest.raises(ValueError):
        mb.surface_impedance(f, d, 10., method='simple')


def test_film_impedance_normal_metal():
    """Thick films: the skin-effect impedance (1 + j) sqrt(pi f mu0 rho).
    Thin films: the sheet resistance rho / d in parallel with the backing
    (free space by default; a perfect conductor shorts the film)."""

    f, rho = 10e9, 2e-8
    zs = mb.film_impedance(f, [(1 / rho, 10e-6)])
    assert isinstance(zs, complex)
    assert zs == pytest.approx((1 + 1j) * np.sqrt(np.pi * f * sc.mu_0 * rho), rel=1e-7)
    z_vac = sc.mu_0 * sc.c
    thin = mb.film_impedance(f, [(1 / rho, 5e-9)])
    assert thin == pytest.approx(1 / (5e-9 / rho + 1 / z_vac), rel=1e-4)
    shorted = mb.film_impedance(f, [(1 / rho, 5e-9)], z_back=0.)
    assert abs(shorted) < 1e-2 * abs(thin)

    with pytest.raises(ValueError):
        mb.film_impedance(-1., [(1 / rho, 5e-9)])
    with pytest.raises(ValueError):
        mb.film_impedance(f, [])


def test_film_impedance_zero_frequency():
    """At f=0: sheet conductances in parallel with the backing, continuous
    with low frequencies; a superconducting film is exactly zero, as with
    method='MB'."""

    rho, d = 2e-8, 5e-9
    z_vac = sc.mu_0 * sc.c
    assert mb.film_impedance(0., [(1 / rho, d)]) == pytest.approx(1 / (d / rho + 1 / z_vac), rel=1e-14)
    assert mb.film_impedance(0., [(1 / rho, d)], z_back=np.inf) == pytest.approx(rho / d, rel=1e-14)
    assert mb.film_impedance(0., [(1 / rho, d)], z_back=0.) == 0
    stack = mb.film_impedance(0., [(1 / rho, d), (1 / (2 * rho), 2 * d)])
    assert stack == pytest.approx(1 / (d / rho + d / rho + 1 / z_vac), rel=1e-14)
    assert mb.film_impedance(1e3, [(1 / rho, d)]) == pytest.approx(mb.film_impedance(0., [(1 / rho, d)]), rel=1e-6)

    # Arrays with f=0, and per-frequency conductivity and backing
    f = np.array([0., 1e9, 1e11])
    zs = mb.film_impedance(f, [(np.full(3, 1 / rho), d)], z_back=np.array([z_vac, z_vac, np.inf]))
    assert zs[0] == pytest.approx(1 / (d / rho + 1 / z_vac), rel=1e-14)
    assert zs[1] == pytest.approx(mb.film_impedance(1e9, [(1 / rho, d)]), rel=1e-14)
    assert zs[2] == pytest.approx(mb.film_impedance(1e11, [(1 / rho, d)], z_back=np.inf), rel=1e-14)

    # Through surface_impedance: superconductor -> 0, normal state -> DC formula
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        zs = mb.surface_impedance(f, 100e-9, 4.2, method='film')
        zn = mb.surface_impedance(f, 100e-9, 10., method='film', sigma_n=1.74e7)
    assert zs[0] == 0 and np.isfinite(zs).all()
    np.testing.assert_allclose(zs[1:], mb.surface_impedance(f[1:], 100e-9, 4.2, method='film'), rtol=1e-14)
    assert zn[0] == pytest.approx(1 / (1.74e7 * 100e-9 + 1 / z_vac), rel=1e-14)
    assert mb.surface_impedance(0., 100e-9, 4.2, method='film') == 0


def test_method_names():
    """Methods are case-insensitive, and an unknown method is reported as
    such (also when z_back is given)."""

    f, d, T = np.array([100e9, 200e9]), 300e-9, 4.
    for name in ('mb', 'film', 'simple'):
        np.testing.assert_array_equal(mb.surface_impedance(f, d, T, method=name.upper()),
                                      mb.surface_impedance(f, d, T, method=name))
    with pytest.raises(ValueError, match="Method not recognized"):
        mb.surface_impedance(f, d, T, method='typo')
    with pytest.raises(ValueError, match="Method not recognized"):
        mb.surface_impedance(f, d, T, method='typo', z_back=0.)
    with pytest.raises(ValueError, match="z_back"):
        mb.surface_impedance(f, d, T, method='MB', z_back=0.)


def test_film_impedance_superconductor():
    """method='film': with z_back = inf, it agrees with the thin-film formula
    (method='MB') up to the displacement current; with free space behind a
    thin film it does not. Thick films at low frequency: an inductance
    mu0 lambda per square, with the BCS penetration depth."""

    f = np.array([100e9, 500e9, 1000e9])
    T, Vgap, d = 4.2, 2.8e-3, 50e-9
    zs_mb = mb.surface_impedance(f, d, T, Vgap)
    zs_inf = mb.surface_impedance(f, d, T, Vgap, method='film', z_back=np.inf)
    np.testing.assert_allclose(zs_inf, zs_mb, rtol=1e-5)  # displacement current: ~6e-6 at 1 THz
    zs_vac = mb.surface_impedance(f, d, T, Vgap, method='film')
    assert np.all(np.abs(zs_vac - zs_mb) / np.abs(zs_mb) > 2e-4)
    np.testing.assert_allclose(zs_vac, mb.film_impedance(f, [(mb.conductance(f, T, Vgap), d)]), rtol=1e-14)
    with pytest.raises(ValueError):
        mb.surface_impedance(f, d, T, Vgap, z_back=0.)

    # Thick film: Zs = j w mu0 lambda
    zs = mb.surface_impedance(1e9, 1e-6, 0.5, method='film')
    assert zs == pytest.approx(2j * np.pi * 1e9 * sc.mu_0 * mb.penetration_depth(0.5), rel=1e-6)
    assert mb.penetration_depth(0.) == pytest.approx(mb._lambda0(**mb.PARAM), rel=1e-14)
    assert mb.penetration_depth(0.5) == pytest.approx(mb.penetration_depth(0.), rel=1e-10)
    assert mb.penetration_depth(8.) > mb.penetration_depth(4.2)
    # BCS: Delta(T) tanh(Delta(T) / 2 kT)
    vgap = mb.gap_voltage(6.)
    expected = mb.penetration_depth(0.) * np.sqrt(mb.PARAM['Vgap0'] / vgap / np.tanh(vgap / 2 / (2 * kb * 6.)))
    assert mb.penetration_depth(6.) == pytest.approx(expected, rel=1e-12)
    assert mb.penetration_depth(4.2, Vgap=2.5e-3) == pytest.approx(
        mb.penetration_depth(0.) * np.sqrt(mb.PARAM['Vgap0'] / 2.5e-3 / np.tanh(2.5e-3 / 2 / (2 * kb * 4.2))))


def test_multilayer():
    """Stacks of films: two layers of the same film equal one thick film; a
    thick top layer hides what is below; a thin top layer is the top film
    backed by the impedance of the rest."""

    f = np.array([100e9, 500e9, 900e9])
    T, Vgap = 4.2, 2.8e-3
    sigma = mb.conductance(f, T, Vgap)
    one = mb.film_impedance(f, [(sigma, 200e-9)])
    np.testing.assert_allclose(mb.film_impedance(f, [(sigma, 100e-9), (sigma, 100e-9)]), one, rtol=1e-12)

    al = (1 / 3e-9, 300e-9)
    np.testing.assert_allclose(mb.film_impedance(f, [(sigma, 2e-6), al]), mb.film_impedance(f, [(sigma, 2e-6)]),
                               rtol=1e-6)
    stack = mb.film_impedance(f, [(sigma, 20e-9), al])
    backed = mb.film_impedance(f, [(sigma, 20e-9)], z_back=mb.film_impedance(f, [al]))
    np.testing.assert_allclose(stack, backed, rtol=1e-14)
    assert np.all(np.abs(stack - mb.film_impedance(f, [(sigma, 20e-9)])) > 1e-3)


def test_constants_at_call_time(monkeypatch):
    """The physical constants are read from scipy.constants when the
    functions are called, so they can be replaced (e.g., by an older CODATA
    set)."""

    from types import SimpleNamespace

    f, T, Vgap = np.array([100e9, 900e9]), 4.2, 2.8e-3
    sigma, zs = mb.conductance(f, T, Vgap), mb.surface_impedance(f, 100e-9, T, Vgap, method='film')
    lam = mb.penetration_depth(T)
    old = SimpleNamespace(e=1.60217733e-19, k=1.380658e-23, h=6.6260755e-34, mu_0=4e-7 * np.pi, c=299792458.)
    monkeypatch.setattr(mb, 'sc', old)
    assert np.all(mb.conductance(f, T, Vgap) != sigma)
    np.testing.assert_allclose(mb.conductance(f, T, Vgap), sigma, rtol=1e-5)
    assert np.all(mb.surface_impedance(f, 100e-9, T, Vgap, method='film') != zs)
    assert mb.penetration_depth(T) != lam


def test_simple_model_uses_lambda0():
    """method='simple' uses the lambda0 parameter (two-fluid temperature
    dependence): Zs = j w mu0 lambda coth(d / lambda)."""

    f, d, T = np.array([100e9, 200e9]), 300e-9, 4.
    for lambda0 in (80e-9, 120e-9):
        lam = lambda0 / np.sqrt(1 - (T / mb.PARAM['Tc']) ** 4)
        expected = 2j * np.pi * f * sc.mu_0 * lam / np.tanh(d / lam)
        np.testing.assert_allclose(mb.surface_impedance(f, d, T, method='simple', lambda0=lambda0), expected,
                                   rtol=1e-14)


if __name__ == "__main__":

    test_fermi()
    test_gap_voltage()
    test_vgap_default_uses_bcs()
    test_zero_temperature_closed_form()
    test_mpmath_reference()
    test_low_temperature_limit()
    test_zero_frequency()
    test_input_shapes()

    test_belitsky1995_fig6()
    test_belitsky2006_fig2()

    test_kerr1996_finite_thickness()
    test_compare_simple_and_MB_models()
