"""Calculate the electrical properties of superconductors using Mattis-Bardeen
theory.

The integrals from the Mattis-Bardeen paper are evaluated with tanh-sinh
(double-exponential) quadrature, vectorized over frequency.

References:

    [1] A. R. Kerr, “Surface Impedance of Superconductors and Normal
    Conductors in EM Simulators,” Green Bank, West Virginia, 1996.

    [2] V. Y. Belitsky and E. L. Kollberg, “Superconductor–insulator–
    superconductor tunnel strip line: Features and applications,” J.
    Appl. Phys., vol. 80, no. 8, pp. 4741–4748, Oct. 1996.

    [3] B. Mühlschlegel, “Die thermodynamischen Funktionen des
    Supraleiters,” Z. Phys., vol. 155, pp. 313–327, 1959.

"""

from copy import deepcopy

import numpy as np
import scipy.constants as sc
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit


# Constants (all energies in eV). The functions below read the constants from
# scipy.constants (``sc``) when they are called, so replacing ``sc`` (e.g.,
# with an older CODATA set, to compare with other software) changes the
# results; these module-level values are for convenience.
kb = sc.k / sc.e
h = sc.h / sc.e
hbar = sc.h / (2 * np.pi) / sc.e


def _kb():
    """Boltzmann constant, in [eV/K] (from ``sc`` at call time)."""
    return sc.k / sc.e


def _hbar():
    """Reduced Planck constant, in [eV s] (from ``sc`` at call time)."""
    return sc.h / (2 * np.pi) / sc.e


def _z_vacuum():
    """Impedance of free space, in [ohms] (from ``sc`` at call time)."""
    return sc.mu_0 * sc.c


# Integration tolerances (scipy.integrate.quad, used for the BCS gap equation)
_EPSABS = 1.49e-8
_EPSREL = 1.49e-8


# Default parameters (for niobium, from [2])
PARAM = dict(Tc=9.0,              # Critical temperature [K]
             Vgap0=2.862e-3,      # Gap voltage at T=0K [V]
             sigma_n=1.565e7,     # Normal-state conductivity [1/ohm*m]
             lambda0=85*sc.nano,  # Penetration depth at T=0K [m] (method='simple' only)
             )


# Gap voltage ----------------------------------------------------------------

# k_B*Tc / Delta(0) in weak-coupling BCS theory (= e^gamma / pi ~ 1/1.764)
_BCS_TAU0 = np.exp(np.euler_gamma) / np.pi


def gap_voltage(T, Vgap0=None, Tc=None):
    """Calculate the gap voltage at temperature T from BCS theory.

    Solves the weak-coupling BCS gap equation for the reduced gap
    Delta(T)/Delta(0) as a function of the reduced temperature T/Tc (the
    universal curve tabulated by Mühlschlegel [3]), then scales it by the
    measured gap voltage at T=0K. Using the measured Vgap0 rather than the
    BCS value 3.53*k_B*Tc keeps the result consistent for strong-coupling
    materials such as niobium.

    Args:
        T (float): temperature, in units [K]
        Vgap0 (float): gap voltage at T=0K, in units [V]
        Tc (float): critical temperature, in units [K]

    Returns:
        gap voltage at temperature T, in units [V] (zero for T >= Tc)

    >>> round(gap_voltage(0., Vgap0=2.8e-3, Tc=9.0) * 1e3, 6)
    2.8
    >>> round(gap_voltage(4.5, Vgap0=1., Tc=9.0), 4)
    0.9569

    """

    if Vgap0 is None:
        Vgap0 = PARAM['Vgap0']
    if Tc is None:
        Tc = PARAM['Tc']

    return Vgap0 * _bcs_gap_ratio(T / Tc)


def _bcs_gap_ratio(t):
    """Reduced BCS gap Delta(T)/Delta(0) at reduced temperature t = T/Tc.

    In these units the BCS gap equation is

        ln(1 / delta) = 2 int_0^inf f(E) / E dxi,   E = sqrt(xi^2 + delta^2)

    with f the Fermi function at k_B*T = t * _BCS_TAU0. The substitution
    xi = delta*sinh(u) gives dxi/E = du, leaving a smooth integrand that
    is truncated where the Fermi factor underflows.

    """

    if t <= 0:
        return 1.
    if t >= 1:
        return 0.

    tau = t * _BCS_TAU0

    def _residual(delta):
        u_max = np.arccosh(max(60 * tau / delta, 1.))
        integral = quad(lambda u: 1 - np.tanh(0.5 * delta * np.cosh(u) / tau),
                        0, u_max, epsabs=_EPSABS, epsrel=_EPSREL)[0]
        return np.log(1 / delta) - integral

    return brentq(_residual, 1e-12, 1., xtol=1e-14, rtol=1e-12)


def _resolve_gap(Vgap, T, kw):
    """Return Vgap, computing it from BCS theory if it is None. Zero means
    the normal state (T >= Tc)."""

    if Vgap is None:
        Vgap = gap_voltage(T, kw['Vgap0'], kw['Tc'])
    if Vgap < 0:
        raise ValueError("Gap voltage must not be negative.")
    return Vgap


# Quadrature rule ------------------------------------------------------------

def _tanh_sinh_rule(step, t_max):
    """Tanh-sinh (double-exponential) quadrature rule on [0, 1].

    With x = tanh(pi/2 * sinh(t)) the nodes cluster double-exponentially
    toward both endpoints, so the 1/sqrt endpoint singularities of the
    Mattis-Bardeen integrands -- and the near-singular peaks just inside an
    endpoint, which appear at low frequency and near the gap frequency -- are
    integrated with exponential convergence in 1/step.

    Returns the distances of each node from the lower and upper endpoints
    (computed directly, so they stay accurate when they underflow toward
    zero) and the weights.

    """

    t = np.arange(-t_max, t_max + step / 2, step)
    u = np.pi / 2 * np.sinh(t)
    from_a = expit(2 * u)    # (1 + x) / 2
    from_b = expit(-2 * u)   # (1 - x) / 2
    weight = step * np.pi * np.cosh(t) * from_a * from_b
    return from_a, from_b, weight


# Step 1/16 on [-4, 4] (129 nodes) agrees with 40-digit reference values to
# ~1e-11 or better (see validation-data/reference/).
_TS_FROM_A, _TS_FROM_B, _TS_WEIGHT = _tanh_sinh_rule(1 / 16, 4.)

# Number of frequency points evaluated at once (bounds temporary memory)
_CHUNK = 2048


# Conductance ----------------------------------------------------------------

def conductance(f, T, Vgap=None, **kwargs):
    """Calculate conductance from Mattis-Bardeen theory.

    Args:
        f (float or array-like): frequency, in units [Hz]
        T (float): temperature, in units [K]
        Vgap (float): gap voltage at temperature T, in units [V]. If None,
            it is calculated from Vgap0 and Tc using BCS theory (see
            gap_voltage). Zero is the normal state.

    Keyword arguments:
        Tc (float): critical temperature, in units [K]
        Vgap0 (float): gap voltage at T=0K, in units [V]
        sigma_n (float): normal-state conductance, in units [1/ohm*m]

    Returns:
        conductance (complex, or complex ndarray with the shape of f), as
        sigma1 - 1j * sigma2; sigma_n in the normal state (T >= Tc)

    """

    # Import keyword arguments
    kw = deepcopy(PARAM)
    kw.update(kwargs)
    Vgap = _resolve_gap(Vgap, T, kw)

    f = np.asarray(f, dtype=float)
    sigma = _conductance(f.ravel(), T, Vgap, kw['sigma_n']).reshape(f.shape)

    return complex(sigma) if sigma.ndim == 0 else sigma


def _conductance(f, T, Vgap, sigma_n):
    """Calculate conductance from Mattis-Bardeen theory.

    Args:
        f (ndarray): 1-D array of frequencies, in units [Hz]
        T (float): temperature, in units [K]
        Vgap (float): gap voltage at temperature T, in units [V]
        sigma_n (float): normal-state conductance, in units [1/ohm*m]

    Returns:
        conductance (complex ndarray)

    """

    if np.any(f < 0):
        raise ValueError("Frequency must be non-negative.")

    # Normal state (T >= Tc): the normal-state conductance at all frequencies
    if Vgap == 0:
        return np.full(f.shape, sigma_n, dtype=complex)

    sigma = np.empty(f.shape, dtype=complex)

    # It's a superconductor -> infinite conductivity at f=0Hz
    dc = f == 0
    sigma[dc] = sigma_n * (1e10 - 1j * 1e10)

    idx = np.flatnonzero(~dc)
    for i in range(0, len(idx), _CHUNK):
        chunk = idx[i:i + _CHUNK]
        sigma[chunk] = sigma_n * _reduced_conductance(f[chunk], T, Vgap)

    return sigma


def _reduced_conductance(f, T, Vgap):
    """Calculate sigma / sigma_n for an array of positive frequencies.

    The three integrals of the Mattis-Bardeen paper are evaluated with the
    tanh-sinh rule above, all frequencies at once (one row per frequency).
    Each integrand is written in terms of the distances to the endpoints of
    its interval (s_a = E - a, s_b = b - E), so the vanishing factors under
    the square roots are computed without cancellation.

    """

    delta = Vgap / 2   # binding energy
    hbar, kb = _hbar(), _kb()
    hw = (hbar * 2 * np.pi * f)[:, None]  # photon energy (column)
    above = hw[:, 0] > 2 * delta          # above the gap frequency?

    def _one_minus_2fermi(e):
        # 1 - 2 f(E) = tanh(E / 2kT), evaluated without cancellation
        if T == 0:
            return np.sign(e)
        return np.tanh(e / (2 * kb * T))

    # Thermal contribution ---------------------------------------------------
    # Contribution from thermally excited quasi-particles:
    #   2/hw int_delta^inf [f(E) - f(E+hw)] g(E) dE
    # with g(E) = (E^2 + delta^2 + hw*E) /
    #             (sqrt(E^2 - delta^2) sqrt((E+hw)^2 - delta^2)).
    # The upper limit is truncated at delta + 60kT, where the Fermi factors
    # have underflowed. With s_a = E - delta the denominator is
    # sqrt(s_a (s_a + 2 delta) (s_a + hw) (s_a + hw + 2 delta)): singular at
    # s_a = 0 and, when hw << delta, nearly singular at s_a = -hw.

    if T == 0:
        # No thermally excited quasi-particles at T=0
        thermal = 0.
    else:
        kT = kb * T
        length = 60 * kT
        s_a = length * _TS_FROM_A
        e = delta + s_a
        # f(E) - f(E+hw) = f(E) [1 - f(E+hw)] [1 - exp(-hw/kT)], which keeps
        # full relative precision deep in the Fermi tails and for hw << kT
        df = expit(-e / kT) * expit((e + hw) / kT) * -np.expm1(-hw / kT)
        g = df * (e**2 + delta**2 + hw*e) / \
            np.sqrt(s_a * (s_a + 2*delta) * (s_a + hw) * (s_a + hw + 2*delta))
        thermal = 2 / hw[:, 0] * length * (g @ _TS_WEIGHT)

    # Radiation contribution -------------------------------------------------
    # Contribution from quasi-particles generated by radiation (f > fgap):
    #   1/hw int_{delta-hw}^{-delta} [1 - 2f(E+hw)] |g(E)| dE
    # With s_a = E - (delta - hw) and s_b = -delta - E the denominator is
    # sqrt(s_b (s_b + 2 delta) s_a (s_a + 2 delta)).

    radiation = np.zeros(len(f))
    if above.any():
        hwa = hw[above]
        length = hwa - 2 * delta
        s_a, s_b = length * _TS_FROM_A, length * _TS_FROM_B
        e = delta - hwa + s_a
        g = _one_minus_2fermi(e + hwa) * np.abs(e**2 + delta**2 + hwa*e) / \
            np.sqrt(s_a * (s_a + 2*delta) * s_b * (s_b + 2*delta))
        radiation[above] = 1 / hwa[:, 0] * length[:, 0] * (g @ _TS_WEIGHT)

    # Cooper pair contribution -----------------------------------------------
    #   1/hw int_a^delta [1 - 2f(E+hw)] (E^2 + delta^2 + hw*E) /
    #       (sqrt(delta^2 - E^2) sqrt((E+hw)^2 - delta^2)) dE
    # with a = max(delta - hw, -delta). With s_a = E - a and s_b = delta - E,
    # both below and above the gap the denominator is
    # sqrt(s_b s_a (s_a + eta) (E + hw + delta)), eta = |hw - 2 delta|, which
    # is nearly singular at the lower endpoint when hw is close to 2 delta.

    a = np.maximum(delta - hw, -delta)
    length = delta - a
    eta = np.abs(hw - 2 * delta)
    s_a, s_b = length * _TS_FROM_A, length * _TS_FROM_B
    e = a + s_a
    g = _one_minus_2fermi(e + hw) * (e**2 + delta**2 + hw*e) / \
        np.sqrt(s_b * s_a * (s_a + eta) * (e + hw + delta))
    cooper = 1 / hw[:, 0] * length[:, 0] * (g @ _TS_WEIGHT)

    return thermal + radiation - 1j * cooper


# Surface Impedance ----------------------------------------------------------

def surface_impedance(f, d, T, Vgap=None, method='MB', z_back=None, **kw):
    """Calculate the surface impedance of a superconductor using Mattis-
    Bardeen theory.

    Args:
        f (float or array-like): frequency, in units [Hz]
        d (float): superconductor thickness, in units [m]
        T (float): temperature, in units [K]
        Vgap (float): gap voltage at temperature T, in units [V]. If None,
            it is calculated from Vgap0 and Tc using BCS theory (see
            gap_voltage). Zero (or T >= Tc) is the normal state.
        method (str): 'MB' for Mattis-Bardeen with the thin-film formula
            of Belitsky & Kollberg (Eqn. 5 in [2], i.e., no fields behind
            the film and no displacement current), 'film' for
            Mattis-Bardeen with the transmission-line model of the film
            (see film_impedance), or 'simple' for the simplified surface
            impedance from Kerr (1996).
        z_back (complex): with method='film', the impedance behind the
            film, in units [ohms]; default is the impedance of free space

    Keyword arguments:
        Tc (float): critical temperature, in units [K]
        Vgap0 (float): gap voltage at T=0K, in units [V]
        sigma_n (float): normal-state conductance, in units [1/ohm*m]
        lambda0 (float): penetration depth at T=0K, in units [m]. Only used
            by method='simple'; the Mattis-Bardeen methods use the
            penetration depth implied by sigma_n and Vgap0 (see
            penetration_depth).

    Returns:
        surface impedance (complex, or complex ndarray with the shape of f)

    """

    # Import keyword arguments (use defaults if not specified)
    tmp = deepcopy(PARAM)
    tmp.update(kw)
    kw = tmp

    method = method.lower()
    if method not in ('mb', 'film', 'simple'):
        raise ValueError(f"Method not recognized: {method!r} (use 'MB', 'film' or 'simple')")
    if z_back is not None and method != 'film':
        raise ValueError("z_back is only used with method='film'")

    # Using Mattis-Bardeen theory...
    if method == 'mb':
        Vgap = _resolve_gap(Vgap, T, kw)
        f = np.asarray(f, dtype=float)
        zs = _surface_impedance(f.ravel(), d, T, Vgap, kw['sigma_n']).reshape(f.shape)
        return complex(zs) if zs.ndim == 0 else zs

    # Using Mattis-Bardeen theory and the transmission-line model of the film...
    elif method == 'film':
        Vgap = _resolve_gap(Vgap, T, kw)
        f = np.asarray(f, dtype=float)
        zs = np.asarray(film_impedance(f, [(conductance(f, T, Vgap, **kw), d)], z_back))
        if Vgap > 0:
            # Superconductor at f=0: exactly zero, as with method='MB'
            zs = np.where(f == 0, 0j, zs)
        return complex(zs) if zs.ndim == 0 else zs

    # Using "simple" method from [1]...
    else:
        if T >= kw['Tc']:
            raise ValueError("method='simple' needs T < Tc")
        dl = _lambda(T, **kw)
        return _surface_impedance_simple(f, d, dl)


def _surface_impedance(f, d, T, Vgap, sigma_n):
    """Calculate the surface impedance of a superconductor.

    Args:
        f (ndarray): 1-D array of frequencies, in units [Hz]
        d (float): superconductor thickness, in units [m]
        T (float): temperature, in units [K]
        Vgap (float): gap voltage, in units [V]
        sigma_n (float): normal-state conductance, in units [1/ohm*m]

    Returns:
        surface impedance (complex ndarray)

    """

    # Angular frequency
    w = 2 * np.pi * f

    # Conductance (using Mattis-Bardeen theory)
    sigma = _conductance(f, T, Vgap, sigma_n)

    # Surface impedance (Eqn. 5 in [2]); at f=0 this is 0/0, but the limit
    # for a superconductor is zero, and for a normal metal the DC sheet
    # resistance
    zs = np.zeros(f.shape, dtype=complex)
    if Vgap == 0:
        zs[:] = 1 / (sigma_n * d)
    ac = f > 0
    zs[ac] = (np.sqrt(1j * w[ac] * sc.mu_0 / sigma[ac])
              / np.tanh(np.sqrt(1j * w[ac] * sc.mu_0 * sigma[ac]) * d))

    return zs


def _surface_impedance_simple(f, d, dl):
    """Calculate the surface impedance of a superconductor using the simple
    model presented in [1]. This techniques assumes that the frequency much
    lower than the gap frequency.

    Args:
        f (ndarray): frequency, in units [Hz]
        d (float): superconductor thickness, in units [m]
        dl (float): London penetration depth

    Returns:
        surfance impedance

    """

    # Angular frequency
    w = 2 * np.pi * f

    # Surface impedance (see Appendix A5 in [1])
    zs = 1j * w * sc.mu_0 * dl / np.tanh(d / dl)

    return zs


# Films of finite thickness --------------------------------------------------

def film_impedance(f, layers, z_back=None):
    """Surface impedance of a film, or a stack of films, of finite thickness.

    Each layer is treated as a section of transmission line, with the wave
    impedance and wavenumber of a plane wave in the metal (local
    conductivity, including the displacement current). The back of the
    stack is terminated by ``z_back``. This models the fields behind the
    film, and it only matters for films thinner than a few penetration
    depths. With ``z_back=np.inf`` (no fields behind the film) and a
    negligible displacement current, a single layer gives the thin-film
    formula of surface_impedance (method='MB').

    Normal metals have a constant conductivity, e.g., ``[(1 / rho, d)]``.
    For a stack of superconductors, use conductance for each layer, e.g., a
    Nb film on NbTiN: ``[(conductance(f, T, **nb), d_nb), (conductance(f,
    T, **nbtin), d_nbtin)]``.

    At f=0 the layers are sheet conductances (sigma * d) in parallel with
    the backing, 1 / (sum(sigma * d) + 1 / z_back): a normal-metal film on
    free space gives its sheet resistance in parallel with 377 ohms, and a
    superconductor (with the huge f=0 value from conductance) gives ~0.

    Args:
        f (float or array-like): frequency, in units [Hz] (non-negative)
        layers (list): ``(sigma, d)`` pairs, starting from the layer that
            carries the current (the surface facing the dielectric) and
            moving away from it. ``sigma`` is the complex conductivity, in
            units [1/ohm*m] (scalar, or one value per frequency, with the
            shape of f), and ``d`` the thickness, in units [m].
        z_back (complex): impedance behind the last layer, in units [ohms]
            (scalar or one value per frequency; np.inf for no fields behind
            the film); default is the impedance of free space

    Returns:
        surface impedance, in units [ohms] (complex, or complex ndarray with
        the shape of f)

    >>> zs = film_impedance(100e9, [(1 / 2e-8, 1e-6)])  # 1 um of copper
    >>> round(zs.real, 4) == round(zs.imag, 4)  # thick film: R = X
    True

    """

    z_vac = _z_vacuum()
    f = np.asarray(f, dtype=float)
    if np.any(f < 0):
        raise ValueError("Frequency must be non-negative.")
    layers = [(np.broadcast_to(np.asarray(sigma, dtype=complex), f.shape), d) for sigma, d in layers]
    if len(layers) == 0:
        raise ValueError("Need at least one layer.")
    z_back = np.broadcast_to(np.asarray(z_vac if z_back is None else z_back, dtype=complex), f.shape)

    ac, dc = f > 0, f == 0
    zs = np.empty(f.shape, dtype=complex)

    # Transmission-line model, from the back of the stack to the front
    k0 = 2 * np.pi * f[ac] / sc.c
    z_in = z_back[ac].copy()
    for sigma, d in reversed(layers):
        factor = np.sqrt(1. - 1j * z_vac * sigma[ac] / k0 + 0j)
        factor = np.where(factor.imag > 0, -factor, factor)  # waves decay into the metal
        k1 = k0 * factor
        z1 = z_vac / factor
        with np.errstate(invalid='ignore'):
            refl = np.where(np.isinf(z_in), 1., (z_in - z1) / (z_in + z1))
        gamma = refl * np.exp(-2j * k1 * d)
        z_in = z1 * (1 + gamma) / (1 - gamma)
    zs[ac] = z_in

    # DC: sheet conductances in parallel with the backing
    if dc.any():
        with np.errstate(divide='ignore'):
            g_back = np.where(z_back[dc] == 0, np.inf, 1 / np.where(z_back[dc] == 0, 1., z_back[dc]))
        g = g_back + sum(sigma[dc] * d for sigma, d in layers)
        zs[dc] = np.where(np.isinf(g), 0j, 1 / np.where(np.isinf(g), 1., g))

    return complex(zs) if zs.ndim == 0 else zs


# Penetration depths ---------------------------------------------------------

def penetration_depth(T, Vgap=None, **kw):
    """London penetration depth from Mattis-Bardeen theory (local limit).

    This is the low-frequency limit of the conductance,
    ``lambda = sqrt(hbar / (pi mu0 sigma_n Delta tanh(Delta / 2 k_B T)))``
    with ``Delta = e Vgap / 2``, i.e., with the BCS temperature dependence
    (not the two-fluid approximation used by method='simple').

    Args:
        T (float): temperature, in units [K]
        Vgap (float): gap voltage at temperature T, in units [V]. If None,
            it is calculated from Vgap0 and Tc using BCS theory (see
            gap_voltage).

    Keyword arguments:
        Tc (float): critical temperature, in units [K]
        Vgap0 (float): gap voltage at T=0K, in units [V]
        sigma_n (float): normal-state conductance, in units [1/ohm*m]

    Returns:
        penetration depth, in units [m] (inf in the normal state)

    """

    tmp = deepcopy(PARAM)
    tmp.update(kw)
    kw = tmp
    Vgap = _resolve_gap(Vgap, T, kw)
    if Vgap == 0:
        return np.inf
    delta = Vgap / 2
    tanh = np.tanh(delta / (2 * _kb() * T)) if T > 0 else 1.

    return float(np.sqrt(_hbar() / (np.pi * sc.mu_0 * kw['sigma_n'] * delta * tanh)))


def efield_penetration(zs, f):
    """Calculate the electric field penetration depth.

    Args:
        zs (ndarray): surface impedance
        f (ndarray): frequency, in units [Hz]

    Returns:
        E-field penetration depth

    """

    # Angular frequency
    w = 2 * np.pi * f

    return zs.imag / (w * sc.mu_0)


def hfield_penetration(zs, f):
    """Calculate the magnetic field penetration depth.

    Args:
        zs (ndarray): surface impedance
        f (ndarray): frequency, in units [Hz]

    Returns:
        H-field penetration depth

    """

    # Angular frequency
    w = 2 * np.pi * f

    return zs.real / (w * sc.mu_0)


# Lambda ---------------------------------------------------------------------

def _lambda0(Vgap0=None, sigma_n=None, **kw):
    """Penetration depth at T=0K from Mattis-Bardeen theory (local limit),
    calculated from Vgap0 and sigma_n (the lambda0 parameter is not used;
    see penetration_depth for other temperatures)."""

    delta = Vgap0 / 2

    return (_hbar() / np.pi / sc.mu_0 / sigma_n / delta) ** 0.5


def _lambda(T, **kw):
    """Penetration depth at temperature T for method='simple': the lambda0
    parameter with the two-fluid temperature dependence, 1/sqrt(1 - t^4)."""

    t = T / kw['Tc']

    return kw['lambda0'] / np.sqrt(1 - t ** 4)


# Fermi function -------------------------------------------------------------

def fermi(energy, T):
    """Fermi function.

    Args:
        energy (ndarray): energy in units [eV]
        T (float): temperature in units [K]

    Returns:
        fermi function

    """

    if T != 0.:
        # expit keeps full relative precision deep in the tail
        return expit(-energy / (_kb() * T))
    else:
        return np.heaviside(-energy, 0.5)
