"""Generate high-precision reference values of the Mattis-Bardeen conductance.

The integrals are evaluated directly in their original form with mpmath at
40 significant digits (tanh-sinh quadrature with error control, split at the
near-singular points), independently of the vectorized implementation in
mattisbardeen.py. The grid covers T=0 to near Tc, very small gaps, very low
frequencies, and frequencies within 1e-4 of the gap frequency.

Requires mpmath (not a dependency of the package). Run from the repository
root:

    python validation-data/reference/generate.py

"""

import mpmath as mp
import numpy as np
import scipy.constants as sc

mp.mp.dps = 40

# Same constants as mattisbardeen.py (energies in eV)
KB = mp.mpf(sc.k) / mp.mpf(sc.e)
HBAR = mp.mpf(sc.h) / (2 * mp.pi) / mp.mpf(sc.e)

# (Vgap [V], T [K])
CASES = [(2.8e-3, 0.), (2.8e-3, 0.5), (2.8e-3, 4.), (2.8e-3, 8.5),
         (0.05e-3, 4.), (0.36e-3, 0.1)]

# Frequency in units of the gap frequency
X = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.,
     1.0001, 1.001, 1.01, 1.1, 2., 5., 50.]


def reduced_conductance(x, T, Vgap):
    """sigma / sigma_n at frequency x * Vgap / h."""

    d = mp.mpf(Vgap) / 2
    hw = 2 * d * mp.mpf(x)
    T = mp.mpf(T)

    if T > 0:
        def fermi(e):
            return 1 / (1 + mp.exp(e / (KB * T)))
    else:
        def fermi(e):
            return mp.mpf(1) if e < 0 else mp.mpf(0)

    def g(e):
        return e**2 + d**2 + hw * e

    def q(num, *factors):
        # num / sqrt(|prod factors|); zero where a node rounds onto an endpoint
        den = mp.sqrt(abs(mp.fprod(factors)))
        return num / den if den != 0 else mp.mpf(0)

    thermal = 0
    if T > 0:
        thermal = 2 / hw * mp.quad(
            lambda e: q((fermi(e) - fermi(e + hw)) * g(e), e - d, e + d, e + hw - d, e + hw + d),
            [d, d + hw, d + 2 * hw + KB * T, d + 60 * KB * T])

    radiation = 0
    if hw > 2 * d:
        radiation = 1 / hw * mp.quad(
            lambda e: q((1 - 2 * fermi(e + hw)) * abs(g(e)), -d - e, d - e, e + hw - d, e + hw + d),
            [d - hw, -d])

    a = max(d - hw, -d)
    split = a + min(abs(hw - 2 * d), (d - a) / 2)
    if split == a:
        split = a + (d - a) / 2
    cooper = 1 / hw * mp.quad(
        lambda e: q((1 - 2 * fermi(e + hw)) * g(e), d - e, d + e, e + hw - d, e + hw + d),
        [a, split, d])

    return thermal + radiation, cooper


if __name__ == '__main__':

    rows = []
    for Vgap, T in CASES:
        for x in X:
            s1, s2 = reduced_conductance(x, T, Vgap)
            rows.append((Vgap, T, x, float(s1), float(s2)))
            print(f"Vgap={Vgap} T={T} x={x}: {mp.nstr(s1, 12)} {mp.nstr(s2, 12)}")

    np.savetxt('validation-data/reference/mpmath-conductance.txt', rows, delimiter=',',
               fmt='%.17g', header='Vgap [V], T [K], f / fgap, sigma1 / sigma_n, sigma2 / sigma_n')
