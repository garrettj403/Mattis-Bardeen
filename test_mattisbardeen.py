"""Tests for the mattisbardeen.py module."""

import mattisbardeen as mb
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.constants as sc
import timeit


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

        V. Y. Belitsky and E. L. Kollberg, “Superconductor–insulator–
        superconductor tunnel strip line: Features and applications,” J. Appl.
        Phys., vol. 80, no. 8, pp. 4741–4748, Oct. 1996.

    """

    # Parameters
    T = 4.0
    Vgap = 2.53e-3  # Gap voltage is not specified in the paper!!!
    param = dict(sigma_n=1.739e7, Tc=8.1, Vgap0=2.65e-3, lambda0=85*sc.nano)

    # Calculate sigma_1
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-sigma1.txt'
    fghz, lit_sigma1 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)
    max_difference = np.abs(lit_sigma1 - sigma.real / param['sigma_n']).max()
    assert max_difference < 0.02

    # Calculate sigma_2
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-sigma2.txt'
    fghz, lit_sigma2 = np.genfromtxt(filename, delimiter=',').T
    sigma = mb.conductance(fghz * 1e9, T, Vgap, **param)
    max_difference = np.abs(lit_sigma2 + sigma.imag / param['sigma_n']).max()
    assert max_difference < 0.2

    # Calculate E-field penetration
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-lambda.txt'
    fghz, lit_lambda = np.genfromtxt(filename, delimiter=',').T
    d = 500e-9  # The thickness is not specified in the paper!!!
    zs = mb.surface_impedance(fghz * 1e9, d, T, Vgap, **param)
    _lambda = mb.efield_penetration(zs, fghz * 1e9) / mb._lambda0(**param)
    max_difference = np.abs(_lambda - lit_lambda).max()
    assert max_difference < 0.1

    # Calculate H-field penetration
    filename = 'validation-data/belitsky2006/belitsky2006-fig2-delta.txt'
    fghz, lit_delta = np.genfromtxt(filename, delimiter=',').T
    d = 500e-9  # The thickness is not specified in the paper!!!
    zs = mb.surface_impedance(fghz * 1e9, d, T, Vgap, **param)
    _delta = mb.hfield_penetration(zs, fghz * 1e9) / mb._lambda0(**param)
    max_difference = np.abs(_delta - lit_delta).max()
    assert max_difference < 0.1



def test_compare_simple_and_MB_models():

    # Frequency
    f = np.arange(100, 1000, 100) * 1e9

    # Surface impedance (using both models)
    zs_simple = mb.surface_impedance(f, 250e-9, 4., 2.75e-3, method='simple')
    zs_mb     = mb.surface_impedance(f, 250e-9, 4., 2.75e-3, method='MB')

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


if __name__ == "__main__":

    test_fermi()

    test_belitsky1995_fig6()
    test_belitsky2006_fig2()

    test_compare_simple_and_MB_models()
