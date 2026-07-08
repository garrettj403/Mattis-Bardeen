# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Run tests: `make test` (runs `py.test test_mattisbardeen.py --verbose --doctest-modules`)
- Run a single test: `python -m pytest test_mattisbardeen.py::test_belitsky1995_fig6 -v`
- Coverage: `make cov` (terminal) or `make cov-report` (HTML)
- Install for development: `pip install -e .`

## Architecture

Single-module Python package (`mattisbardeen.py`) that computes the electrical properties of superconductors from Mattis-Bardeen theory. No subpackages; `example.py` is a standalone plotting script.

### Core computation flow

`surface_impedance(f, d, T, Vgap, method='MB', **kw)` → `_conductance(f, ...)` → three numerical integrals. The public functions (`conductance`, `surface_impedance`) accept scalar or array frequency and loop over a private per-frequency scalar function (`_conductance`, `_surface_impedance`). Material parameters (Tc, Vgap0, sigma_n, lambda0) come from the module-level `PARAM` dict (niobium defaults) and are overridden via kwargs.

The complex conductivity σ = σ_n·(thermal + radiation − i·cooper) is built from three integrals of the Mattis-Bardeen paper, each with integrable 1/√x endpoint singularities. They are evaluated with `scipy.integrate.quad` **after variable substitutions that cancel the singularities analytically**:

- Thermal (σ1): E = Δ·cosh θ cancels √(E²−Δ²); upper limit truncated at Δ + 60·k_B·T where the Fermi factor underflows (do not integrate to ∞ — cosh overflows and produces NaN).
- Radiation (σ1, only f > fgap) and Cooper (σ2): E = c + r·sin θ over the interval cancels the singularities at both endpoints and maps shrinking near-gap intervals onto a fixed [−π/2, π/2] domain, so ħω ≈ 2Δ edge cases need no special guards.

If you modify these integrals, keep the integrands smooth (singularities cancelled by the substitution's Jacobian); accuracy is controlled by the `_EPSABS`/`_EPSREL` module constants.

Other conventions:
- All energies are in **eV**: the module-level `kb`, `h`, `hbar` are the SI constants divided by the electron charge.
- `f == 0` returns a hard-coded huge conductivity (superconductor DC limit).
- `fermi(energy, T)` must accept both scalars (quad passes scalars) and arrays, including T=0.
- `method='simple'` in `surface_impedance` uses the Kerr (1996) thin-film approximation instead of Mattis-Bardeen; valid only well below the gap frequency.

### Validation tests

`test_mattisbardeen.py` validates against digitized curves from the literature stored in `validation-data/` (Belitsky 1995 Fig. 6, Belitsky & Kollberg 2006 Fig. 2), loaded with `np.genfromtxt` and compared with absolute tolerances of 0.01–0.2 on σ/σ_n. Any change to the integration must keep these passing; they are the accuracy ground truth for this package.
