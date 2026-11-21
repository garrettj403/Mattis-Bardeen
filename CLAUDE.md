# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Run tests: `make test` or `pytest` (test paths and `--doctest-modules` are set in `pyproject.toml`, so module docstring examples are tested too)
- Run a single test: `python -m pytest test_mattisbardeen.py::test_belitsky1995_fig6 -v`
- Coverage: `make cov` (terminal) or `make cov-report` (HTML)
- Lint: `flake8 .` (config in `.flake8`: `max-line-length = 120`, excludes `.ipynb_checkpoints`)
- Install for development: `pip install -e ".[dev]"`
- Build and check a release: `make build` (`python -m build` + `twine check`); upload with `make upload-test` (TestPyPI) or `make upload` (PyPI). Bump `version` in `pyproject.toml` first — PyPI never accepts the same version twice.

## CI

Two GitHub Actions workflows run on every push and pull request (`.github/workflows/`):

- `ci.yml` — runs `pytest -v` on ubuntu, windows, and macOS (Python 3.12, via conda).
- `linter.yml` — runs `flake8 .`; the build fails on any lint error, so keep the tree flake8-clean before pushing.

Both must be green. The `flake8` config in `.flake8` is the source of truth for style; match it rather than reformatting to a different convention.

## Architecture

Single-module Python package (`mattisbardeen.py`) that computes the electrical properties of superconductors from Mattis-Bardeen theory. No subpackages; `example.py` is a standalone plotting script.

Packaging is in `pyproject.toml` (setuptools backend; there is no `setup.py`). The PyPI distribution name is `mattis-bardeen`; the import name is `mattisbardeen`. Because the code is a single top-level module (not a package directory), it is shipped via `[tool.setuptools] py-modules = ["mattisbardeen"]` — do not switch to package auto-discovery, which would silently package no code. Keep `py-modules` in sync if the module is ever renamed or split.

### Core computation flow

`surface_impedance(f, d, T, Vgap=None, method='MB', **kw)` → `_surface_impedance` → `_conductance` (splits frequencies into chunks of `_CHUNK`) → `_reduced_conductance` (σ/σ_n for a 1-D array of positive frequencies, fully vectorized). The public functions accept a scalar, list, or array of any shape and return a Python `complex` for scalar input, otherwise an array of the same shape. Material parameters (Tc, Vgap0, sigma_n, lambda0) come from the module-level `PARAM` dict (niobium defaults) and are overridden via kwargs.

The complex conductivity σ = σ_n·(thermal + radiation − i·cooper) is built from three integrals of the Mattis-Bardeen paper. They have 1/√ endpoint singularities, and **near-singular peaks just inside an endpoint** (thermal at ħω ≪ Δ; Cooper at ħω ≈ 2Δ). They are evaluated with a fixed **tanh-sinh rule** (`_tanh_sinh_rule`, 129 nodes, built once at import), whose nodes cluster double-exponentially at the endpoints and handle both, applied to all frequencies at once as a matrix product. No adaptive `quad` is used here (it was 20× slower, and its absolute tolerance silently failed for the tiny σ1 at low T).

- Integrands are written in terms of the endpoint distances `s_a = E − a`, `s_b = b − E`, which the rule supplies directly (`_TS_FROM_A/_TS_FROM_B` via `expit`), so the vanishing factors under the square roots never suffer cancellation. Keep that form if you edit them.
- Thermal: upper limit truncated at Δ + 60·k_B·T where the Fermi factors underflow. f(E) − f(E+ħω) is computed as f(E)·[1−f(E+ħω)]·(−expm1(−ħω/kT)) to keep relative precision in the Fermi tails and for ħω ≪ kT.
- Accuracy is fixed by the rule (no tolerance knob): ~1e-11 relative vs. the mpmath reference. `_EPSABS`/`_EPSREL` now only apply to the BCS gap solver, which still uses `quad`.

Other conventions:
- All energies are in **eV**: the module-level `kb`, `h`, `hbar` are the SI constants divided by the electron charge.
- `Vgap` is optional in `conductance`/`surface_impedance`/`penetration_depth`: if `None`, `gap_voltage(T, Vgap0, Tc)` computes it by solving the weak-coupling BCS gap equation for Δ(T)/Δ(0) vs. T/Tc and scaling by the measured `Vgap0` (not 3.53·k_B·Tc). T ≥ Tc gives zero gap, which is the normal state: `conductance` returns `sigma_n`, `surface_impedance` the normal-metal impedance, `penetration_depth` inf; `method='simple'` raises `ValueError` there. A negative `Vgap` raises.
- `f == 0` gives a hard-coded huge conductivity (superconductor DC limit) and Zs = 0 (the formula would give 0/0 = NaN; in the normal state, Zs = 1/(sigma_n d)). Negative frequencies raise `ValueError`.
- Physical constants are read from `sc` (scipy.constants) at call time (`_kb`, `_hbar`, `_z_vacuum`), never from the module-level `kb`/`h`/`hbar`, so callers (e.g., QMix's SuperMix comparison) can monkeypatch `mattisbardeen.sc` with another CODATA set. Use `np.pi`, not `sc.pi` (the replacement namespaces don't have it).

### Films of finite thickness

- `surface_impedance(method='MB')` is Belitsky & Kollberg's thin-film formula `sqrt(jωμ/σ) coth(kd)`: no fields behind the film (equivalent to `z_back = inf`) and no displacement current. Keep it as the default (existing results).
- `film_impedance(f, layers, z_back=None)` is the transmission-line model from SuperMix (via QMix, which checks it against SuperMix to round-off): each layer a line section with `k = k0 sqrt(1 - j Z0 σ / k0)` (includes the displacement current), layers listed from the current-carrying side, backing `z_back` (default free space at call time; `np.inf` supported). At f=0 it uses the DC limit, sheet conductances in parallel with the backing, `1/(Σσd + 1/z_back)`; `surface_impedance(method='film')` returns exactly 0 there for a superconductor, matching `'MB'`. Stacks and normal metals (`(1/rho, d)`) go through it; `surface_impedance(method='film', z_back=...)` is the single-layer Mattis-Bardeen case. For a 50 nm Nb film, 'film' and 'MB' differ by 3e-4 (100 GHz) to 1% (1 THz).
- `surface_impedance` validates `method` (case-insensitive `'MB'`, `'film'`, `'simple'`) before anything else, so an unknown method is always reported as such.
- `penetration_depth(T, Vgap=None, **kw)` is the BCS local-limit λ (`Δ tanh(Δ/2kT)`); `_lambda` (used by `method='simple'`) is the `lambda0` parameter with the two-fluid `1/sqrt(1 - t^4)` law (before, it ignored `lambda0` and used `_lambda0`); `_lambda0` is the local-limit value computed from `sigma_n` and `Vgap0` (used to normalize the Belitsky 2006 penetration depths in the tests).

QMix (`qmix.superconductor`) depends on `conductance`, `gap_voltage`, `film_impedance` and `penetration_depth`; keep their signatures and accuracy stable.
- `fermi(energy, T)` uses `expit` (accurate deep in the tail) and must accept scalars and arrays, including T=0.
- `method='simple'` in `surface_impedance` uses the Kerr (1996) thin-film approximation instead of Mattis-Bardeen; valid only well below the gap frequency.

### Validation tests

`test_mattisbardeen.py` checks the conductance against 40-digit mpmath values of the original integrals (`validation-data/reference/`, rtol 1e-9; regenerate with `python validation-data/reference/generate.py`, needs `mpmath`), the T=0 elliptic-integral closed forms, the low-T analytic limit, and the BCS gap against its near-Tc and low-T limits. It also validates against digitized curves from the literature stored in `validation-data/` (Belitsky 1995 Fig. 6, Belitsky & Kollberg 2006 Fig. 2), loaded with `np.genfromtxt` and compared with absolute tolerances of 0.01–0.2 on σ/σ_n. Any change to the integration must keep these passing; they are the accuracy ground truth for this package.
