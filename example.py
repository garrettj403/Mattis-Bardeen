import mattisbardeen as mb
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

import scienceplots  # noqa: F401 -- registers the 'science' style
plt.style.use(['science'])

# Properties of the superconducting material, i.e., normal-state conductance,
# critical temperature, gap voltage at T=0K, and penetration depth
param = dict(sigma_n=1.74e7, Tc=8.1, Vgap0=2.65e-3, lambda0=86*sc.nano)

# Properties of the specific superconductor
d = 100e-9     # thickness of the superconducting film in [m]
T = 4.         # ambient temperature in [K]
Vgap = 2.8e-3  # gap voltage at temperature T in [V]

# Frequency in [Hz]
f = np.linspace(0, 4000, 201) * 1e9

# Surface impedance [ohm / sq.]
Zs = mb.surface_impedance(f, d, T, Vgap, **param)

# Plot results
fig, ax = plt.subplots(figsize=(4, 3))
ln, = ax.plot(f/1e9, Zs.real, label='Real')
ax.plot(f/1e9, np.sqrt(sc.mu_0 * (2 * np.pi * f) / 2 / 1.74e7), c=ln.get_color(), ls='--', label='Normal state')
ax.plot(f/1e9, Zs.imag, 'r', label='Imaginary')
ax.plot(f/1e9, sc.mu_0 * (86*sc.nano) / np.tanh(d / 86 / sc.nano) * (2 * np.pi * f), 'r--', label="Surface inductance")
ax.legend()
ax.set(xlabel='Frequency (GHz)', xlim=[0, 4000])
ax.set(ylabel=r'Surface impedance ($\Omega$/sq.)', ylim=[0, 1.1])
# plt.show()
fig.savefig("example.png", dpi=400)
