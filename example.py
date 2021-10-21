import mattisbardeen as mb
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc


# Optional
plt.style.use(['science', 'notebook'])

# Properties of the superconducting material, i.e., normal-state conductance,
# critical temperature, gap voltage at T=0K, and penetration depth
param = dict(sigma_n=1.74e7, Tc=8.1, Vgap0=2.65e-3, lambda0=86*sc.nano)

# Properties of the specific superconductor
d = 100e-9     # thickness of the superconducting film in [m]
T = 4.         # ambient temperature in [K]
Vgap = 2.8e-3  # gap voltage at temperature T in [V]

# Frequency in [Hz]
f = np.linspace(0, 1000, 201) * 1e9

# Surface impedance [ohm / sq.]
Zs = mb.surface_impedance(f, d, T, Vgap, **param)

# Plot results
fig, ax = plt.subplots()
ax.plot(f/1e9, Zs.real, label='Real')
ax.plot(f/1e9, Zs.imag, 'r', label='Imaginary')
ax.legend()
ax.set(xlabel='Frequency (GHz)', xlim=[0, 1000])
ax.set(ylabel=r'Surface impedance ($\Omega$/sq.)', ylim=[0, 1.1])
plt.savefig("example.png", dpi=600)
