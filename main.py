from stellar_structure import StellarStructure

import matplotlib.pyplot as plt


# k = (c / 4) * (((3 * h**3) / (8 * pi)) ** (1 / 3)) * ((1 / m_e) ** (4 / 3))
stellar_structure = StellarStructure(k=1200000, power=4 / 3)

mass, radius = stellar_structure.calc_mass_radius(dr=10, rho_0=10**12)
print(f"M = {mass} Msun\nR = {radius} km")

masses_radii = stellar_structure.sweep_rho_0(
    a=10**10, b=10**13, step=10**11, dr=10
)

masses_radii = list(zip(*masses_radii))

plt.xlabel("mass (Msun)")
plt.ylabel("R (km)")
plt.savefig("Mass-Radius")
plt.plot(masses_radii[0], masses_radii[1])
plt.savefig("figs/Mass-Radius")
