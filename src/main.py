from stellar_structure import StellarStructure

import matplotlib.pyplot as plt
import scienceplots


stellar_structure = StellarStructure()
mass, radius = stellar_structure.calc_mass_radius(dr=100, rho_0=10**12)
print(f"M = {mass} Msun\nR = {radius} km")


masses_radii = stellar_structure.sweep_rho_0(a=7, b=12.5, step=0.5, dr=1)
masses_radii = list(zip(*masses_radii))

plt.style.use(["science", "ieee"])
plt.title("Mass-Radius relationship for white dwarfs")
plt.xlabel("Mass ($M / M_{\odot}$)")
plt.ylabel("Radius (km)")
plt.ylim(1000, 25000)
plt.xlim(0, 1.5)
plt.plot(masses_radii[0], masses_radii[1], "--o", linewidth=0.5, markersize=3)
plt.savefig("../figs/Mass-Radius")
