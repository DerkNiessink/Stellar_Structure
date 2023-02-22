from math import pi
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit


G = 6.6743 * 10 ** (-11)
solar_mass = 2 * 10**30
h = 6.6 * 10 ** (-34)
c = 3.0 * 10**8
m_e = 9.1093837 * 10 ** (-31)
n_e_transition = ((c * 5 * m_e / 4) ** 3) * (8 * pi / (3 * h**3))


@jit(nopython=True)
def mass_continuity(r: float, rho: float) -> float:
    return 4 * pi * r**2 * rho


@jit(nopython=True)
def hydrostatic_eq(M_r: float, r: float, rho: float) -> float:
    return -G * M_r * rho / r**2


@dataclass
class StellarStructure:
    k: float
    power: float

    def EoS(self, rho: float) -> float:
        """Returns the pressure"""
        return self.k * rho**self.power

    def inv_EoS(self, P: float) -> float:
        """Returns the density"""
        return (P / self.k) ** (1 / self.power)

    def calc_mass_radius(self, dr: int, rho_0: float) -> tuple[float, float]:
        """Returns the mass and radius of the star."""

        P = self.EoS(rho_0)
        rho = rho_0
        M_r, r = 0, 0

        while P > 0:
            r += dr
            M_r += dr * mass_continuity(r, rho)
            P += dr * hydrostatic_eq(M_r, r, rho)
            rho = self.inv_EoS(P)

        return M_r / solar_mass, r / 1000

    def sweep_rho_0(
        self, a: float, b: float, step: float, dr: int
    ) -> tuple[list, list]:
        """
        Returns a list of tuples of masses and radii for rho_0 in a range from a to b
        """
        return [
            self.calc_mass_radius(dr, rho_0)
            for rho_0 in tqdm(range(int(a), int(b), step))
        ]


if __name__ == "__main__":

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
    plt.savefig("Mass-Radius")
