from math import pi
from dataclasses import dataclass
from tqdm import tqdm
from numba import jit
import numpy as np


G = 6.6743 * 10 ** (-11)  # N m^2 kg^-2
solar_mass = 2 * 10**30  # kg
h = 6.6 * 10 ** (-34)  # Js
c = 3.0 * 10**8  # m/s
m_e = 9.1093837 * 10 ** (-31)  # kg
m_n = 1.675e-27  # kg

# Electron density where P_NR = P_ER
n_e_transition = ((c * 5 * m_e / 4) ** 3) * (8 * pi / (3 * h**3))


@jit(nopython=True)
def mass_continuity(r: float, rho: float) -> float:
    return 4 * pi * r**2 * rho


@jit(nopython=True)
def hydrostatic_eq(M_r: float, r: float, rho: float) -> float:
    return G * M_r * rho / r**2


@jit(nopython=True)
def P_ER(n_e: float) -> float:
    """
    Returns the pressure for a specific electron density in the extremely-
    relativistic case.
    """
    return (c / 4) * (3 * h**3 / (8 * pi)) ** (1 / 3) * n_e ** (4 / 3)


@jit(nopython=True)
def P_ER_inv(P_ER: float) -> float:
    """
    Returns the electron density for a specific pressure in the extremely-
    relativistic case.
    """
    return (P_ER * ((c / 4) * (3 * h**3 / (8 * pi)) ** (1 / 3)) ** (-1)) ** (3 / 4)


@jit(nopython=True)
def P_NR(n_e: float) -> float:
    """
    Returns the pressure for a specific electron density in the not-
    relativistic case.
    """
    return (1 / (5 * m_e)) * (3 * h**3 / (8 * pi)) ** (2 / 3) * n_e ** (5 / 3)


@jit(nopython=True)
def P_NR_inv(P_NR: float) -> float:
    """
    Returns the electron density for a specific pressure in the not-
    relativistic case.
    """
    return (P_NR * ((1 / (5 * m_e)) * (3 * h**3 / (8 * pi)) ** (2 / 3)) ** (-1)) ** (
        3 / 5
    )


@dataclass
class StellarStructure:
    def _check_case(self, n_e: float):
        """
        Check which case should be used, not-relativistic or extremely-
        relativistic, depending on the electron density
        """
        if n_e > n_e_transition:
            self.calc_n_e = P_ER_inv
            self.calc_P = P_ER
        else:
            self.calc_n_e = P_NR_inv
            self.calc_P = P_NR

    def calc_mass_radius(self, dr: int, rho_0: float) -> tuple[float, float]:
        """Returns the mass and radius of the star."""

        # Every electron has a coresponding neutron and proton, where m_n ~ m_p
        # Electron mass can be neglected as m_e << m_n.
        n_e = rho_0 / (2 * m_n)
        self._check_case(n_e)
        P = self.calc_P(n_e)

        rho = rho_0
        M_r, r = 0, 0

        while P > 0:
            r += dr

            # Mass increases with the number of shells.
            M_r += dr * mass_continuity(r, rho)

            # Pressure, electron density and mass density decrease while considering
            # shells with increasing radius.
            P -= dr * hydrostatic_eq(M_r, r, rho)
            n_e = self.calc_n_e(P)
            rho = 2 * m_n * n_e

            self._check_case(n_e)

        return M_r / solar_mass, r / 1000

    def sweep_rho_0(
        self, a: float, b: float, step: float, dr: int
    ) -> tuple[list, list]:
        """
        Returns a list of tuples of masses and radii for rho_0 in a range from a to b
        """
        return [
            self.calc_mass_radius(dr, 10**power)
            for power in tqdm(np.arange(a, b, step))
        ]
