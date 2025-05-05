"""
Base class for the structure of a crystal.
"""

import numpy as np
import numpy.typing as npt


class Structure:
    """
    Base class for calculating the structure factor of a crystal.

    Parameters
    ----------
    energies : npt.ArrayLike
        Array of energies (in eV) for which the scattering factors are defined.
    scattering_factors : dict[int, tuple[npt.ArrayLike, npt.ArrayLike]]
        Dictionary containing the scattering factors for each atom type.
        The keys are the atomic numbers, and the values are either complex
        arrays, or tuples of two arrays (f1, f2) where f1 is the real part,
        f2 is the imaginary part.
    atomic_basis : dict[int, list[tuple[float, float, float]]]
        Dictionary containing the positions of atoms in the atomic basis,
        for each relevant atomic number. The keys are the atomic numbers,
        and the values are lists of tuples (x, y, z) representing the
        fractional positions of the atoms in the unit cell (lattice).
    """

    def __init__(
        self,
        energies: npt.ArrayLike,
        scattering_factors: dict[
            int, tuple[npt.ArrayLike, npt.ArrayLike] | npt.NDArray[np.complex128]
        ],
        atomic_basis: dict[int, list[tuple[float, float, float]] | npt.ArrayLike],
    ) -> None:
        energies = np.asarray(energies, dtype=np.float64)
        # Convert the scattering factors to complex numbers if they are tuples.
        new_factors: dict[int, npt.NDArray[np.complex128]] = {}
        for key, value in scattering_factors.items():
            if isinstance(value, tuple):
                new_factors[key] = np.array(
                    value[0], dtype=np.complex128
                ) + 1j * np.array(value[1], dtype=np.complex128)
            else:
                new_factors[key] = np.array(value, dtype=np.complex128)
        # Convert the atomic basis to a numpy array if it is a list of tuples.
        new_basis: dict[int, npt.NDArray[np.float64]] = {}
        for key, value in atomic_basis.items():
            if isinstance(value, list):
                new_basis[key] = np.array(value, dtype=np.float64)
            else:
                new_basis[key] = np.array(value, dtype=np.float64)
        # Initialize the attributes.
        self.energies = energies
        self.scattering_factors = new_factors
        self.atomic_basis = new_basis

    def calculate_structure_factor(self, h: int, k: int, l: int):  # noqa: E741
        """
        Calculate the structure factor for given Miller indices (h, k, l).

        The structure factor is calculated for a lattice periodicity in the
        Miller notation (h,k,l), also representing the diffraction order.

        Parameters
        ----------
        h : int
            Miller index h.
        k : int
            Miller index k.
        l : int
            Miller index l.

        Returns
        -------
        complex
            The total structure factor for the given Miller index.
        """
        total_structure_factors: npt.NDArray[np.complex128] = np.zeros_like(
            self.energies, dtype=np.complex128
        )
        for element, positions in self.atomic_basis.items():
            # Get the scattering factor for the element.
            f = self.scattering_factors[element]
            # Calculate the phase factor for each atom in the basis.
            phase_factors: npt.NDArray[np.complex128] = np.exp(
                2j * np.pi * np.array([np.dot([h, k, l], pos) for pos in positions])
            )
            # Sum the contributions from all atoms in the basis.
            total_structure_factors += f * np.sum(phase_factors)
        return total_structure_factors

    def calculate_structure_factor_intensity(
        self, h: int, k: int, l: int  # noqa: E741
    ):
        """
        The structure factor intensity for given Miller indices (h, k, l).

        Parameters
        ----------
        h : int
            Miller index h.
        k : int
            Miller index k.
        l : int
            Miller index l.

        Returns
        -------
        float
            The intensity of the structure factor for the given Miller indices.
        """
        structure_factor = self.calculate_structure_factor(h, k, l)
        return np.abs(structure_factor) ** 2
