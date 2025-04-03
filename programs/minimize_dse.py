import numpy as np
from scipy.optimize import minimize

from proteusPy import Disulfide, Load_PDB_SS


def minimize_standard_energy(initial_conformation=[-60, -60, -90, -60, -60]):
    """
    Minimizes the standard energy (kcal/mol) of a Disulfide object using the Nelder-Mead optimization method.

    Parameters:
        initial_conformation (list): Starting dihedral angles [chi1, chi2, chi3, chi4, chi5]
                                    Default: [-60, -60, -90, -60, -60]

    Returns:
        Disulfide: The minimized Disulfide object with optimal conformation
    """

    def objective_function(x):
        """Calculate standard energy for given dihedral angles"""
        temp_ss = Disulfide("temp")
        temp_ss.dihedrals = x
        return temp_ss.energy

    # Run the minimization
    result = minimize(objective_function, initial_conformation, method="Nelder-Mead")

    # Create a Disulfide object with the minimized conformation
    minimized_ss = Disulfide("minimized")
    minimized_ss.dihedrals = result.x
    minimized_ss.build_yourself()

    return minimized_ss


def minimize_dse_energy(initial_conformation=[-60, -60, -90, -60, -60]):
    """
    Minimizes the DSE energy of a Disulfide object using the Nelder-Mead optimization method.

    Parameters:
        initial_conformation (list): Starting dihedral angles [chi1, chi2, chi3, chi4, chi5]
                                    Default: [-60, -60, -90, -60, -60]

    Returns:
        Disulfide: The minimized Disulfide object with optimal conformation
    """

    def objective_function(x):
        """Calculate DSE energy for given dihedral angles"""
        temp_ss = Disulfide("temp")
        temp_ss.dihedrals = x
        return temp_ss._calculate_dse()

    # Run the minimization
    result = minimize(objective_function, initial_conformation, method="Nelder-Mead")

    # Create a Disulfide object with the minimized conformation
    minimized_ss = Disulfide("minimized")
    minimized_ss.dihedrals = result.x
    minimized_ss.build_yourself()

    return minimized_ss


if __name__ == "__main__":
    # Test the functions with the default starting conformation
    pdb_ss = Load_PDB_SS(verbose=True, subset=False)
    best = pdb_ss["2q7q_75D_140D"]
    initial_ss = best

    print("Initial conformation:")
    print(
        f"Dihedrals [χ1, χ2, χ3, χ4, χ5]: [{', '.join(f'{x:.1f}' for x in initial_ss.dihedrals)}]"
    )
    print(f"DSE Energy: {initial_ss._calculate_dse():.2f} kJ/mol")
    print(f"Standard Energy: {initial_ss.energy:.2f} kcal/mol")

    print("\n=== DSE Energy Minimization ===")
    min_dse = minimize_dse_energy()
    print(
        f"Dihedrals [χ1, χ2, χ3, χ4, χ5]: [{', '.join(f'{x:.1f}' for x in min_dse.dihedrals)}]"
    )
    print(f"DSE Energy: {min_dse._calculate_dse():.2f} kJ/mol")
    print(f"Standard Energy: {min_dse.energy:.2f} kcal/mol")
    dse_reduction = initial_ss._calculate_dse() - min_dse._calculate_dse()
    print(f"DSE Energy reduction: {dse_reduction:.2f} kJ/mol")

    print("\n=== Standard Energy Minimization ===")
    min_std = minimize_standard_energy()
    print(
        f"Dihedrals [χ1, χ2, χ3, χ4, χ5]: [{', '.join(f'{x:.1f}' for x in min_std.dihedrals)}]"
    )
    print(f"DSE Energy: {min_std._calculate_dse():.2f} kJ/mol")
    print(f"Standard Energy: {min_std.energy:.2f} kcal/mol")
    std_reduction = initial_ss.energy - min_std.energy
    print(f"Standard Energy reduction: {std_reduction:.2f} kcal/mol")

    neighbors = best.torsion_neighbors(pdb_ss.SSList, 3.0)
    print(f"Number of neighbors within 3.0 Å: {len(neighbors)}")
    for i, neighbor in enumerate(neighbors):
        print(
            f"Neighbor {i + 1}: {neighbor.name} with DSE Energy: {neighbor._calculate_dse():.2f} kJ/mol, standard Energy: {neighbor.energy:.2f} kcal/mol"
        )
