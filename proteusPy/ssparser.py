"""
ssparser.py

This module provides functionality to parse PDB files to extract disulfide bond (SSBOND) and atom (ATOM) records.
It includes functions to read SSBOND records from the header section of PDB files, extract proximal and distal
parameters, and collect relevant atom information. The extracted data is organized into a dictionary format
for further processing or analysis.

Functions:
- extract_ssbonds_and_atoms(input_pdb_file, verbose=False): Extracts SSBOND and ATOM records from a PDB file.
- extract_and_write_ssbonds_and_atoms(input_pdb_file, output_pkl_file, verbose=False): Extracts disulfide bonds
  and atom information from a PDB file and writes it to a .pkl file.

Dependencies:
- os
- pickle
- proteusPy.logger_config

Usage:
- Import the module and call the desired functions with appropriate arguments.
"""

import os
import pickle

from proteusPy.logger_config import get_logger

_logger = get_logger("ssparser")

# New function to extract the disulfide bonds from the PDB files by
# directly reading the SSBOND records in the header section of the PDB file,
# extracting the proximal, distal parameters and atoms. This creates a dict
# containing the relevant info as shown below.


def extract_ssbonds_and_atoms(input_pdb_file, verbose=False) -> tuple:
    """
    Extracts SSBOND and ATOM records from a PDB file.

    This function reads a PDB file to collect SSBOND records and ATOM records for cysteine residues.
    It then extracts the ATOM records corresponding to the SSBOND records and returns the collected
    data as a dictionary, along with the number of SSBOND records found and any errors encountered.

    Args:
    - input_pdb_file (str): The path to the input PDB file.

    Returns:
    - tuple: A tuple containing:
        - dict: A dictionary containing the SSBOND records and the corresponding ATOM records. The dictionary
          has the following structure:
            {
                "ssbonds": list of SSBOND records (str),
                "atoms": {
                    (chain_id, res_seq_num, atom_name): {
                        "line": ATOM record line (str),
                        "x": x-coordinate (float),
                        "y": y-coordinate (float),
                        "z": z-coordinate (float)
                    },
                    ...
                },
                "pairs": [
                    {
                        "proximal": (chain_id1, res_seq_num1),
                        "distal": (chain_id2, res_seq_num2),
                        "chains": (chain_id1, chain_id2),
                        "phipsi": {
                            "proximal-1": {"N": [x, y, z], "C": [x, y, z]},
                            "proximal+1": {"N": [x, y, z], "C": [x, y, z]},
                            "distal-1": {"N": [x, y, z], "C": [x, y, z]},
                            "distal+1": {"N": [x, y, z], "C": [x, y, z]}
                        }
                    },
                    ...
                ]
            }
        - int: The number of SSBOND records found.
        - list: A list of error messages encountered during processing.
    """
    if not os.path.exists(input_pdb_file):
        return None

    ssbonds = []
    atom_list = {}
    errors = []
    pairs = []

    # Read the PDB file and collect SSBOND and ATOM records

    with open(input_pdb_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("SSBOND"):
            ssbonds.append(line)
        elif line.startswith("ATOM"):
            # Create a map to quickly find ATOM records by residue sequence number, chain ID, and atom name
            chain_id = line[21].strip()
            res_seq_num = line[22:26].strip()
            atom_name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            key = (chain_id, res_seq_num, atom_name)
            atom_list[key] = {"line": line, "x": x, "y": y, "z": z}
            if verbose:
                print(
                    f"Found ATOM record for chain {chain_id}, residue {res_seq_num}, atom {atom_name}"
                )

    # Extract the ATOM records corresponding to SSBOND
    ssbond_atom_list = {"ssbonds": ssbonds, "atoms": {}, "pairs": pairs}
    for ssbond in ssbonds:
        parts = ssbond.split()
        chain_id1 = parts[3]
        res_seq_num1 = parts[4]
        chain_id2 = parts[6]
        res_seq_num2 = parts[7]

        # Add the corresponding ATOM records to the ssbond_atom_list
        for atom_name in ["N", "CA", "C", "O", "CB", "SG"]:
            atom_record1 = atom_list.get((chain_id1, res_seq_num1, atom_name))
            atom_record2 = atom_list.get((chain_id2, res_seq_num2, atom_name))
            if atom_record1:
                ssbond_atom_list["atoms"][
                    (chain_id1, res_seq_num1, atom_name)
                ] = atom_record1
            else:
                errors.append(
                    f"Atom record not found for chain {chain_id1}, residue {res_seq_num1}, atom {atom_name}"
                )
                if verbose:
                    _logger.error(
                        f"Atom record not found for chain {chain_id1}, residue {res_seq_num1}, atom {atom_name}"
                    )

            if atom_record2:
                ssbond_atom_list["atoms"][
                    (chain_id2, res_seq_num2, atom_name)
                ] = atom_record2
            else:
                errors.append(
                    f"Atom record not found for chain {chain_id2}, residue {res_seq_num2}, atom {atom_name}"
                )
                if verbose:
                    _logger.error(
                        f"Atom record not found for chain {chain_id2}, residue {res_seq_num2}, atom {atom_name}"
                    )

        # Collect phi/psi related atoms
        def get_phipsi_atoms(chain_id, res_seq_num):
            phipsi_atoms = {}
            for offset in [-1, 1]:
                for atom_name in ["N", "C"]:
                    key = (chain_id, str(int(res_seq_num) + offset), atom_name)
                    atom_record = atom_list.get(key)
                    if atom_record:
                        if f"{res_seq_num}{offset}" not in phipsi_atoms:
                            phipsi_atoms[f"{res_seq_num}{offset}"] = {}
                        phipsi_atoms[f"{res_seq_num}{offset}"][atom_name] = [
                            atom_record["x"],
                            atom_record["y"],
                            atom_record["z"],
                        ]
                    else:
                        errors.append(
                            f"Atom record not found for chain {chain_id}, residue {str(int(res_seq_num) + offset)}, atom {atom_name}"
                        )
                        if verbose:
                            _logger.error(
                                f"Atom record not found for chain {chain_id}, residue {str(int(res_seq_num) + offset)}, atom {atom_name}"
                            )
            return phipsi_atoms

        phipsi = {
            "proximal-1": get_phipsi_atoms(chain_id1, res_seq_num1).get(
                f"{res_seq_num1}-1", {}
            ),
            "proximal+1": get_phipsi_atoms(chain_id1, res_seq_num1).get(
                f"{res_seq_num1}+1", {}
            ),
            "distal-1": get_phipsi_atoms(chain_id2, res_seq_num2).get(
                f"{res_seq_num2}-1", {}
            ),
            "distal+1": get_phipsi_atoms(chain_id2, res_seq_num2).get(
                f"{res_seq_num2}+1", {}
            ),
        }

        # Add the pair information to the pairs list
        pairs.append(
            {
                "proximal": (chain_id1, res_seq_num1),
                "distal": (chain_id2, res_seq_num2),
                "chains": (chain_id1, chain_id2),
                "phipsi": phipsi,
            }
        )

    return ssbond_atom_list, len(ssbonds), len(errors)


def extract_and_write_ssbonds_and_atoms(input_pdb_file, output_pkl_file, verbose=False):
    """
    Extracts disulfide bonds and atom information from a PDB file and writes it to a .pkl file.

    Args:
    - input_pdb_file (str): Path to the input PDB file.
    - output_pkl_file (str): Path to the output .pkl file.
    - verbose (bool): Flag to enable verbose logging.
    """
    if not os.path.exists(input_pdb_file):
        _logger.error(f"Input PDB file {input_pdb_file} does not exist.")
        return None

    if verbose:
        _logger.info(f"Loading disulfides from {input_pdb_file}")

    ssbond_atom_list, _, _ = extract_ssbonds_and_atoms(input_pdb_file)

    if verbose:
        _logger.info(
            f"Writing disulfide bond and atom information to {output_pkl_file}"
        )

    with open(output_pkl_file, "wb") as f:
        pickle.dump(ssbond_atom_list, f)

    if verbose:
        _logger.info(
            f"Successfully wrote disulfide bond and atom information to {output_pkl_file}"
        )


# Example usage:
# extract_and_write_ssbonds_and_atoms("path_to_pdb_file.pdb", "output_file.pkl", verbos


def print_disulfide_bond_info_dict(ssbond_atom_data):
    """
    Prints the disulfide bond information in a pretty format.

    Args:
    - ssbond_atom_data (dict): A dictionary containing the SSBOND records and the corresponding ATOM records. The dictionary
          has the following structure:
            {
                "ssbonds": list of SSBOND records (str),
                "atoms": {
                    (chain_id, res_seq_num, atom_name): {
                        "line": ATOM record line (str),
                        "x": x-coordinate (float),
                        "y": y-coordinate (float),
                        "z": z-coordinate (float)
                    },
                    ...
                },
                "pairs": [
                    {
                        "proximal": (chain_id1, res_seq_num1),
                        "distal": (chain_id2, res_seq_num2),
                        "chains": (chain_id1, chain_id2),
                        "phipsi": {
                            "proximal-1": {"N": [x, y, z], "C": [x, y, z]},
                            "proximal+1": {"N": [x, y, z], "C": [x, y, z]},
                            "distal-1": {"N": [x, y, z], "C": [x, y, z]},
                            "distal+1": {"N": [x, y, z], "C": [x, y, z]}
                        }
                    },
                    ...
                ]
            }
    """
    if ssbond_atom_data is None:
        print("No disulfide bonds found.")
        return

    ssbonds = ssbond_atom_data.get("ssbonds", [])
    atoms = ssbond_atom_data.get("atoms", {})
    pairs = ssbond_atom_data.get("pairs", [])

    for pair in pairs:
        proximal = pair["proximal"]
        distal = pair["distal"]
        chain_id1, res_seq_num1 = proximal
        chain_id2, res_seq_num2 = distal

        print(
            f"Disulfide Bond between Chain {chain_id1} Residue {res_seq_num1} and Chain {chain_id2} Residue {res_seq_num2}"
        )
        print(f"Proximal Residue (Chain {chain_id1}, Residue {res_seq_num1}):")
        for atom_name in ["N", "CA", "C", "O", "CB", "SG"]:
            atom_record = atoms.get((chain_id1, res_seq_num1, atom_name))
            if atom_record:
                print(
                    f"  Atom {atom_name}: ({atom_record['x']:.3f}, {atom_record['y']:.3f}, {atom_record['z']:.3f})"
                )
            else:
                print(f"  Atom {atom_name}: Not found")

        print(f"Distal Residue (Chain {chain_id2}, Residue {res_seq_num2}):")
        for atom_name in ["N", "CA", "C", "O", "CB", "SG"]:
            atom_record = atoms.get((chain_id2, res_seq_num2, atom_name))
            if atom_record:
                print(
                    f"  Atom {atom_name}: ({atom_record['x']:.3f}, {atom_record['y']:.3f}, {atom_record['z']:.3f})"
                )
            else:
                print(f"  Atom {atom_name}: Not found")

        print("Phi/Psi Atoms:")
        for key, phipsi_atoms in pair["phipsi"].items():
            print(f"  {key}:")
            for atom_name, coords in phipsi_atoms.items():
                res_seq_num = (
                    int(res_seq_num1) - 1
                    if "proximal-1" in key
                    else (
                        int(res_seq_num1) + 1
                        if "proximal+1" in key
                        else (
                            int(res_seq_num2) - 1
                            if "distal-1" in key
                            else int(res_seq_num2) + 1
                        )
                    )
                )
                print(
                    f"    Atom {atom_name} (Residue {res_seq_num}): ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})"
                )

        print("-" * 50)


from Bio.PDB import Vector


def get_atom_coordinates(ssbond_dict, chain_id, res_seq_num, atom_name) -> Vector:
    """
    Accessor function to get the coordinates of a specific atom in a residue.

    Args:
    - ssbond_dict (dict): The dictionary containing SSBOND and ATOM records.
    - chain_id (str): The chain identifier.
    - res_seq_num (str): The residue sequence number.
    - atom_name (str): The name of the atom.

    Returns:
    - list: A list containing the x, y, z coordinates of the atom if found, otherwise None.
    """
    key = (chain_id, res_seq_num, atom_name)
    if key in ssbond_dict["atoms"]:
        atom_record = ssbond_dict["atoms"][key]
        return Vector([atom_record["x"], atom_record["y"], atom_record["z"]])
    else:
        return Vector([])


def get_residue_atoms_coordinates(ssbond_dict, chain_id, res_seq_num):
    """
    Accessor function to get the coordinates of specific atoms in a residue in the order N, CA, C, O, CB, SG.

    Args:
    - ssbond_dict (dict): The dictionary containing SSBOND and ATOM records.
    - chain_id (str): The chain identifier.
    - res_seq_num (str): The residue sequence number.

    Returns:
    - list: A list of vectors, where each vector is a list containing the x, y, z coordinates of the atom.
            If an atom is not found, None is placed in its position.
    """
    from Bio.PDB import Vector

    atom_names = ["N", "CA", "C", "O", "CB", "SG"]
    coordinates = []

    for atom_name in atom_names:
        key = (chain_id, res_seq_num, atom_name)
        if key in ssbond_dict["atoms"]:
            atom_record = ssbond_dict["atoms"][key]
            coordinates.append(
                Vector([atom_record["x"], atom_record["y"], atom_record["z"]])
            )
        else:
            coordinates.append(Vector(0, 0, 0))
            _logger.error(
                f"Atom {atom_name} in residue {chain_id} {res_seq_num} not found."
            )

    return coordinates


# Example usage
# ssbond_dict, num_ssbonds, errors = extract_ssbonds_and_atoms(
#    "/Users/egs/PDB/good/pdb6f99.ent"
# )
# coordinates = get_residue_atoms_coordinates(ssbond_dict, "A", "100")
# print("Coordinates:", coordinates)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
