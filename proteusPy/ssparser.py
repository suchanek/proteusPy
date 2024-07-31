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


def extract_id_from_filename(filename: str) -> str:
    """
    Extract the ID from a filename formatted as 'pdb{id}.ent'.

    Parameters:
    - filename (str): The filename to extract the ID from.

    Returns:
    - str: The extracted ID.
    """
    basename = os.path.basename(filename)
    # Check if the filename follows the expected format
    if basename.startswith("pdb") and filename.endswith(".ent"):
        # Extract the ID part of the filename
        return filename[3:-4]
    else:
        mess = f"Filename {filename} does not follow the expected format 'pdbid .ent'"
        raise ValueError(mess)


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
                "pdbid": The PDB ID (str),
                "ssbonds": list of SSBOND records (str),
                "atoms": {
                    (chain_id, res_seq_num, atom_name): {
                        "line": ATOM record line (str),
                        "coords": [x, y, z]
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
    pdbid = extract_id_from_filename(input_pdb_file)

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
            atom_list[key] = {"line": line, "coords": [x, y, z]}
            if verbose:
                print(
                    f"Found ATOM record for chain {chain_id}, residue {res_seq_num}, atom {atom_name}"
                )

    # Extract the ATOM records corresponding to SSBOND
    ssbond_atom_list = {"pdbid": pdbid, "ssbonds": ssbonds, "atoms": {}, "pairs": pairs}
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
                        phipsi_atoms[f"{res_seq_num}{offset}"][atom_name] = atom_record[
                            "coords"
                        ]  #

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


# Example usage:
# input_pdb_file = "path/to/pdbfile.pdb"
# result = extract_ssbonds_and_atoms(input_pdb_file, verbose=True)
# print(result)


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
                "pdbid": The PDB ID (str),
                "ssbonds": list of SSBOND records (str),
                "atoms": {
                    (chain_id, res_seq_num, atom_name): {
                        "line": ATOM record line (str),
                        "coords": [x, y, z]
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
        proximal = pair["proximal"][1]
        distal = pair["distal"][1]
        res_seq_num1 = proximal
        res_seq_num2 = distal
        chains = pair["chains"]
        chain_id1 = chains[0]
        chain_id2 = chains[1]

        print(
            f"Disulfide Bond between Chain {chain_id1} Residue {res_seq_num1} and Chain {chain_id2} Residue {res_seq_num2}"
        )
        print(f"Proximal Residue (Chain {chain_id1}, Residue {res_seq_num1}):")
        for atom_name in ["N", "CA", "C", "O", "CB", "SG"]:
            atom_record = atoms.get((chain_id1, res_seq_num1, atom_name))
            if atom_record:
                print(f"  Atom {atom_name}: {atom_record['coords']}")
            else:
                print(f"  Atom {atom_name}: Not found")

        print(f"Distal Residue (Chain {chain_id2}, Residue {res_seq_num2}):")
        for atom_name in ["N", "CA", "C", "O", "CB", "SG"]:
            atom_record = atoms.get((chain_id2, res_seq_num2, atom_name))
            if atom_record:
                print(f"  Atom {atom_name}: {atom_record['coords']}")
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
                print(f"    Atom {atom_name} (Residue {res_seq_num}): {coords}")

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
        # return Vector([atom_record["x"], atom_record["y"], atom_record["z"]])
        return Vector(atom_record["coords"])
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
                # Vector([atom_record["x"], atom_record["y"], atom_record["z"]])
                Vector(atom_record["coords"])
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


def get_phipsi_atoms(data_dict, chain_id, key):
    """
    Retrieve the phi/psi atoms based on the input dictionary, chain ID, and key.

    :param data_dict: Dictionary containing SSBOND and ATOM records.
    :param chain_id: Chain ID to look for.
    :param key: Key in the form "proximal+1", "distal-1", etc.
    :return: List of Vectors representing the N and C atoms with their coordinates.
    """
    from Bio.PDB import Vector

    from proteusPy.ssparser import (
        extract_ssbonds_and_atoms,
        print_disulfide_bond_info_dict,
    )

    for pair in data_dict.get("pairs", []):
        if chain_id in pair.get("chains", []):
            phipsi_data = pair.get("phipsi", {})
            if key in phipsi_data:
                n_coords = phipsi_data[key].get("N")
                c_coords = phipsi_data[key].get("C")
                if n_coords and c_coords:
                    return [Vector(*n_coords), Vector(*c_coords)]
    return []


# Example usage:
# ssbond_dict, num_ssbonds, errors = extract_ssbonds_and_atoms(
#    "/Users/egs/PDB/good/pdb6f99.ent"
# )

from proteusPy.Disulfide import Disulfide


def initialize_disulfide_from_coords(
    ssbond_atom_data,
    pdb_id,
    proximal_chain_id,
    distal_chain_id,
    proximal,
    distal,
    resolution,
    verbose=False,
    quiet=True,
) -> Disulfide:
    """
    Initialize a new Disulfide object with atomic coordinates from
    the proximal and distal coordinates, typically taken from a PDB file.
    This routine is primarily used internally when building the compressed
    database.

    :param resolution: structure resolution
    :param quiet: Quiet or noisy parsing, defaults to True
    :raises DisulfideConstructionWarning: Raised when not parsed correctly
    """
    import logging
    import warnings

    import numpy as np
    from Bio.PDB import Vector, calc_dihedral, distance3d

    from proteusPy.ssparser import (
        get_phipsi_atom_coordinates,
        get_residue_atom_coordinates,
    )

    ssbond_name = f"{pdb_id}_{proximal}{proximal_chain_id}_{distal}{distal_chain_id}"
    new_ss = Disulfide(ssbond_name)

    new_ss.pdb_id = pdb_id
    new_ss.resolution = resolution
    prox_atom_list = []
    dist_atom_list = []

    if quiet:
        _logger.setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.CRITICAL)

    # set the objects proximal and distal values
    new_ss.set_resnum(proximal, distal)

    if resolution is not None:
        new_ss.resolution = resolution
    else:
        new_ss.resolution = -1.0

    new_ss.proximal_chain = proximal_chain_id
    new_ss.distal_chain = distal_chain_id

    new_ss.proximal_residue_fullid = proximal
    new_ss.distal_residue_fullid = distal

    # restore loggins
    if quiet:
        _logger.setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)  ## may want to be CRITICAL

    # Get the coordinates for the proximal and distal residues as vectors
    # so we can do math on them later. Trap errors here to avoid problems
    # with missing residues or atoms.

    # proximal residue

    try:
        prox_atom_list = get_residue_atom_coordinates(
            ssbond_atom_data, proximal_chain_id, proximal
        )

        n1 = prox_atom_list[0]
        ca1 = prox_atom_list[1]
        c1 = prox_atom_list[2]
        o1 = prox_atom_list[3]
        cb1 = prox_atom_list[4]
        sg1 = prox_atom_list[5]

    except Exception:
        # i'm torn on this. there are a lot of missing coordinates, so is
        # it worth the trouble to note them? I think so.
        _logger.error(f"Invalid/missing coordinates for: {id}, proximal: {proximal}")
        return False

    # distal residue
    try:
        dist_atom_list = get_residue_atom_coordinates(
            ssbond_atom_data, distal_chain_id, distal
        )
        n2 = dist_atom_list[0]
        ca2 = dist_atom_list[1]
        c2 = dist_atom_list[2]
        o2 = dist_atom_list[3]
        cb2 = dist_atom_list[4]
        sg2 = dist_atom_list[5]

    except Exception:
        _logger.error(f"Invalid/missing coordinates for: {id}, distal: {distal}")
        return False

    # previous residue and next residue - optional, used for phi, psi calculations
    prevprox_atom_list = get_phipsi_atom_coordinates(
        ssbond_atom_data, proximal_chain_id, "proximal-1"
    )

    nextprox_atom_list = get_phipsi_atom_coordinates(
        ssbond_atom_data, proximal_chain_id, "proximal+1"
    )

    prevdist_atom_list = get_phipsi_atom_coordinates(
        ssbond_atom_data, distal_chain_id, "distal-1"
    )

    nextdist_atom_list = get_phipsi_atom_coordinates(
        ssbond_atom_data, distal_chain_id, "distal+1"
    )

    if len(prevprox_atom_list) != 0 and len(nextprox_atom_list) != 0:
        # list is N, C
        cprev_prox = prevprox_atom_list["C"]
        nnext_prox = nextprox_atom_list["N"]
        new_ss.phiprox = np.degrees(calc_dihedral(cprev_prox, n1, ca1, c1))
        new_ss.psiprox = np.degrees(calc_dihedral(n1, ca1, c1, nnext_prox))
    else:
        cprev_prox = nnext_prox = Vector(-1.0, -1.0, -1.0)
        new_ss.missing_atoms = True
        if verbose:
            _logger.warning(
                f"Missing Proximal coords for: {id} {proximal-1} or {proximal+1}, SS {proximal}-{distal}, phi/psi not computed."
            )

    if len(prevdist_atom_list) != 0 and len(nextdist_atom_list) != 0:
        # list is N, C
        cprev_dist = prevdist_atom_list[1]
        nnext_dist = nextdist_atom_list[0]
        new_ss.phidist = np.degrees(calc_dihedral(cprev_dist, n2, ca2, c2))
        new_ss.psidist = np.degrees(calc_dihedral(n2, ca2, c2, nnext_dist))
    else:
        cprev_dist = nnext_dist = Vector(-1.0, -1.0, -1.0)
        new_ss.missing_atoms = True
        if verbose:
            _logger.warning(
                f"Missing Distal coords for: {id} {distal-1} or {distal+1} SS {proximal}-{distal}, phi/psi not computed."
            )

    # update the positions and conformation
    new_ss.set_positions(
        n1,
        ca1,
        c1,
        o1,
        cb1,
        sg1,
        n2,
        ca2,
        c2,
        o2,
        cb2,
        sg2,
        cprev_prox,
        nnext_prox,
        cprev_dist,
        nnext_dist,
    )

    # calculate and set the disulfide dihedral angles
    new_ss.chi1 = np.degrees(calc_dihedral(n1, ca1, cb1, sg1))
    new_ss.chi2 = np.degrees(calc_dihedral(ca1, cb1, sg1, sg2))
    new_ss.chi3 = np.degrees(calc_dihedral(cb1, sg1, sg2, cb2))
    new_ss.chi4 = np.degrees(calc_dihedral(sg1, sg2, cb2, ca2))
    new_ss.chi5 = np.degrees(calc_dihedral(sg2, cb2, ca2, n2))
    new_ss.rho = np.degrees(calc_dihedral(n1, ca1, ca2, n2))

    new_ss.ca_distance = distance3d(new_ss.ca_prox, new_ss.ca_dist)
    new_ss.cb_distance = distance3d(new_ss.cb_prox, new_ss.cb_dist)
    new_ss.torsion_array = np.array(
        (new_ss.chi1, new_ss.chi2, new_ss.chi3, new_ss.chi4, new_ss.chi5)
    )
    new_ss.compute_torsion_length()

    # calculate and set the SS bond torsional energy
    new_ss.compute_torsional_energy()

    # compute and set the local coordinates
    new_ss.compute_local_coords()

    # turn warnings back on
    if quiet:
        _logger.setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)

    return new_ss


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
