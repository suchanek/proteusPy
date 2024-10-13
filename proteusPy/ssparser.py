# pylint: disable=C0301

"""
ssparser.py

This module provides functionality to parse PDB files to extract disulfide bond (SSBOND)
and atom (ATOM) records. It includes functions to read SSBOND records from the header section 
of PDB files, extract proximal and distal parameters, and collect relevant atom information. 
The extracted data is organized into a dictionary format for further processing or analysis.

Functions:
- extract_ssbonds_and_atoms(input_pdb_file, verbose=False): Extracts SSBOND and ATOM records 
from a PDB file.
- extract_and_write_ssbonds_and_atoms(input_pdb_file, output_pkl_file, verbose=False): Extracts 
disulfide bonds and atom information from a PDB file and writes it to a .pkl file.

Dependencies:
- os
- pickle
- proteusPy.logger_config

Usage:
- Import the module and call the desired functions with appropriate arguments.
"""

import os
import pickle

from proteusPy.logger_config import create_logger
from proteusPy.vector3D import Vector3D

_logger = create_logger(__name__)


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
        mess = f"Filename {filename} does not follow the expected format 'pdbID.ent'"
        raise ValueError(mess)


def parse_helix_record(record):
    """
    Parses a HELIX record from a PDB file to extract information about helices.

    :param record: The HELIX record as a string.
    :type record: str
    :return: A dictionary containing helix ID, start and end residue information, and helix class.
    :rtype: dict
    """
    helix_id = record[11:14].strip()
    start_chain_id = record[19].strip()
    start_res_seq = int(record[21:25].strip())
    end_chain_id = record[31].strip()
    end_res_seq = int(record[33:37].strip())
    helix_class = int(record[38:40].strip())

    return {
        "helix_id": helix_id,
        "start": (start_chain_id, start_res_seq),
        "end": (end_chain_id, end_res_seq),
        "helix_class": helix_class,
    }


def parse_sheet_record(record):
    """
    Parses a SHEET record from a PDB file to extract information about beta strands.

    :Param record: The SHEET record as a string.
    :Return: A dictionary containing strand ID, start and end residue information, and sheet ID.
    """
    strand_id = record[7:10].strip()
    sheet_id = record[11:14].strip()
    start_chain_id = record[21].strip()
    start_res_seq = int(record[22:26].strip())
    end_chain_id = record[32].strip()
    end_res_seq = int(record[33:37].strip())

    return {
        "strand_id": strand_id,
        "sheet_id": sheet_id,
        "start": (start_chain_id, start_res_seq),
        "end": (end_chain_id, end_res_seq),
    }


def parse_turn_record(record):
    """
    Parses a TURN record from a PDB file to extract information about turns in the protein structure.

    :Param record: The TURN record as a string.
    :Return: A dictionary containing turn ID, start and end residue information.
    """
    turn_id = record[7:10].strip()
    start_chain_id = record[19].strip()
    start_res_seq = int(record[20:24].strip())
    end_chain_id = record[30].strip()
    end_res_seq = int(record[31:35].strip())

    return {
        "turn_id": turn_id,
        "start": (start_chain_id, start_res_seq),
        "end": (end_chain_id, end_res_seq),
    }


# New function to extract the disulfide bonds from the PDB files by
# directly reading the SSBOND records in the header section of the PDB file,
# extracting the proximal, distal parameters and atoms. This creates a dict
# containing the relevant info as shown below.

pdb_dict_template = {
    "pdbid": "str",  # Placeholder for a string
    "ssbonds": "str",  # Placeholder for a string
    "atoms": {
        ("chain_id", "res_seq_num", "atom_name"): {
            "coords": ["x", "y", "z"]  # Placeholder for coordinates
        },
    },
    "pairs": [
        {
            "proximal": (
                "chain_id1",
                "res_seq_num1",
            ),  # Placeholder for proximal chain and residue
            "distal": (
                "chain_id2",
                "res_seq_num2",
            ),  # Placeholder for distal chain and residue
            "chains": ("chain_id1", "chain_id2"),  # Placeholder for chain IDs
            "phipsi": {
                "proximal-1": {
                    "N": ["x", "y", "z"],
                    "C": ["x", "y", "z"],
                },  # Placeholder for phi/psi angles
                "proximal+1": {"N": ["x", "y", "z"], "C": ["x", "y", "z"]},
                "distal-1": {"N": ["x", "y", "z"], "C": ["x", "y", "z"]},
                "distal+1": {"N": ["x", "y", "z"], "C": ["x", "y", "z"]},
            },
        },
    ],
    "helices": [
        {
            "strand_id": "strand_id",  # Placeholder for strand ID
            "sheet_id": "sheet_id",  # Placeholder for sheet ID
            "start": (
                "start_chain_id",
                "start_res_seq",
            ),  # Placeholder for start chain and residue
            "end": (
                "end_chain_id",
                "end_res_seq",
            ),  # Placeholder for end chain and residue
        }
    ],
    "sheets": [
        {
            "strand_id": "strand_id",  # Placeholder for strand ID
            "sheet_id": "sheet_id",  # Placeholder for sheet ID
            "start": (
                "start_chain_id",
                "start_res_seq",
            ),  # Placeholder for start chain and residue
            "end": (
                "end_chain_id",
                "end_res_seq",
            ),  # Placeholder for end chain and residue
        }
    ],
    "turns": [
        {
            "turn_id": "turn_id",  # Placeholder for turn ID
            "start": (
                "start_chain_id",
                "start_res_seq",
            ),  # Placeholder for start chain and residue
            "end": (
                "end_chain_id",
                "end_res_seq",
            ),  # Placeholder for end chain and residue
        }
    ],
    "resolution": "float",  # Placeholder for resolution
}


def extract_ssbonds_and_atoms(input_pdb_file, verbose=False, dbg=False) -> tuple:
    """
    Extracts SSBOND, ATOM records, and RESOLUTION from a PDB file.

    This function reads a PDB file to collect SSBOND records, ATOM records for cysteine residues,
    and the RESOLUTION line. It then extracts the ATOM records corresponding to the SSBOND records
    and returns the collected data as a dictionary, along with the number of SSBOND records found
    and any errors encountered.

    Args:
    - input_pdb_file (str): The path to the input PDB file.

    Returns:
    - tuple: A tuple containing:
        - dict: A dictionary containing the SSBOND records, the corresponding ATOM records,
                and the resolution.
          The dictionary has the following structure:
            {
                "pdbid": The PDB ID (str),
                "ssbonds": list of SSBOND records (str),
                "atoms": {
                    (chain_id, res_seq_num, atom_name): {
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
                ],
                "resolution": The resolution value (float)
            }
        - int: The number of SSBOND records found.
        - list: A list of error messages encountered during processing.
    """
    if not os.path.exists(input_pdb_file):
        _logger.error("Input PDB file {input_pdb_file} does not exist.")
        return {}, 0, 0

    ssbonds = []
    atom_list = {}
    errors = []
    pairs = []
    helices = []
    sheets = []
    turns = []
    resolution = None
    pdbid = extract_id_from_filename(input_pdb_file)

    # Read the PDB file and collect SSBOND, ATOM records, and RESOLUTION
    with open(input_pdb_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("SSBOND"):
            ssbonds.append(line)
            if dbg:
                _logger.debug(str(f"Found SSBOND record for {pdbid}: {line.strip()}"))

        elif line.startswith("ATOM") or line.startswith(
            "HETATM"
        ):  # Added HETATM to include non-standard residues like CSS
            # Create a map to quickly find ATOM records by residue sequence number,
            # chain ID, and atom name

            chain_id = line[21].strip()
            res_seq_num = line[22:26].strip()
            atom_name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            key = (chain_id, res_seq_num, atom_name)
            atom_list[key] = {"coords": [x, y, z]}
            if dbg:
                _logger.debug(
                    str(
                        f"Found ATOM record for chain {chain_id}, residue {res_seq_num}, atom {atom_name}"
                    )
                )
        elif line.startswith("REMARK   2 RESOLUTION"):
            # Extract the resolution value using fixed-width columns
            resolution_str = line[22:27].strip()
            try:
                resolution = float(resolution_str)
            except ValueError:
                if verbose:
                    _logger.error(
                        str(
                            f"Error parsing resolution value from line: {line.strip()}. Found: {resolution_str}"
                        )
                    )
            if dbg:
                _logger.debug(str(f"Found RESOLUTION record: {resolution} Å"))

        elif line.startswith("HELIX"):
            helix_info = parse_helix_record(line)
            helices.append(helix_info)
            if verbose:
                _logger.debug(str(f"Found HELIX record: {helix_info}"))

        elif line.startswith("SHEET"):
            sheet_info = parse_sheet_record(line)
            sheets.append(sheet_info)
            if verbose:
                _logger.debug(str(f"Found SHEET record: {sheet_info}"))

        elif line.startswith("TURN"):
            turn_info = parse_turn_record(line)
            turns.append(turn_info)
            if verbose:
                _logger.debug(str(f"Found TURN record: {turn_info}"))

    # Extract the ATOM records corresponding to SSBOND
    ssbond_atom_list = {
        "pdbid": pdbid,
        "ssbonds": ssbonds,
        "atoms": {},
        "pairs": pairs,
        "resolution": resolution,
        "helices": helices,
        "sheets": sheets,
        "turns": turns,
    }

    for ssbond in ssbonds:
        # Extract fields based on fixed-width columns
        chain_id1 = ssbond[15:16].strip()
        res_seq_num1 = ssbond[17:21].strip()
        chain_id2 = ssbond[29:30].strip()
        res_seq_num2 = ssbond[31:35].strip()
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
                    _logger.warning(
                        str(
                            f"Atom record not found for chain {chain_id1}, residue {res_seq_num1}, atom {atom_name}"
                        )
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
                    _logger.warning(
                        str(
                            f"Atom record not found for chain {chain_id2}, residue {res_seq_num2}, atom {atom_name}"
                        )
                    )

        # Collect phi/psi related atoms
        def get_phipsi_atoms(chain_id, res_seq_num):
            phipsi_atoms = {}
            for offset in [-1, 1]:
                try:
                    int(res_seq_num) + offset
                except ValueError:
                    _logger.error(
                        "get_phiipsi_atoms: ValueError: {res_seq_num} + {offset}"
                    )
                    continue

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
                            _logger.warning(
                                str(
                                    f"Atom record not found for chain {chain_id}, residue {str(int(res_seq_num) + offset)}, atom {atom_name}"
                                )
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

        prox_secondary = get_secondary_structure(
            chain_id1, res_seq_num1, ssbond_atom_list
        )
        dist_secondary = get_secondary_structure(
            chain_id2, res_seq_num2, ssbond_atom_list
        )

        # Add the pair information to the pairs list
        pairs.append(
            {
                "proximal": (chain_id1, res_seq_num1),
                "distal": (chain_id2, res_seq_num2),
                "chains": (chain_id1, chain_id2),
                "phipsi": phipsi,
                "prox_secondary": prox_secondary,
                "dist_secondary": dist_secondary,
            }
        )

    return ssbond_atom_list, len(ssbonds), len(errors)


# Example usage:
# input_pdb_file = "path/to/pdbfile.pdb"
# result = extract_ssbonds_and_atoms(input_pdb_file, verbose=True)
# print(result)


def get_secondary_structure(chain_id, res_seq_num, pdb_data) -> str:
    """
    Determine the secondary structure type for a given chain_id and res_seq_num.

    :param chain_id: The chain identifier.
    :type chain_id: str
    :param res_seq_num: The residue sequence number.
    :type res_seq_num: int
    :param pdb_data: The PDB data dictionary containing helices, sheets, and turns.
    :type pdb_data: dict
    :return: The secondary structure type ('helix', 'sheet', 'turn', 'nosecondary').
    :rtype: str
    """
    # Check helices
    for helix in pdb_data.get("helices", []):
        if (
            helix["start"][0] == chain_id
            and helix["start"][1] <= int(res_seq_num) <= helix["end"][1]
        ):
            return "helix"

    # Check sheets
    for sheet in pdb_data.get("sheets", []):
        if (
            sheet["start"][0] == chain_id
            and sheet["start"][1] <= int(res_seq_num) <= sheet["end"][1]
        ):
            return "sheet"

    # Check turns
    for turn in pdb_data.get("turns", []):
        if (
            turn["start"][0] == chain_id
            and turn["start"][1] <= int(res_seq_num) <= turn["end"][1]
        ):
            return "turn"

    # If no secondary structure is found
    return "nosecondary"


def extract_and_write_ssbonds_and_atoms(
    input_pdb_file, output_pkl_file, verbose=False, dbg=False
) -> None:
    """
    Extracts disulfide bonds and atom information from a PDB file and writes it to a .pkl file.

    Args:
    - input_pdb_file (str): Path to the input PDB file.
    - output_pkl_file (str): Path to the output .pkl file.
    - verbose (bool): Flag to enable verbose logging.
    """
    if not os.path.exists(input_pdb_file):
        _logger.error(str(f"Input PDB file {input_pdb_file} does not exist."))
        return None

    if verbose:
        _logger.info(str(f"Loading disulfides from {input_pdb_file}"))

    ssbond_atom_list, _, _ = extract_ssbonds_and_atoms(
        input_pdb_file, verbose=verbose, dbg=dbg
    )

    if verbose:
        _logger.info(
            str(f"Writing disulfide bond and atom information to {output_pkl_file}")
        )

    with open(output_pkl_file, "wb") as f:
        pickle.dump(ssbond_atom_list, f)

    if verbose:
        _logger.info(
            str(
                f"Successfully wrote disulfide bond and atom information to {output_pkl_file}"
            )
        )


# Example usage:
# extract_and_write_ssbonds_and_atoms("path_to_pdb_file.pdb", "output_file.pkl", verbos


def print_disulfide_bond_info_dict(ssbond_atom_data) -> None:
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
                ],
                "resolution": The resolution value (float)
            }
    """
    if ssbond_atom_data is None:
        print("No disulfide bonds found.")
        return

    i = 1
    # ssbonds = ssbond_atom_data.get("ssbonds", [])
    atoms = ssbond_atom_data.get("atoms", {})
    pairs = ssbond_atom_data.get("pairs", [])
    resolution = ssbond_atom_data.get("resolution", -1.0)
    pdb_id = ssbond_atom_data.get("pdbid", "Unknown")

    print(f"Disulfides in {pdb_id} with resolution {resolution} Å:")
    for pair in pairs:
        proximal = pair["proximal"][1]
        distal = pair["distal"][1]
        res_seq_num1 = proximal
        res_seq_num2 = distal
        chains = pair["chains"]
        chain_id1 = chains[0]
        chain_id2 = chains[1]
        pdb_id = ssbond_atom_data.get("pdbid", "Unknown")

        print(
            f"PDB: {pdb_id} SS {i}: Chain {chain_id1} Residue {res_seq_num1} and Chain {chain_id2} Residue {res_seq_num2}"
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
        i += 1
        print("-" * 50)


def get_atom_coordinates(
    ssbond_dict, chain_id, res_seq_num, atom_name, verbose=False
) -> Vector3D:
    """
    Accessor function to get the coordinates of a specific atom in a residue.

    :param ssbond_dict: The dictionary containing SSBOND and ATOM records.
    :type ssbond_dict: dict
    :param chain_id: The chain identifier.
    :type chain_id: str
    :param res_seq_num: The residue sequence number.
    :type res_seq_num: str
    :param atom_name: The name of the atom.
    :type atom_name: str
    :param verbose: Flag to enable verbose logging.
    :type verbose: bool

    :return: A Vector3D object containing the x, y, z coordinates of the atom if
    found, otherwise an empty Vector3D.
    :rtype: Vector3D
    """
    key = (chain_id, str(res_seq_num), atom_name)
    if key in ssbond_dict["atoms"]:
        atom_record = ssbond_dict["atoms"][key]
        return Vector3D(atom_record["coords"])
    else:
        if verbose:
            _logger.warning(
                str(
                    f"--> get_atom_coordinates: PDB: {ssbond_dict['pdbid']}: Atom {atom_name} in residue {chain_id} {res_seq_num} not found."
                )
            )
        return Vector3D([])


def get_residue_atoms_coordinates(
    ssbond_dict, chain_id, res_seq_num, verbose=False
) -> list:
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

    atom_names = ["N", "CA", "C", "O", "CB", "SG"]
    coordinates = []
    atoms = ssbond_dict.get("atoms", {})
    pdb_id = ssbond_dict.get("pdbid", "Unknown")

    for atom_name in atom_names:
        atom_record = atoms.get((chain_id, str(res_seq_num), atom_name))
        if atom_record:
            coordinates.append(Vector3D(atom_record["coords"]))
        else:
            coordinates.append(Vector3D(0.01, 0.01, 0.01))
            if verbose:
                _logger.warning(
                    str(
                        f"PDB: {pdb_id} Atom {atom_name} in residue {chain_id} {res_seq_num} not found."
                    )
                )

    return coordinates


# Example usage
# ssbond_dict, num_ssbonds, errors = extract_ssbonds_and_atoms(
#    "/Users/egs/PDB/good/pdb6f99.ent"
# )
# coordinates = get_residue_atoms_coordinates(ssbond_dict, "A", "100")
# print("Coordinates:", coordinates)


def get_phipsi_atoms_coordinates(data_dict, chain_id, key):
    """
    Retrieve the phi/psi atoms based on the input dictionary, chain ID, and key.

    :param data_dict: Dictionary containing SSBOND and ATOM records.
    :param chain_id: Chain ID to look for.
    :param key: Key in the form "proximal+1", "distal-1", etc.
    :return: List of Vectors representing the N and C atoms with their coordinates.
    """

    # from proteusPy.ssparser import (
    #    extract_ssbonds_and_atoms,
    #    print_disulfide_bond_info_dict,
    # )

    for pair in data_dict.get("pairs", []):
        if chain_id in pair.get("chains", []):
            phipsi_data = pair.get("phipsi", {})
            if key in phipsi_data:
                n_coords = phipsi_data[key].get("N")
                c_coords = phipsi_data[key].get("C")
                if n_coords and c_coords:
                    return [Vector3D(*n_coords), Vector3D(*c_coords)]
    return []


# Example usage:
# ssbond_dict, num_ssbonds, errors = extract_ssbonds_and_atoms(
#    "/Users/egs/PDB/good/pdb6f99.ent"
# )


def check_file(
    fname: str,
    verbose=False,
    quiet=True,
    dbg=False,
    cutoff=-1.0,
) -> tuple:
    """
    Check all PDB files in the directory `pdb_dir` for SS bond consistency.

    This function processes each PDB file in the specified directory, checking for
    disulfide bond consistency. Files that pass the check are moved to `good_dir`,
    while files that fail are moved to `bad_dir`.

    :param pdb_dir: The directory containing the PDB files to check.
    :type pdb_dir: str
    :param good_dir: The directory to move good files to.
    :type good_dir: str
    :param bad_dir: The directory to move bad files to.
    :type bad_dir: str
    :param verbose: If True, enables verbose logging. Default is False.
    :type verbose: bool
    :param quiet: If True, suppresses non-critical output. Default is True.
    :type quiet: bool

    :return: None
    """

    # from proteusPy import extract_ssbonds_and_atoms, print_disulfide_bond_info_dict

    if os.path.exists(fname) is False:
        print(f"File {fname} does not exist! Exiting...")
        return None

    # Returns > 0 if we can't parse the SSBOND header
    ssbond_dict, found, errors = extract_ssbonds_and_atoms(
        fname, verbose=verbose, dbg=dbg
    )

    if verbose:
        _logger.info(str(f"Found: {found} errors: {errors}"))
        print_disulfide_bond_info_dict(ssbond_dict)

    return found, errors


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
