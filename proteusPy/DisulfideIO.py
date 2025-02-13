"""
This module provides I/O operations for the proteusPy package's disulfide bond functionality.
It handles loading and extracting disulfide bonds from PDB files.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-12
"""

import copy
import logging
import os
from pathlib import Path

from proteusPy import Disulfide
from proteusPy.Disulfide import DisulfideList, Initialize_Disulfide_From_Coords
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import MODEL_DIR, PDB_DIR
from proteusPy.ssparser import extract_ssbonds_and_atoms

_logger = create_logger(__name__)


def load_disulfides_from_id(
    pdb_id: str,
    pdb_dir=MODEL_DIR,
    verbose=False,
    quiet=True,
    dbg=False,
    cutoff=-1.0,
    sg_cutoff=-1.0,
) -> DisulfideList:
    """
    Loads the Disulfides by PDB ID and returns a DisulfideList of Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.

    :param pdb_id: The name of the PDB entry.
    :param pdb_dir: Path to the PDB files, defaults to MODEL_DIR
    :param verbose: Print info while parsing.
    :param quiet: Suppress non-error logging output.
    :param dbg: Enable debug logging.
    :param cutoff: Distance cutoff for filtering disulfides.
    :param sg_cutoff: SG distance cutoff for filtering disulfides.
    :return: A DisulfideList of Disulfide objects initialized from the file.
    """
    i = 1
    proximal = distal = -1
    chain1_id = chain2_id = ""
    ssbond_atom_list = {}
    num_ssbonds = 0
    delta = 0
    errors = 0
    resolution = -1.0

    structure_fname = os.path.join(pdb_dir, f"pdb{pdb_id}.ent")

    if verbose:
        mess = f"Parsing structure: {pdb_id}:"
        _logger.info(mess)

    SSList = DisulfideList([], pdb_id, resolution)

    ssbond_atom_list, num_ssbonds, errors = extract_ssbonds_and_atoms(
        structure_fname, verbose=verbose
    )

    if num_ssbonds == 0:
        mess = f"->{pdb_id} has no SSBonds."
        if verbose:
            print(mess)
        _logger.warning(mess)
        return None

    if quiet:
        _logger.setLevel(logging.ERROR)

    if verbose:
        mess = f"{pdb_id} has {num_ssbonds} SSBonds, found: {errors} errors"
        _logger.info(mess)

    resolution = ssbond_atom_list["resolution"]
    for pair in ssbond_atom_list["pairs"]:
        proximal = pair["proximal"][1]
        chain1_id = pair["proximal"][0]
        distal = pair["distal"][1]
        chain2_id = pair["distal"][0]
        proximal_secondary = pair["prox_secondary"]
        distal_secondary = pair["dist_secondary"]

        if dbg:
            mess = f"Proximal: {proximal} {chain1_id} Distal: {distal} {chain2_id}"
            _logger.debug(mess)

        proximal_int = int(proximal)
        distal_int = int(distal)

        if proximal == distal:
            if verbose:
                mess = (
                    f"SSBond record has (proximal == distal): "
                    f"{pdb_id} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}."
                )
                _logger.error(mess)

        if proximal == distal and chain1_id == chain2_id:
            mess = (
                f"SSBond record has self reference, skipping: "
                f"{pdb_id} <{proximal} {chain1_id}> <{distal} {chain2_id}>"
            )

            _logger.error(mess)
            continue

        if verbose:
            mess = (
                f"SSBond: {i}: {pdb_id}: {proximal} {chain1_id} - {distal} {chain2_id}"
            )
            _logger.info(mess)

        new_ss = Initialize_Disulfide_From_Coords(
            ssbond_atom_list,
            pdb_id,
            chain1_id,
            chain2_id,
            proximal_int,
            distal_int,
            resolution,
            proximal_secondary,
            distal_secondary,
            verbose=verbose,
            quiet=quiet,
            dbg=dbg,
        )

        if new_ss is not None:
            SSList.append(new_ss)
            if verbose:
                mess = f"Initialized Disulfide: {pdb_id} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}."
                _logger.info(mess)
        else:
            mess = f"Cannot initialize Disulfide: {pdb_id} <{proximal} {chain1_id}> <{distal} {chain2_id}>"
            _logger.error(mess)

        i += 1

    # restore default logging level
    if quiet:
        _logger.setLevel(logging.WARNING)

    num_ssbonds = len(SSList)

    if cutoff > 0:
        SSList = SSList.filter_by_distance(cutoff)
        delta = num_ssbonds - len(SSList)
        if delta:
            _logger.error(
                "Filtered %d -> %d SSBonds by Ca distance, %s, delta is: %d",
                num_ssbonds,
                len(SSList),
                pdb_id,
                delta,
            )
        num_ssbonds = len(SSList)

    if sg_cutoff > 0:
        SSList = SSList.filter_by_sg_distance(sg_cutoff)
        delta = num_ssbonds - len(SSList)
        if delta:
            _logger.error(
                "Filtered %d -> %d SSBonds by Sγ distance, %s, delta is: %d",
                num_ssbonds,
                len(SSList),
                pdb_id,
                delta,
            )

    return copy.deepcopy(SSList)


def extract_disulfide(
    pdb_filename: str, verbose=False, quiet=True, pdbdir=PDB_DIR
) -> DisulfideList:
    """
    Read the PDB file represented by `pdb_filename` and return a `DisulfideList`
    containing the Disulfide bonds found.

    :param pdb_filename: The filename of the PDB file to read.
    :param verbose: Display more messages (default: False).
    :param quiet: Turn off DisulfideConstruction warnings (default: True).
    :param pdbdir: Path to PDB files (default: PDB_DIR).
    :return: A `DisulfideList` containing the Disulfide bonds found.
    """

    def extract_id_from_filename(filename: str) -> str:
        """
        Extract the ID from a filename formatted as 'pdb{id}.ent'.

        :param filename: The filename to extract the ID from.
        :return: The extracted ID.
        """
        basename = os.path.basename(filename)
        # Check if the filename follows the expected format
        if basename.startswith("pdb") and filename.endswith(".ent"):
            # Extract the ID part of the filename
            return filename[3:-4]

        mess = f"Filename {filename} does not follow the expected format 'pdb{id}.ent'"
        raise ValueError(mess)

    pdbid = extract_id_from_filename(pdb_filename)

    # returns an empty list if none are found.
    _sslist = DisulfideList([], pdbid)
    _sslist = load_disulfides_from_id(
        pdbid, verbose=verbose, quiet=quiet, pdb_dir=pdbdir
    )

    if len(_sslist) == 0 or _sslist is None:
        mess = f"Can't find SSBonds: {pdbid}"
        _logger.error(mess)
        return DisulfideList([], pdbid)

    return _sslist
