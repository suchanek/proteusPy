"""
This module provides I/O operations for the proteusPy package's disulfide bond functionality.
It handles loading and extracting disulfide bonds from PDB files.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-12
"""

# pylint: disable=C0103 # snake case
# pylint: disable=C0301 # line too long
# pylint: disable=C0302 # too many lines in module
# pylint: disable=W1203 # use of % formatting in logging functions

import copy
import logging
import os

import numpy as np

from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import MODEL_DIR, PDB_DIR
from proteusPy.ssparser import (
    extract_ssbonds_and_atoms,
    get_phipsi_atoms_coordinates,
    get_residue_atoms_coordinates,
)
from proteusPy.vector3D import Vector3D, calc_dihedral, distance3d

_logger = create_logger(__name__)


def Initialize_Disulfide_From_Coords(
    ssbond_atom_data,
    pdb_id,
    proximal_chain_id,
    distal_chain_id,
    proximal,
    distal,
    resolution,
    proximal_secondary,
    distal_secondary,
    verbose=False,
    quiet=True,
    dbg=False,
) -> Disulfide:
    """
    Initialize a new Disulfide object with atomic coordinates from
    the proximal and distal coordinates, typically taken from a PDB file.
    This routine is primarily used internally when building the compressed
    database.

    :param ssbond_atom_data: Dictionary containing atomic data for the disulfide bond.
    :type ssbond_atom_data: dict
    :param pdb_id: PDB identifier for the structure.
    :type pdb_id: str
    :param proximal_chain_id: Chain identifier for the proximal residue.
    :type proximal_chain_id: str
    :param distal_chain_id: Chain identifier for the distal residue.
    :type distal_chain_id: str
    :param proximal: Residue number for the proximal residue.
    :type proximal: int
    :param distal: Residue number for the distal residue.
    :type distal: int
    :param resolution: Structure resolution.
    :type resolution: float
    :param verbose: If True, enables verbose logging. Defaults to False.
    :type verbose: bool, optional
    :param quiet: If True, suppresses logging output. Defaults to True.
    :type quiet: bool, optional
    :param dbg: If True, enables debug mode. Defaults to False.
    :type dbg: bool, optional
    :return: An instance of the Disulfide class initialized with the provided coordinates.
    :rtype: Disulfide
    :raises DisulfideConstructionWarning: Raised when the disulfide bond is not parsed correctly.

    """

    ssbond_name = f"{pdb_id}_{proximal}{proximal_chain_id}_{distal}{distal_chain_id}"
    new_ss = Disulfide(ssbond_name)

    new_ss.pdb_id = pdb_id
    new_ss.resolution = resolution
    new_ss.proximal_secondary = proximal_secondary
    new_ss.distal_secondary = distal_secondary
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

    # restore loggins
    if quiet:
        _logger.setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.CRITICAL)  ## may want to be CRITICAL

    # Get the coordinates for the proximal and distal residues as vectors
    # so we can do math on them later. Trap errors here to avoid problems
    # with missing residues or atoms.

    # proximal residue

    try:
        prox_atom_list = get_residue_atoms_coordinates(
            ssbond_atom_data, proximal_chain_id, proximal
        )

        n1 = prox_atom_list[0]
        ca1 = prox_atom_list[1]
        c1 = prox_atom_list[2]
        o1 = prox_atom_list[3]
        cb1 = prox_atom_list[4]
        sg1 = prox_atom_list[5]

    except KeyError:
        # i'm torn on this. there are a lot of missing coordinates, so is
        # it worth the trouble to note them? I think so.
        _logger.error(f"Invalid/missing coordinates for: {id}, proximal: {proximal}")
        return None

    # distal residue
    try:
        dist_atom_list = get_residue_atoms_coordinates(
            ssbond_atom_data, distal_chain_id, distal
        )
        n2 = dist_atom_list[0]
        ca2 = dist_atom_list[1]
        c2 = dist_atom_list[2]
        o2 = dist_atom_list[3]
        cb2 = dist_atom_list[4]
        sg2 = dist_atom_list[5]

    except KeyError:
        _logger.error(f"Invalid/missing coordinates for: {id}, distal: {distal}")
        return False

    # previous residue and next residue - optional, used for phi, psi calculations
    prevprox_atom_list = get_phipsi_atoms_coordinates(
        ssbond_atom_data, proximal_chain_id, "proximal-1"
    )

    nextprox_atom_list = get_phipsi_atoms_coordinates(
        ssbond_atom_data, proximal_chain_id, "proximal+1"
    )

    prevdist_atom_list = get_phipsi_atoms_coordinates(
        ssbond_atom_data, distal_chain_id, "distal-1"
    )

    nextdist_atom_list = get_phipsi_atoms_coordinates(
        ssbond_atom_data, distal_chain_id, "distal+1"
    )

    if len(prevprox_atom_list) != 0:
        cprev_prox = prevprox_atom_list[1]
        new_ss.phiprox = calc_dihedral(cprev_prox, n1, ca1, c1)

    else:
        cprev_prox = Vector3D(-1.0, -1.0, -1.0)
        new_ss.missing_atoms = True
        if verbose:
            _logger.warning(
                f"Missing Proximal coords for: {id} {proximal}-1. SS: {proximal}-{distal}, phi/psi not computed."
            )

    if len(prevdist_atom_list) != 0:
        # list is N, C
        cprev_dist = prevdist_atom_list[1]
        new_ss.phidist = calc_dihedral(cprev_dist, n2, ca2, c2)
    else:
        cprev_dist = nnext_dist = Vector3D(-1.0, -1.0, -1.0)
        new_ss.missing_atoms = True
        if verbose:
            _logger.warning(
                f"Missing Distal coords for: {id} {distal}-1). S:S {proximal}-{distal}, phi/psi not computed."
            )

    if len(nextprox_atom_list) != 0:
        nnext_prox = nextprox_atom_list[0]
        new_ss.psiprox = calc_dihedral(n1, ca1, c1, nnext_prox)
    else:
        nnext_prox = Vector3D(-1.0, -1.0, -1.0)
        new_ss.missing_atoms = True
        _logger.warning(
            f"Missing Proximal coords for: {id} {proximal}+1). SS: {proximal}-{distal}, phi/psi not computed."
        )

    if len(nextdist_atom_list) != 0:
        nnext_dist = nextdist_atom_list[0]
        new_ss.psidist = calc_dihedral(n2, ca2, c2, nnext_dist)
    else:
        nnext_dist = Vector3D(-1.0, -1.0, -1.0)
        new_ss.missing_atoms = True
        _logger.warning(
            f"Missing Distal coords for: {id} {distal}+1). SS: {proximal}-{distal}, phi/psi not computed."
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
    new_ss.chi1 = calc_dihedral(n1, ca1, cb1, sg1)
    new_ss.chi2 = calc_dihedral(ca1, cb1, sg1, sg2)
    new_ss.chi3 = calc_dihedral(cb1, sg1, sg2, cb2)
    new_ss.chi4 = calc_dihedral(sg1, sg2, cb2, ca2)
    new_ss.chi5 = calc_dihedral(sg2, cb2, ca2, n2)
    new_ss.ca_distance = distance3d(new_ss.ca_prox, new_ss.ca_dist)
    new_ss.cb_distance = distance3d(new_ss.cb_prox, new_ss.cb_dist)
    new_ss.sg_distance = distance3d(new_ss.sg_prox, new_ss.sg_dist)

    new_ss.torsion_array = np.array(
        (new_ss.chi1, new_ss.chi2, new_ss.chi3, new_ss.chi4, new_ss.chi5)
    )
    new_ss._compute_torsion_length()

    # calculate and set the SS bond torsional energy
    new_ss._compute_torsional_energy()

    # compute and set the local coordinates
    new_ss._compute_local_coords()

    # compute rho
    new_ss._compute_rho()

    # turn warnings back on
    if quiet:
        _logger.setLevel(logging.ERROR)

    if verbose:
        _logger.info(f"Disulfide {ssbond_name} initialized.")

    return new_ss


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
        SSList = SSList.filter_by_ca_distance(cutoff)
        delta = num_ssbonds - len(SSList)
        if delta:
            _logger.error(
                f"Filtered {num_ssbonds} -> {len(SSList)} SSBonds by Ca distance, {pdb_id}, delta is: {delta}"
            )
        num_ssbonds = len(SSList)

    if sg_cutoff > 0:
        SSList = SSList.filter_by_sg_distance(sg_cutoff)
        delta = num_ssbonds - len(SSList)
        if delta:
            _logger.error(
                f"Filtered {num_ssbonds} -> {len(SSList)} SSBonds by Sγ distance, {pdb_id}, delta is: {delta}"
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
