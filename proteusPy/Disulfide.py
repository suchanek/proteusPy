"""
This module, *Disulfide*, is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
It represents the core of the current implementation of *proteusPy*.

This work is based on the original C/C++ implementation by Eric G. Suchanek. \n
Author: Eric G. Suchanek, PhD
Last Modification: 2025-01-02 12:55:02
"""

# pylint: disable=W1203 # use of print
# pylint: disable=C0103 # invalid name
# pylint: disable=C0301 # line too long
# pylint: disable=W0212 # access to protected member
# pylint: disable=W0612 # unused variable
# pylint: disable=W0613 # unused argument

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

import copy
import logging
import math
import warnings
from math import cos

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

# import proteusPy
from proteusPy.atoms import (
    ATOM_COLORS,
    ATOM_RADII_COVALENT,
    ATOM_RADII_CPK,
    BOND_COLOR,
    BOND_RADIUS,
    BS_SCALE,
    FONTSIZE,
    SPEC_POWER,
    SPECULARITY,
)
from proteusPy.DisulfideExceptions import (
    DisulfideConstructionException,
    DisulfideConstructionWarning,
    ProteusPyWarning,
)
from proteusPy.DisulfideList import DisulfideList
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import _ANG_INIT, _FLOAT_INIT, WINSIZE
from proteusPy.Residue import build_residue
from proteusPy.ssparser import (
    get_phipsi_atoms_coordinates,
    get_residue_atoms_coordinates,
)
from proteusPy.turtle3D import ORIENT_SIDECHAIN, Turtle3D
from proteusPy.utility import get_theme
from proteusPy.vector3D import (
    Vector3D,
    calc_dihedral,
    calculate_bond_angle,
    distance3d,
    rms_difference,
)

np.set_printoptions(suppress=True)
pv.global_theme.color = "white"

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# columns for the torsions file dataframe.
Torsion_DF_Cols = [
    "source",
    "ss_id",
    "proximal",
    "distal",
    "chi1",
    "chi2",
    "chi3",
    "chi4",
    "chi5",
    "energy",
    "ca_distance",
    "cb_distance",
    "sg_distance",
    "phi_prox",
    "psi_prox",
    "phi_dist",
    "psi_dist",
    "torsion_length",
    "rho",
]

ORIGIN = Vector3D(0.0, 0.0, 0.0)

_logger = create_logger(__name__)
_logger.setLevel(logging.ERROR)


# class for the Disulfide bond
class Disulfide:
    r"""
    This class provides a Python object and methods representing a physical disulfide bond
    either extracted from the RCSB protein databank or built using the
    [proteusPy.Turtle3D](turtle3D.html) class. The disulfide bond is an important
    intramolecular stabilizing structural element and is characterized by:

    * Atomic coordinates for the atoms N, Cα, Cβ, C', Sγ for both residues.
    These are stored as both raw atomic coordinates as read from the RCSB file
    and internal local coordinates.
    * The dihedral angles Χ1 - Χ5 for the disulfide bond
    * A name, by default {pdb_id}{prox_resnumb}{prox_chain}_{distal_resnum}{distal_chain}
    * Proximal residue number
    * Distal residue number
    * Approximate bond torsional energy (kcal/mol):

    $$
    E_{kcal/mol} \\approx 2.0 * cos(3.0 * \\chi_{1}) + cos(3.0 * \\chi_{5}) + cos(3.0 * \\chi_{2}) +
    $$
    $$
    cos(3.0 * \chi_{4}) + 3.5 * cos(2.0 * \chi_{3}) + 0.6 * cos(3.0 * \chi_{3}) + 10.1
    $$

    The equation embodies the typical 3-fold rotation barriers associated with single bonds,
    (Χ1, Χ5, Χ2, Χ4) and a high 2-fold barrier for Χ3, resulting from the partial double bond
    character of the S-S bond. This property leads to two major disulfide families, characterized
    by the sign of Χ3. *Left-handed* disulfides have Χ3 < 0° and *right-handed* disulfides have
    Χ3 > 0°. Within this breakdown there are numerous subfamilies, broadly known as the *hook*,
    *spiral* and *staple*. These are under characgterization.

    * Euclidean length of the dihedral angles (degrees) defined as:
    $$\sqrt(\chi_{1}^{2} + \chi_{2}^{2} + \chi_{3}^{2} + \chi_{4}^{2} + \chi_{5}^{2})$$
    * Cα - Cα distance (Å)
    * Cβ - Cβ distance (Å)
    * The previous C' and next N for both the proximal and distal residues. These are needed
    to calculate the backbone dihedral angles Φ and Ψ.
    * Backbone dihedral angles Φ and Ψ, when possible. Not all structures are complete and
    in those cases the atoms needed may be undefined. In this case the Φ and Ψ angles are set
    to -180°.

    The class also provides a rendering capabilities using the excellent [PyVista](https://pyvista.org)
    library, and can display disulfides interactively in a variety of display styles:
    * 'sb' - Split Bonds style - bonds colored by their atom type
    * 'bs' - Ball and Stick style - split bond coloring with small atoms
    * 'pd' - Proximal/Distal style - bonds colored *Red* for proximal residue and *Green* for
    the distal residue.
    * 'cpk' - CPK style rendering, colored by atom type:
        * Carbon   - Grey
        * Nitrogen - Blue
        * Sulfur   - Yellow
        * Oxygen   - Red
        * Hydrogen - White

    Individual renderings can be saved to a file, and animations created.
    """

    def __init__(
        self,
        name: str = "SSBOND",
        proximal: int = -1,
        distal: int = -1,
        proximal_chain: str = "A",
        distal_chain: str = "A",
        pdb_id: str = "1egs",
        quiet: bool = True,
        torsions: list = None,
    ) -> None:
        """
        Initialize the class to defined internal values. If torsions are provided, the
        Disulfide object is built using the torsions and initialized.

        :param name: Disulfide name, by default "SSBOND"
        :param proximal: Proximal residue number, by default -1
        :param distal: Distal residue number, by default -1
        :param proximal_chain: Chain identifier for the proximal residue, by default "A"
        :param distal_chain: Chain identifier for the distal residue, by default "A"
        :param pdb_id: PDB identifier, by default "1egs"
        :param quiet: If True, suppress output, by default True
        :param torsions: List of torsion angles, by default None
        """
        self.name = name
        self.proximal = proximal
        self.distal = distal
        self.energy = _FLOAT_INIT
        self.proximal_chain = proximal_chain
        self.distal_chain = distal_chain
        self.pdb_id = pdb_id
        self.quiet = quiet
        self.proximal_secondary = "Nosecondary"
        self.distal_secondary = "Nosecondary"
        self.ca_distance = _FLOAT_INIT
        self.cb_distance = _FLOAT_INIT
        self.sg_distance = _FLOAT_INIT
        self.torsion_array = np.array(
            (_ANG_INIT, _ANG_INIT, _ANG_INIT, _ANG_INIT, _ANG_INIT)
        )
        self.phiprox = _ANG_INIT
        self.psiprox = _ANG_INIT
        self.phidist = _ANG_INIT
        self.psidist = _ANG_INIT

        # global coordinates for the Disulfide, typically as
        # returned from the PDB file

        self.n_prox = ORIGIN
        self.ca_prox = ORIGIN
        self.c_prox = ORIGIN
        self.o_prox = ORIGIN
        self.cb_prox = ORIGIN
        self.sg_prox = ORIGIN
        self.sg_dist = ORIGIN
        self.cb_dist = ORIGIN
        self.ca_dist = ORIGIN
        self.n_dist = ORIGIN
        self.c_dist = ORIGIN
        self.o_dist = ORIGIN

        # set when we can't find previous or next prox or distal
        # C' or N atoms.
        self.missing_atoms = False
        self.modelled = False
        self.resolution = -1.0

        # need these to calculate backbone dihedral angles
        self.c_prev_prox = ORIGIN
        self.n_next_prox = ORIGIN
        self.c_prev_dist = ORIGIN
        self.n_next_dist = ORIGIN

        # local coordinates for the Disulfide, computed using the Turtle3D in
        # Orientation #1. these are generally private.

        self._n_prox = ORIGIN
        self._ca_prox = ORIGIN
        self._c_prox = ORIGIN
        self._o_prox = ORIGIN
        self._cb_prox = ORIGIN
        self._sg_prox = ORIGIN
        self._sg_dist = ORIGIN
        self._cb_dist = ORIGIN
        self._ca_dist = ORIGIN
        self._n_dist = ORIGIN
        self._c_dist = ORIGIN
        self._o_dist = ORIGIN

        # need these to calculate backbone dihedral angles
        self._c_prev_prox = ORIGIN
        self._n_next_prox = ORIGIN
        self._c_prev_dist = ORIGIN
        self._n_next_dist = ORIGIN

        # Dihedral angles for the disulfide bond itself, set to _ANG_INIT
        self.chi1 = _ANG_INIT
        self.chi2 = _ANG_INIT
        self.chi3 = _ANG_INIT
        self.chi4 = _ANG_INIT
        self.chi5 = _ANG_INIT
        self._rho = _ANG_INIT  # new dihedral angle: Nprox - Ca_prox - Ca_dist - N_dist

        self.torsion_length = _FLOAT_INIT

        if torsions is not None and len(torsions) == 5:
            # computes energy, torsion length and rho
            self.dihedrals = torsions
            self.build_yourself()

    # comparison operators, used for sorting. keyed to SS bond energy
    def __lt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy < other.energy
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Disulfide):
            return self.energy <= other.energy
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy > other.energy
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Disulfide):
            return self.energy >= other.energy
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Disulfide):
            return (
                math.isclose(self.torsion_length, other.torsion_length, rel_tol=1e-1)
                and self.proximal == other.proximal
                and self.distal == other.distal
            )
        return False

    def __ne__(self, other):
        if isinstance(other, Disulfide):
            return self.proximal != other.proximal or self.distal != other.distal
        return NotImplemented

    def __repr__(self):
        """
        Representation for the Disulfide class
        """
        s1 = self.repr_ss_info()
        res = f"{s1}>"
        return res

    def _draw_bonds(
        self,
        pvp,
        coords,
        bond_radius=BOND_RADIUS,
        style="sb",
        bcolor=BOND_COLOR,
        all_atoms=True,
        res=100,
    ):
        """
        Generate the appropriate pyVista cylinder objects to represent
        a particular disulfide bond. This utilizes a connection table
        for the starting and ending atoms and a color table for the
        bond colors. Used internally.

        :param pvp: input plotter object to be updated
        :param bradius: bond radius
        :param style: bond style. One of sb, plain, pd
        :param bcolor: pyvista color
        :param missing: True if atoms are missing, False othersie
        :param all_atoms: True if rendering O, False if only backbone rendered

        :return pvp: Updated Plotter object.

        """
        _bond_conn = np.array(
            [
                [0, 1],  # n-ca
                [1, 2],  # ca-c
                [2, 3],  # c-o
                [1, 4],  # ca-cb
                [4, 5],  # cb-sg
                [6, 7],  # n-ca
                [7, 8],  # ca-c
                [8, 9],  # c-o
                [7, 10],  # ca-cb
                [10, 11],  # cb-sg
                [5, 11],  # sg -sg
                [12, 0],  # cprev_prox-n
                [2, 13],  # c-nnext_prox
                [14, 6],  # cprev_dist-n_dist
                [8, 15],  # c-nnext_dist
            ]
        )

        # modeled disulfides only have backbone atoms since
        # phi and psi are undefined, which makes the carbonyl
        # oxygen (O) undefined as well. Their previous and next N
        # are also undefined.

        missing = self.missing_atoms
        bradius = bond_radius

        _bond_conn_backbone = np.array(
            [
                [0, 1],  # n-ca
                [1, 2],  # ca-c
                [1, 4],  # ca-cb
                [4, 5],  # cb-sg
                [6, 7],  # n-ca
                [7, 8],  # ca-c
                [7, 10],  # ca-cb
                [10, 11],  # cb-sg
                [5, 11],  # sg -sg
            ]
        )

        # colors for the bonds. Index into ATOM_COLORS array
        _bond_split_colors = np.array(
            [
                ("N", "C"),
                ("C", "C"),
                ("C", "O"),
                ("C", "C"),
                ("C", "SG"),
                ("N", "C"),
                ("C", "C"),
                ("C", "O"),
                ("C", "C"),
                ("C", "SG"),
                ("SG", "SG"),
                # prev and next C-N bonds - color by atom Z
                ("C", "N"),
                ("C", "N"),
                ("C", "N"),
                ("C", "N"),
            ]
        )

        _bond_split_colors_backbone = np.array(
            [
                ("N", "C"),
                ("C", "C"),
                ("C", "C"),
                ("C", "SG"),
                ("N", "C"),
                ("C", "C"),
                ("C", "C"),
                ("C", "SG"),
                ("SG", "SG"),
            ]
        )
        # work through connectivity and colors
        orig_col = dest_col = bcolor

        if all_atoms:
            bond_conn = _bond_conn
            bond_split_colors = _bond_split_colors
        else:
            bond_conn = _bond_conn_backbone
            bond_split_colors = _bond_split_colors_backbone

        for i, bond in enumerate(bond_conn):
            if all_atoms:
                if i > 10 and missing:  # skip missing atoms
                    continue

            orig, dest = bond
            col = bond_split_colors[i]

            # get the coords
            prox_pos = coords[orig]
            distal_pos = coords[dest]

            # compute a direction vector
            direction = distal_pos - prox_pos

            # compute vector length. divide by 2 since split bond
            height = math.dist(prox_pos, distal_pos) / 2.0

            # the cylinder origins are actually in the
            # middle so we translate

            origin = prox_pos + 0.5 * direction  # for a single plain bond
            origin1 = prox_pos + 0.25 * direction
            origin2 = prox_pos + 0.75 * direction

            if style == "plain":
                orig_col = dest_col = bcolor

            # proximal-distal red/green coloring
            elif style == "pd":
                if i <= 4 or i == 11 or i == 12:
                    orig_col = dest_col = "red"
                else:
                    orig_col = dest_col = "green"
                if i == 10:
                    orig_col = dest_col = "yellow"
            else:
                orig_col = ATOM_COLORS[col[0]]
                dest_col = ATOM_COLORS[col[1]]

            if i >= 11:  # prev and next residue atoms for phi/psi calcs
                bradius = bradius * 0.5  # make smaller to distinguish

            cap1 = pv.Sphere(center=prox_pos, radius=bradius)
            cap2 = pv.Sphere(center=distal_pos, radius=bradius)

            if style == "plain":
                cyl = pv.Cylinder(
                    origin, direction, radius=bradius, height=height * 2.0
                )
                pvp.add_mesh(cyl, color=orig_col)
            else:
                cyl1 = pv.Cylinder(
                    origin1,
                    direction,
                    radius=bradius,
                    height=height,
                    capping=False,
                    resolution=res,
                )
                cyl2 = pv.Cylinder(
                    origin2,
                    direction,
                    radius=bradius,
                    height=height,
                    capping=False,
                    resolution=res,
                )
                pvp.add_mesh(cyl1, color=orig_col)
                pvp.add_mesh(cyl2, color=dest_col)

            pvp.add_mesh(cap1, color=orig_col)
            pvp.add_mesh(cap2, color=dest_col)

        return pvp  # end draw_bonds

    def _render(
        self,
        pvplot: pv.Plotter,
        style="bs",
        plain=False,
        bondcolor=BOND_COLOR,
        bs_scale=BS_SCALE,
        spec=SPECULARITY,
        specpow=SPEC_POWER,
        translate=True,
        bond_radius=BOND_RADIUS,
        res=100,
    ):
        """
        Update the passed pyVista plotter() object with the mesh data for the
        input Disulfide Bond. Used internally.

        :param pvplot: pyvista.Plotter object
        :type pvplot: pv.Plotter

        :param style: Rendering style, by default 'bs'. One of 'bs', 'st', 'cpk'. Render as \
            CPK, ball-and-stick or stick. Bonds are colored by atom color, unless \
            'plain' is specified.
        :type style: str, optional

        :param plain: Used internally, by default False
        :type plain: bool, optional

        :param bondcolor: pyVista color name, optional bond color for simple bonds, by default BOND_COLOR
        :type bondcolor: str, optional

        :param bs_scale: Scale factor (0-1) to reduce the atom sizes for ball and stick, by default BS_SCALE
        :type bs_scale: float, optional

        :param spec: Specularity (0-1), where 1 is totally smooth and 0 is rough, by default SPECULARITY
        :type spec: float, optional

        :param specpow: Exponent used for specularity calculations, by default SPEC_POWER
        :type specpow: int, optional

        :param translate: Flag used internally to indicate if we should translate \
            the disulfide to its geometric center of mass, by default True.
        :type translate: bool, optional

        :returns: Updated pv.Plotter object with atoms and bonds.
        :rtype: pv.Plotter
        """

        def add_atoms(pvp, coords, atoms, radii, colors, spec, specpow):
            for i, atom in enumerate(atoms):
                rad = radii[atom]
                if style == "bs" and i > 11:
                    rad *= 0.75
                pvp.add_mesh(
                    pv.Sphere(center=coords[i], radius=rad),
                    color=colors[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )

        def draw_bonds(pvp, coords, style, all_atoms, bond_radius, bondcolor=None):
            return self._draw_bonds(
                pvp,
                coords,
                style=style,
                all_atoms=all_atoms,
                bond_radius=bond_radius,
                bcolor=bondcolor,
            )

        model = self.modelled
        coords = self.internal_coords
        if translate:
            coords -= self.cofmass

        atoms = (
            "N",
            "C",
            "C",
            "O",
            "C",
            "SG",
            "N",
            "C",
            "C",
            "O",
            "C",
            "SG",
            "C",
            "N",
            "C",
            "N",
        )
        pvp = pvplot
        all_atoms = not model

        if style == "cpk":
            add_atoms(pvp, coords, atoms, ATOM_RADII_CPK, ATOM_COLORS, spec, specpow)
        elif style == "cov":
            add_atoms(
                pvp, coords, atoms, ATOM_RADII_COVALENT, ATOM_COLORS, spec, specpow
            )
        elif style == "bs":
            add_atoms(
                pvp,
                coords,
                atoms,
                {atom: ATOM_RADII_CPK[atom] * bs_scale for atom in atoms},
                ATOM_COLORS,
                spec,
                specpow,
            )
            pvp = draw_bonds(pvp, coords, "bs", all_atoms, bond_radius)
        elif style in ["sb", "pd", "plain"]:
            pvp = draw_bonds(
                pvp,
                coords,
                style,
                all_atoms,
                bond_radius,
                bondcolor if style == "plain" else None,
            )

        return pvp

    def _plot(
        self,
        pvplot,
        style="bs",
        plain=False,
        bondcolor=BOND_COLOR,
        bs_scale=BS_SCALE,
        spec=SPECULARITY,
        specpow=SPEC_POWER,
        translate=True,
        bond_radius=BOND_RADIUS,
        res=100,
    ):
        """
            Update the passed pyVista plotter() object with the mesh data for the
            input Disulfide Bond. Used internally

            Parameters
            ----------
            pvplot : pv.Plotter
                pyvista.Plotter object

            style : str, optional
                Rendering style, by default 'bs'. One of 'bs', 'st', 'cpk', Render as \
                CPK, ball-and-stick or stick. Bonds are colored by atom color, unless \
                'plain' is specified.

            plain : bool, optional
                Used internally, by default False

            bondcolor : pyVista color name, optional bond color for simple bonds, by default BOND_COLOR

            bs_scale : float, optional
                scale factor (0-1) to reduce the atom sizes for ball and stick, by default BS_SCALE
            
            spec : float, optional
                specularity (0-1), where 1 is totally smooth and 0 is rough, by default SPECULARITY

            specpow : int, optional
                exponent used for specularity calculations, by default SPEC_POWER

            translate : bool, optional
                Flag used internally to indicate if we should translate \
                the disulfide to its geometric center of mass, by default True.

            Returns
            -------
            pv.Plotter
                Updated pv.Plotter object with atoms and bonds.
            """

        _bradius = bond_radius
        coords = self.internal_coords
        missing_atoms = self.missing_atoms
        clen = coords.shape[0]

        model = self.modelled
        if model:
            all_atoms = False
        else:
            all_atoms = True

        if translate:
            coords -= self.cofmass

        atoms = (
            "N",
            "C",
            "C",
            "O",
            "C",
            "SG",
            "N",
            "C",
            "C",
            "O",
            "C",
            "SG",
            "C",
            "N",
            "C",
            "N",
        )
        pvp = pvplot.copy()

        # bond connection table with atoms in the specific order shown above:
        # returned by ss.get_internal_coords()

        if style == "cpk":
            for i, atom in enumerate(atoms):
                rad = ATOM_RADII_CPK[atom]
                pvp.append(pv.Sphere(center=coords[i], radius=rad))

        elif style == "cov":
            for i, atom in enumerate(atoms):
                rad = ATOM_RADII_COVALENT[atom]
                pvp.append(pv.Sphere(center=coords[i], radius=rad))

        elif style == "bs":  # ball and stick
            for i, atom in enumerate(atoms):
                rad = ATOM_RADII_CPK[atom] * bs_scale
                if i > 11:
                    rad = rad * 0.75

                pvp.append(pv.Sphere(center=coords[i]))
            pvp = self._draw_bonds(pvp, coords, style="bs", all_atoms=all_atoms)

        else:
            pvp = self._draw_bonds(pvp, coords, style=style, all_atoms=all_atoms)

        return

    def _handle_SS_exception(self, message: str):
        """
        This method catches an exception that occurs in the Disulfide
        object (if quiet), or raises it again, this time adding the
        PDB line number to the error message. (private).

        :param message: Error message
        :raises DisulfideConstructionException: Fatal construction exception.

        """
        # message = "%s at line %i." % (message)
        message = f"{message}"

        if self.quiet:
            # just print a warning - some residues/atoms may be missing
            warnings.warn(
                f"DisulfideConstructionException: {message}\n",
                "Exception ignored.\n",
                "Some atoms may be missing in the data structure.",
                DisulfideConstructionWarning,
            )
        else:
            # exceptions are fatal - raise again with new message (including line nr)
            raise DisulfideConstructionException(message) from None

    @property
    def bond_angle_ideality(self):
        """
        Calculate all bond angles for a disulfide bond and compare them to idealized angles.

        :param np.ndarray atom_coordinates: Array containing coordinates of atoms in the order:
            N1, CA1, C1, O1, CB1, SG1, N2, CA2, C2, O2, CB2, SG2
        :return: RMS difference between calculated bond angles and idealized bond angles.
        :rtype: float
        """

        atom_coordinates = self.coords_array
        verbose = not self.quiet
        if verbose:
            _logger.setLevel(logging.INFO)

        idealized_angles = {
            ("N1", "CA1", "C1"): 111.0,
            ("N1", "CA1", "CB1"): 108.5,
            ("CA1", "CB1", "SG1"): 112.8,
            ("CB1", "SG1", "SG2"): 103.8,  # This angle is for the disulfide bond itself
            ("SG1", "SG2", "CB2"): 103.8,  # This angle is for the disulfide bond itself
            ("SG2", "CB2", "CA2"): 112.8,
            ("CB2", "CA2", "N2"): 108.5,
            ("N2", "CA2", "C2"): 111.0,
        }

        # List of triplets for which we need to calculate bond angles
        # I am omitting the proximal and distal backbone angle N, Ca, C
        # to focus on the disulfide bond angles themselves.
        angle_triplets = [
            ("N1", "CA1", "C1"),
            ("N1", "CA1", "CB1"),
            ("CA1", "CB1", "SG1"),
            ("CB1", "SG1", "SG2"),
            ("SG1", "SG2", "CB2"),
            ("SG2", "CB2", "CA2"),
            ("CB2", "CA2", "N2"),
            ("N2", "CA2", "C2"),
        ]

        atom_indices = {
            "N1": 0,
            "CA1": 1,
            "C1": 2,
            "CB1": 4,
            "SG1": 5,
            "SG2": 11,
            "CB2": 10,
            "CA2": 7,
            "N2": 6,
            "C2": 8,
        }

        calculated_angles = []
        for triplet in angle_triplets:
            a = atom_coordinates[atom_indices[triplet[0]]]
            b = atom_coordinates[atom_indices[triplet[1]]]
            c = atom_coordinates[atom_indices[triplet[2]]]
            ideal = idealized_angles[triplet]
            try:
                angle = calculate_bond_angle(a, b, c)
            except ValueError as e:
                print(f"Error calculating angle for atoms {triplet}: {e}")
                return None
            calculated_angles.append(angle)
            if verbose:
                _logger.info(
                    f"Calculated angle for atoms {triplet}: {angle:.2f}, Ideal angle: {ideal:.2f}"
                )

        # Convert idealized angles to a list
        idealized_angles_list = [
            idealized_angles[triplet] for triplet in angle_triplets
        ]

        # Calculate RMS difference
        rms_diff = rms_difference(
            np.array(calculated_angles), np.array(idealized_angles_list)
        )

        if verbose:
            _logger.info(f"RMS bond angle deviation:, {rms_diff:.2f}")

        return rms_diff

    @property
    def bond_length_ideality(self):
        """
        Calculate bond lengths for a disulfide bond and compare them to idealized angles.

        :param np.ndarray atom_coordinates: Array containing coordinates of atoms in the order:
            N1, CA1, C1, O1, CB1, SG1, N2, CA2, C2, O2, CB2, SG2
        :return: RMS difference between calculated bond angles and idealized bond angles.
        :rtype: float
        """

        atom_coordinates = self.coords_array
        verbose = not self.quiet
        if verbose:
            _logger.setLevel(logging.INFO)

        idealized_bonds = {
            ("N1", "CA1"): 1.46,
            ("CA1", "C1"): 1.52,
            ("CA1", "CB1"): 1.52,
            ("CB1", "SG1"): 1.86,
            ("SG1", "SG2"): 2.044,  # This angle is for the disulfide bond itself
            ("SG2", "CB2"): 1.86,
            ("CB2", "CA2"): 1.52,
            ("CA2", "C2"): 1.52,
            ("N2", "CA2"): 1.46,
        }

        # List of triplets for which we need to calculate bond angles
        # I am omitting the proximal and distal backbone angle N, Ca, C
        # to focus on the disulfide bond angles themselves.
        distance_pairs = [
            ("N1", "CA1"),
            ("CA1", "C1"),
            ("CA1", "CB1"),
            ("CB1", "SG1"),
            ("SG1", "SG2"),  # This angle is for the disulfide bond itself
            ("SG2", "CB2"),
            ("CB2", "CA2"),
            ("CA2", "C2"),
            ("N2", "CA2"),
        ]

        atom_indices = {
            "N1": 0,
            "CA1": 1,
            "C1": 2,
            "CB1": 4,
            "SG1": 5,
            "SG2": 11,
            "CB2": 10,
            "CA2": 7,
            "N2": 6,
            "C2": 8,
        }

        calculated_distances = []
        for pair in distance_pairs:
            a = atom_coordinates[atom_indices[pair[0]]]
            b = atom_coordinates[atom_indices[pair[1]]]
            ideal = idealized_bonds[pair]
            try:
                distance = math.dist(a, b)
            except ValueError as e:
                _logger.error(f"Error calculating bond length for atoms {pair}: {e}")
                return None
            calculated_distances.append(distance)
            if verbose:
                _logger.info(
                    f"Calculated distance for atoms {pair}: {distance:.2f}A, Ideal distance: {ideal:.2f}A"
                )

        # Convert idealized distances to a list
        idealized_distance_list = [idealized_bonds[pair] for pair in distance_pairs]

        # Calculate RMS difference
        rms_diff = rms_difference(
            np.array(calculated_distances), np.array(idealized_distance_list)
        )

        if verbose:
            _logger.info(
                f"RMS distance deviation from ideality for SS atoms: {rms_diff:.2f}"
            )

            # Reset logger level
            _logger.setLevel(logging.WARNING)

        return rms_diff

    @property
    def internal_coords_array(self):
        """
        Return an array of internal coordinates for the disulfide bond.

        This function collects the coordinates of the backbone atoms involved in the
        disulfide bond and returns them as a numpy array.

        :param self: The instance of the Disulfide class.
        :type self: Disulfide
        :return: A numpy array containing the coordinates of the atoms.
        :rtype: np.ndarray
        """
        coords = []
        coords.append(self._n_prox.get_array())
        coords.append(self._ca_prox.get_array())
        coords.append(self._c_prox.get_array())
        coords.append(self._o_prox.get_array())
        coords.append(self._cb_prox.get_array())
        coords.append(self._sg_prox.get_array())
        coords.append(self._n_dist.get_array())
        coords.append(self._ca_dist.get_array())
        coords.append(self._c_dist.get_array())
        coords.append(self._o_dist.get_array())
        coords.append(self._cb_dist.get_array())
        coords.append(self._sg_dist.get_array())

        return np.array(coords)

    @property
    def coords_array(self):
        """
        Return an array of coordinates for the disulfide bond.

        This function collects the coordinates of backbone atoms involved in the
        disulfide bond and returns them as a numpy array.

        :param self: The instance of the Disulfide class.
        :type self: Disulfide
        :return: A numpy array containing the coordinates of the atoms.
        :rtype: np.ndarray
        """
        coords = []
        coords.append(self.n_prox.get_array())
        coords.append(self.ca_prox.get_array())
        coords.append(self.c_prox.get_array())
        coords.append(self.o_prox.get_array())
        coords.append(self.cb_prox.get_array())
        coords.append(self.sg_prox.get_array())
        coords.append(self.n_dist.get_array())
        coords.append(self.ca_dist.get_array())
        coords.append(self.c_dist.get_array())
        coords.append(self.o_dist.get_array())
        coords.append(self.cb_dist.get_array())
        coords.append(self.sg_dist.get_array())

        return np.array(coords)

    @property
    def dihedrals(self) -> list:
        """
        Return a list containing the dihedral angles for the disulfide.

        """
        return [self.chi1, self.chi2, self.chi3, self.chi4, self.chi5]

    @dihedrals.setter
    def dihedrals(self, dihedrals: list) -> None:
        """
        Sets the disulfide dihedral angles to the inputs specified in the list and
        computes the torsional energy and length of the disulfide bond.

        :param dihedrals: list of dihedral angles.
        """
        self.chi1 = dihedrals[0]
        self.chi2 = dihedrals[1]
        self.chi3 = dihedrals[2]
        self.chi4 = dihedrals[3]
        self.chi5 = dihedrals[4]
        self.torsion_array = np.array(dihedrals)
        self._compute_torsional_energy()
        self._compute_torsion_length()
        self._compute_rho()

    def bounding_box(self) -> np.array:
        """
        Return the bounding box array for the given disulfide.

        Returns
        -------
        :return: np.array(3, 2): Array containing the min, max for X, Y, and Z respectively.
        Does not currently take the atom's radius into account.
        """
        coords = self.internal_coords

        xmin, ymin, zmin = coords.min(axis=0)
        xmax, ymax, zmax = coords.max(axis=0)

        res = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])

        return res

    def build_yourself(self) -> None:
        """
        Build a model Disulfide based its internal dihedral state
        Routine assumes turtle is in orientation #1 (at Ca, headed toward
        Cb, with N on left), builds disulfide, and updates the object's internal
        coordinates. It also adds the distal protein backbone,
        and computes the disulfide conformational energy.
        """
        self.build_model(self.chi1, self.chi2, self.chi3, self.chi4, self.chi5)

    def build_model(
        self, chi1: float, chi2: float, chi3: float, chi4: float, chi5: float
    ) -> None:
        """
        Build a model Disulfide based on the input dihedral angles.
        Routine assumes turtle is in orientation #1 (at Ca, headed toward
        Cb, with N on left), builds disulfide, and updates the object's internal
        coordinates. It also adds the distal protein backbone,
        and computes the disulfide conformational energy.

        :param chi1: Chi1 (degrees)
        :param chi2: Chi2 (degrees)
        :param chi3: Chi3 (degrees)
        :param chi4: Chi4 (degrees)
        :param chi5: Chi5 (degrees)

        Example:
        >>> from proteusPy.Disulfide import Disulfide
        >>> modss = Disulfide('model')
        >>> modss.build_model(-60, -60, -90, -60, -60)
        >>> modss.display(style='sb')
        """

        self.dihedrals = [chi1, chi2, chi3, chi4, chi5]
        self.proximal = 1
        self.distal = 2

        tmp = Turtle3D("tmp")
        tmp.Orientation = 1

        n = ORIGIN
        ca = ORIGIN
        cb = ORIGIN
        c = ORIGIN

        self.ca_prox = tmp._position
        tmp.schain_to_bbone()
        n, ca, cb, c = build_residue(tmp)

        self.n_prox = n
        self.ca_prox = ca
        self.c_prox = c
        self.cb_prox = cb

        tmp.bbone_to_schain()
        tmp.move(1.53)
        tmp.roll(self.chi1)
        tmp.yaw(112.8)
        self.cb_prox = Vector3D(tmp._position)

        tmp.move(1.86)
        tmp.roll(self.chi2)
        tmp.yaw(103.8)
        self.sg_prox = Vector3D(tmp._position)

        tmp.move(2.044)
        tmp.roll(self.chi3)
        tmp.yaw(103.8)
        self.sg_dist = Vector3D(tmp._position)

        tmp.move(1.86)
        tmp.roll(self.chi4)
        tmp.yaw(112.8)
        self.cb_dist = Vector3D(tmp._position)

        tmp.move(1.53)
        tmp.roll(self.chi5)
        tmp.pitch(180.0)

        tmp.schain_to_bbone()

        n, ca, cb, c = build_residue(tmp)

        self.n_dist = n
        self.ca_dist = ca
        self.c_dist = c
        self._compute_torsional_energy()
        self._compute_local_coords()
        self._compute_torsion_length()
        self._compute_rho()
        self.ca_distance = distance3d(self.ca_prox, self.ca_dist)
        self.cb_distance = distance3d(self.cb_prox, self.cb_dist)
        self.sg_distance = distance3d(self.sg_prox, self.sg_dist)
        self.torsion_array = np.array([chi1, chi2, chi3, chi4, chi5])
        self.missing_atoms = True
        self.modelled = True

    @property
    def cofmass(self) -> np.array:
        """
        Return the geometric center of mass for the internal coordinates of
        the given Disulfide. Missing atoms are not included.

        :return: 3D array for the geometric center of mass
        """

        res = self.internal_coords.mean(axis=0)
        return res

    @property
    def coord_cofmass(self) -> np.array:
        """
        Return the geometric center of mass for the global coordinates of
        the given Disulfide. Missing atoms are not included.

        :return: 3D array for the geometric center of mass
        """

        res = self.coords.mean(axis=0)
        return res

    def copy(self):
        """
        Copy the Disulfide.

        :return: A copy of self.
        """
        return copy.deepcopy(self)

    def _compute_local_coords(self) -> None:
        """
        Compute the internal coordinates for a properly initialized Disulfide Object.

        :param self: SS initialized Disulfide object
        :returns: None, modifies internal state of the input
        """

        turt = Turtle3D("tmp")
        # get the coordinates as np.array for Turtle3D use.
        cpp = self.c_prev_prox.get_array()
        nnp = self.n_next_prox.get_array()

        n = self.n_prox.get_array()
        ca = self.ca_prox.get_array()
        c = self.c_prox.get_array()
        cb = self.cb_prox.get_array()
        o = self.o_prox.get_array()
        sg = self.sg_prox.get_array()

        sg2 = self.sg_dist.get_array()
        cb2 = self.cb_dist.get_array()
        ca2 = self.ca_dist.get_array()
        c2 = self.c_dist.get_array()
        n2 = self.n_dist.get_array()
        o2 = self.o_dist.get_array()

        cpd = self.c_prev_dist.get_array()
        nnd = self.n_next_dist.get_array()

        turt.orient_from_backbone(n, ca, c, cb, ORIENT_SIDECHAIN)

        # internal (local) coordinates, stored as Vector objects
        # to_local returns np.array objects

        self._n_prox = Vector3D(turt.to_local(n))
        self._ca_prox = Vector3D(turt.to_local(ca))
        self._c_prox = Vector3D(turt.to_local(c))
        self._o_prox = Vector3D(turt.to_local(o))
        self._cb_prox = Vector3D(turt.to_local(cb))
        self._sg_prox = Vector3D(turt.to_local(sg))

        self._c_prev_prox = Vector3D(turt.to_local(cpp))
        self._n_next_prox = Vector3D(turt.to_local(nnp))
        self._c_prev_dist = Vector3D(turt.to_local(cpd))
        self._n_next_dist = Vector3D(turt.to_local(nnd))

        self._n_dist = Vector3D(turt.to_local(n2))
        self._ca_dist = Vector3D(turt.to_local(ca2))
        self._c_dist = Vector3D(turt.to_local(c2))
        self._o_dist = Vector3D(turt.to_local(o2))
        self._cb_dist = Vector3D(turt.to_local(cb2))
        self._sg_dist = Vector3D(turt.to_local(sg2))

    def _compute_torsional_energy(self) -> float:
        """
        Compute the approximate torsional energy for the Disulfide's
        conformation and sets its internal state.

        :return: Energy (kcal/mol)
        """
        # @TODO find citation for the ss bond energy calculation

        def torad(deg):
            return np.radians(deg)

        chi1 = self.chi1
        chi2 = self.chi2
        chi3 = self.chi3
        chi4 = self.chi4
        chi5 = self.chi5

        energy = 2.0 * (cos(torad(3.0 * chi1)) + cos(torad(3.0 * chi5)))
        energy += cos(torad(3.0 * chi2)) + cos(torad(3.0 * chi4))
        energy += 3.5 * cos(torad(2.0 * chi3)) + 0.6 * cos(torad(3.0 * chi3)) + 10.1

        self.energy = energy
        return energy

    def display(
        self, single=True, style="sb", light="Auto", shadows=False, winsize=WINSIZE
    ) -> None:
        """
        Display the Disulfide bond in the specific rendering style.

        :param single: Display the bond in a single panel in the specific style.
        :param style:  Rendering style: One of:
            * 'sb' - split bonds
            * 'bs' - ball and stick
            * 'cpk' - CPK style
            * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            * 'plain' - boring single color
        :param light: If True, light background, if False, dark

        Example:
        >>> import proteusPy
        >>> from proteusPy.Disulfide import Disulfide
        >>> from proteusPy.DisulfideLoader import DisulfideLoader, Load_PDB_SS

        >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)
        >>> ss = PDB_SS[0]
        >>> ss.display(style='cpk')
        >>> ss.screenshot(style='bs', fname='proteus_logo_sb.png')
        """
        src = self.pdb_id
        enrg = self.energy

        title = f"{src}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol. Cα: {self.ca_distance:.2f} Å Cβ: {self.cb_distance:.2f} Å, Sg: {self.sg_distance:.2f} Å Tors: {self.torsion_length:.2f}°"

        if light == "light":
            pv.set_plot_theme("document")
        elif light == "dark":
            pv.set_plot_theme("dark")
        else:
            _theme = get_theme()
            if _theme == "light":
                pv.set_plot_theme("document")
            elif _theme == "dark":
                pv.set_plot_theme("dark")
                _logger.info("Dark mode detected.")
            else:
                pv.set_plot_theme("document")

        if single:
            _pl = pv.Plotter(window_size=winsize)
            _pl.add_title(title=title, font_size=FONTSIZE)
            _pl.enable_anti_aliasing("msaa")
            # _pl.add_camera_orientation_widget()

            self._render(
                _pl,
                style=style,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )
            _pl.reset_camera()
            if shadows:
                _pl.enable_shadows()
            _pl.show()

        else:
            pl = pv.Plotter(window_size=winsize, shape=(2, 2))
            pl.subplot(0, 0)

            pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing("msaa")

            # pl.add_camera_orientation_widget()

            self._render(
                pl,
                style="cpk",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(0, 1)
            pl.add_title(title=title, font_size=FONTSIZE)

            self._render(
                pl,
                style="bs",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(1, 0)
            pl.add_title(title=title, font_size=FONTSIZE)

            self._render(
                pl,
                style="sb",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(1, 1)
            pl.add_title(title=title, font_size=FONTSIZE)

            self._render(
                pl,
                style="pd",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.link_views()
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()
            pl.show()
        return

    @property
    def TorsionEnergy(self) -> float:
        """
        Return the energy of the Disulfide bond.
        """
        return self._compute_torsional_energy()

    @property
    def TorsionLength(self) -> float:
        """
        Return the energy of the Disulfide bond.
        """
        return self._compute_torsion_length()

    def Distance_neighbors(self, others: DisulfideList, cutoff: float) -> DisulfideList:
        """
        Return list of Disulfides whose RMS atomic distance is within
        the cutoff (Å) in the others list.

        :param others: DisulfideList to search
        :param cutoff: Distance cutoff (Å)
        :return: DisulfideList within the cutoff
        """

        res = [ss.copy() for ss in others if self.Distance_RMS(ss) < cutoff]
        return DisulfideList(res, "neighbors")

    def Distance_RMS(self, other) -> float:
        """
        Calculate the RMS distance between the internal coordinates of self and another Disulfide.
        :param other: Comparison Disulfide
        :return: RMS distance (Å)
        """

        # Get internal coordinates of both objects
        ic1 = self.internal_coords
        ic2 = other._internal_coords()

        # Compute the sum of squared differences between corresponding internal coordinates
        totsq = sum(math.dist(p1, p2) ** 2 for p1, p2 in zip(ic1, ic2))

        # Compute the mean of the squared distances
        totsq /= len(ic1)

        # Take the square root of the mean to get the RMS distance
        return math.sqrt(totsq)

    def Torsion_RMS(self, other) -> float:
        """
        Calculate the RMS distance between the dihedral angles of self and another Disulfide.
        :param other: Comparison Disulfide
        :return: RMS distance (deg).
        """

        # Get internal coordinates of both objects
        ic1 = self.torsion_array
        ic2 = other.torsion_array

        # Compute the sum of squared differences between corresponding internal coordinates
        totsq = sum((p1 - p2) ** 2 for p1, p2 in zip(ic1, ic2))
        # Compute the mean of the squared distances
        totsq /= len(ic1)

        # Take the square root of the mean to get the RMS distance
        return math.sqrt(totsq)

    def get_chains(self) -> tuple:
        """
        Return the proximal and distal chain IDs for the Disulfide.

        :return: tuple (proximal, distal) chain IDs
        """
        prox = self.proximal_chain
        dist = self.distal_chain
        return tuple(prox, dist)

    def get_full_id(self) -> tuple:
        """
        Return the Disulfide full IDs (Used with BIO.PDB)

        :return: Disulfide full IDs
        """
        return (self.proximal, self.distal)

    @property
    def internal_coords(self) -> np.array:
        """
        Return the internal coordinates for the Disulfide.

        :return: Array containing the coordinates, [16][3].
        """
        return self._internal_coords()

    def _internal_coords(self) -> np.array:
        """
        Return the internal coordinates for the Disulfide.
        If there are missing atoms the extra atoms for the proximal
        and distal N and C are set to [0,0,0]. This is needed for the center of
        mass calculations, used when rendering.

        :return: Array containing the coordinates, [16][3].
        """

        # if we don't have the prior and next atoms we initialize those
        # atoms to the origin so as to not effect the center of mass calculations
        if self.missing_atoms:
            res_array = np.array(
                (
                    self._n_prox.get_array(),
                    self._ca_prox.get_array(),
                    self._c_prox.get_array(),
                    self._o_prox.get_array(),
                    self._cb_prox.get_array(),
                    self._sg_prox.get_array(),
                    self._n_dist.get_array(),
                    self._ca_dist.get_array(),
                    self._c_dist.get_array(),
                    self._o_dist.get_array(),
                    self._cb_dist.get_array(),
                    self._sg_dist.get_array(),
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                )
            )
        else:
            res_array = np.array(
                (
                    self._n_prox.get_array(),
                    self._ca_prox.get_array(),
                    self._c_prox.get_array(),
                    self._o_prox.get_array(),
                    self._cb_prox.get_array(),
                    self._sg_prox.get_array(),
                    self._n_dist.get_array(),
                    self._ca_dist.get_array(),
                    self._c_dist.get_array(),
                    self._o_dist.get_array(),
                    self._cb_dist.get_array(),
                    self._sg_dist.get_array(),
                    self._c_prev_prox.get_array(),
                    self._n_next_prox.get_array(),
                    self._c_prev_dist.get_array(),
                    self._n_next_dist.get_array(),
                )
            )
        return res_array

    @property
    def coords(self) -> np.array:
        """
        Return the coordinates for the Disulfide as an array.

        :return: Array containing the coordinates, [16][3].
        """
        return self._coords()

    def _coords(self) -> np.array:
        """
        Return the coordinates for the Disulfide as an array.
        If there are missing atoms the extra atoms for the proximal
        and distal N and C are set to [0,0,0]. This is needed for the center of
        mass calculations, used when rendering.

        :return: Array containing the coordinates, [16][3].
        """

        # if we don't have the prior and next atoms we initialize those
        # atoms to the origin so as to not effect the center of mass calculations
        if self.missing_atoms:
            res_array = np.array(
                (
                    self.n_prox.get_array(),
                    self.ca_prox.get_array(),
                    self.c_prox.get_array(),
                    self.o_prox.get_array(),
                    self.cb_prox.get_array(),
                    self.sg_prox.get_array(),
                    self.n_dist.get_array(),
                    self.ca_dist.get_array(),
                    self.c_dist.get_array(),
                    self.o_dist.get_array(),
                    self.cb_dist.get_array(),
                    self.sg_dist.get_array(),
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                )
            )
        else:
            res_array = np.array(
                (
                    self.n_prox.get_array(),
                    self.ca_prox.get_array(),
                    self.c_prox.get_array(),
                    self.o_prox.get_array(),
                    self.cb_prox.get_array(),
                    self.sg_prox.get_array(),
                    self.n_dist.get_array(),
                    self.ca_dist.get_array(),
                    self.c_dist.get_array(),
                    self.o_dist.get_array(),
                    self.cb_dist.get_array(),
                    self.sg_dist.get_array(),
                    self.c_prev_prox.get_array(),
                    self.n_next_prox.get_array(),
                    self.c_prev_dist.get_array(),
                    self.n_next_dist.get_array(),
                )
            )
        return res_array

    def internal_coords_res(self, resnumb) -> np.array:
        """
        Return the internal coordinates for the Disulfide. Missing atoms are not included.

        :return: Array containing the coordinates, [12][3].
        """
        return self._internal_coords_res(resnumb)

    def _internal_coords_res(self, resnumb) -> np.array:
        """
        Return the internal coordinates for the internal coordinates of
        the given Disulfide. Missing atoms are not included.

        :param resnumb: Residue number for disulfide
        :raises DisulfideConstructionWarning: Warning raised if the residue number is invalid
        :return: Array containing the internal coordinates for the disulfide
        """
        res_array = np.zeros(shape=(6, 3))

        if resnumb == self.proximal:
            res_array = np.array(
                (
                    self._n_prox.get_array(),
                    self._ca_prox.get_array(),
                    self._c_prox.get_array(),
                    self._o_prox.get_array(),
                    self._cb_prox.get_array(),
                    self._sg_prox.get_array(),
                )
            )
            return res_array

        elif resnumb == self.distal:
            res_array = np.array(
                (
                    self._n_dist.get_array(),
                    self._ca_dist.get_array(),
                    self._c_dist.get_array(),
                    self._o_dist.get_array(),
                    self._cb_dist.get_array(),
                    self._sg_dist.get_array(),
                )
            )
            return res_array
        else:
            mess = f"-> Disulfide._internal_coords(): Invalid argument. \
             Unable to find residue: {resnumb} "
            raise DisulfideConstructionWarning(mess)

    def make_movie(
        self, style="sb", fname="ssbond.mp4", verbose=False, steps=360
    ) -> None:
        """
        Create an animation for ```self``` rotating one revolution about the Y axis,
        in the given ```style```, saving to ```filename```.

        :param style: Rendering style, defaults to 'sb', one of:
        * 'sb' - split bonds
        * 'bs' - ball and stick
        * 'cpk' - CPK style
        * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
        * 'plain' - boring single color

        :param fname: Output filename, defaults to ```ssbond.mp4```
        :param verbose: Verbosity, defaults to False
        :param steps: Number of steps for one complete rotation, defaults to 360.
        """
        src = self.pdb_id
        name = self.name
        enrg = self.energy

        title = f"{src} {name}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol, Cα: {self.ca_distance:.2f} Å, Tors: {self.torsion_length:.2f}"

        if verbose:
            print(f"Rendering animation to {fname}...")

        pl = pv.Plotter(window_size=WINSIZE, off_screen=True)
        pl.open_movie(fname)
        path = pl.generate_orbital_path(n_points=steps)

        #
        # pl.add_title(title=title, font_size=FONTSIZE)
        pl.enable_anti_aliasing("msaa")
        pl = self._render(
            pl,
            style=style,
            bondcolor=BOND_COLOR,
            bs_scale=BS_SCALE,
            spec=SPECULARITY,
            specpow=SPEC_POWER,
        )
        pl.reset_camera()
        pl.orbit_on_path(path, write_frames=True)
        pl.close()

        if verbose:
            print(f"Saved mp4 animation to: {fname}")

    def plot(
        self, pl, single=True, style="sb", light=True, shadows=False
    ) -> pv.Plotter:
        """
        Return the pyVista Plotter object for the Disulfide bond in the specific rendering style.

        :param single: Display the bond in a single panel in the specific style.
        :param style:  Rendering style: One of:
            * 'sb' - split bonds
            * 'bs' - ball and stick
            * 'cpk' - CPK style
            * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            * 'plain' - boring single color
        :param light: If True, light background, if False, dark
        """
        src = self.pdb_id
        enrg = self.energy

        title = f"{src}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol. Cα: {self.ca_distance:.2f} Å Cβ: {self.cb_distance:.2f} Å Tors: {self.torsion_length:.2f}°"

        if light:
            pv.set_plot_theme("document")
        else:
            pv.set_plot_theme("dark")

        pl.clear()

        if single:
            pl = pv.Plotter(window_size=WINSIZE)
            # pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing("msaa")
            # pl.add_camera_orientation_widget()

            self._render(
                pl,
                style=style,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()
        else:
            pl = pv.Plotter(shape=(2, 2))
            pl.subplot(0, 0)

            # pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing("msaa")

            # pl.add_camera_orientation_widget()

            self._render(
                pl,
                style="cpk",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(0, 1)

            self._render(
                pl,
                style="bs",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(1, 0)

            self._render(
                pl,
                style="sb",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(1, 1)
            self._render(
                pl,
                style="pd",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.link_views()
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()
        return pl

    def pprint(self) -> None:
        """
        Pretty print general info for the Disulfide
        """
        s1 = self.repr_ss_info()
        s2 = self.repr_ss_ca_dist()
        s2b = self.repr_ss_sg_dist()
        s3 = self.repr_ss_conformation()
        s4 = self.repr_ss_torsion_length()
        res = f"{s1} \n{s3} \n{s2} \n{s2b} \n{s4}>"
        print(res)

    def pprint_all(self) -> None:
        """
        Pretty print all info for the Disulfide
        """
        s1 = self.repr_ss_info() + "\n"
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s4b = self.repr_phipsi()
        s6 = self.repr_ss_ca_dist()
        s6b = self.repr_ss_cb_dist()
        s6c = self.repr_ss_sg_dist()
        s7 = self.repr_ss_torsion_length()
        s8 = self.repr_ss_secondary_structure()

        res = f"{s1} {s2} {s3} {s4}\n {s4b}\n {s6}\n {s6b}\n {s6c}\n {s7}\n {s8}>"

        print(res)

    # repr functions. The class is large, so I split it up into sections
    def repr_ss_info(self) -> str:
        """
        Representation for the Disulfide class
        """
        s1 = f"<Disulfide {self.name}, Source: {self.pdb_id}, Resolution: {self.resolution} Å"
        return s1

    def repr_ss_coords(self) -> str:
        """
        Representation for Disulfide coordinates
        """
        s2 = f"\nProximal Coordinates:\n   N: {self.n_prox}\n   Cα: {self.ca_prox}\n   C: {self.c_prox}\n   O: {self.o_prox}\n   Cβ: {self.cb_prox}\n   Sγ: {self.sg_prox}\n   Cprev {self.c_prev_prox}\n   Nnext: {self.n_next_prox}\n"
        s3 = f"Distal Coordinates:\n   N: {self.n_dist}\n   Cα: {self.ca_dist}\n   C: {self.c_dist}\n   O: {self.o_dist}\n   Cβ: {self.cb_dist}\n   Sγ: {self.sg_dist}\n   Cprev {self.c_prev_dist}\n   Nnext: {self.n_next_dist}\n\n"
        stot = f"{s2} {s3}"
        return stot

    def repr_ss_conformation(self) -> str:
        """
        Representation for Disulfide conformation
        """
        s4 = f"Χ1-Χ5: {self.chi1:.2f}°, {self.chi2:.2f}°, {self.chi3:.2f}°, {self.chi4:.2f}° {self.chi5:.2f}°, {self.rho:.2f}°, {self.energy:.2f} kcal/mol"
        stot = f"{s4}"
        return stot

    def repr_ss_local_coords(self) -> str:
        """
        Representation for the Disulfide internal coordinates.
        """
        s2i = f"Proximal Internal Coords:\n   N: {self._n_prox}\n   Cα: {self._ca_prox}\n   C: {self._c_prox}\n   O: {self._o_prox}\n   Cβ: {self._cb_prox}\n   Sγ: {self._sg_prox}\n   Cprev {self.c_prev_prox}\n   Nnext: {self.n_next_prox}\n"
        s3i = f"Distal Internal Coords:\n   N: {self._n_dist}\n   Cα: {self._ca_dist}\n   C: {self._c_dist}\n   O: {self._o_dist}\n   Cβ: {self._cb_dist}\n   Sγ: {self._sg_dist}\n   Cprev {self.c_prev_dist}\n   Nnext: {self.n_next_dist}\n"
        stot = f"{s2i}{s3i}"
        return stot

    def repr_ss_residue_ids(self) -> str:
        """
        Representation for Disulfide chain IDs
        """
        return f"Proximal Residue fullID: <{self.proximal}> Distal Residue fullID: <{self.distal}>"

    def repr_ss_ca_dist(self) -> str:
        """
        Representation for Disulfide Ca distance
        """
        s1 = f"Cα Distance: {self.ca_distance:.2f} Å"
        return s1

    def repr_ss_cb_dist(self) -> str:
        """
        Representation for Disulfide Ca distance
        """
        s1 = f"Cβ Distance: {self.cb_distance:.2f} Å"
        return s1

    def repr_ss_sg_dist(self) -> str:
        """
        Representation for Disulfide Ca distance
        """
        s1 = f"Sγ Distance: {self.sg_distance:.2f} Å"
        return s1

    def repr_ss_torsion_length(self) -> str:
        """
        Representation for Disulfide torsion length
        """
        s1 = f"Torsion length: {self.torsion_length:.2f} deg"
        return s1

    def repr_ss_secondary_structure(self) -> str:
        """
        Representation for Disulfide secondary structure
        """
        s1 = f"Proximal secondary: {self.proximal_secondary} Distal secondary: {self.distal_secondary}"
        return s1

    def repr_phipsi(self) -> str:
        """
        Representation for Disulfide phi psi angles
        """
        s1 = f"PhiProx: {self.phiprox:.2f}° PsiProx: {self.psiprox:.2f}°, PhiDist: {self.phidist:.2f}° PsiDist: {self.psidist:.2f}°"
        return s1

    def repr_all(self) -> str:
        """
        Return a string representation for all Disulfide information
        contained in self.
        """

        s1 = self.repr_ss_info() + "\n"
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s4b = self.repr_phipsi()
        s6 = self.repr_ss_ca_dist()
        s8 = self.repr_ss_cb_dist()
        s7 = self.repr_ss_torsion_length()
        s9 = self.repr_ss_secondary_structure()

        res = f"{s1} {s2} {s3} {s4} {s4b} {s6} {s7} {s8} {s9}>"
        return res

    def repr_compact(self) -> str:
        """
        Return a compact representation of the Disulfide object
        :return: string
        """
        return f"{self.repr_ss_info()} {self.repr_ss_conformation()}"

    def repr_conformation(self) -> str:
        """
        Return a string representation of the Disulfide object's conformation.
        :return: string
        """
        return f"{self.repr_ss_conformation()}"

    def repr_coords(self) -> str:
        """
        Return a string representation of the Disulfide object's coordinates.
        :return: string
        """
        return f"{self.repr_ss_coords()}"

    def repr_internal_coords(self) -> str:
        """
        Return a string representation of the Disulfide object's internal coordinaes.
        :return: string
        """
        return f"{self.repr_ss_local_coords()}"

    def repr_chain_ids(self) -> str:
        """
        Return a string representation of the Disulfide object's chain ids.
        :return: string
        """
        return f"{self.repr_ss_residue_ids()}"

    @property
    def rho(self) -> float:
        """
        Return the dihedral angle rho for the Disulfide.
        """
        return self._compute_rho()

    @rho.setter
    def rho(self, value: float):
        """
        Set the dihedral angle rho for the Disulfide.
        """
        self._rho = value

    def _compute_rho(self) -> float:
        """
        Compute the dihedral angle rho for a Disulfide object and
        sets the internal state of the object.
        """

        v1 = self.n_prox - self.ca_prox
        v2 = self.c_prox - self.ca_prox
        n1 = np.cross(v2.get_array(), v1.get_array())

        v4 = self.n_dist - self.ca_dist
        v3 = self.c_dist - self.ca_dist
        n2 = np.cross(v4.get_array(), v3.get_array())
        self._rho = calc_dihedral(
            Vector3D(n1), self.ca_prox, self.ca_dist, Vector3D(n2)
        )
        return self._rho

    def reset(self) -> None:
        """
        Resets the disulfide object to its initial state. All distances,
        angles and positions are reset. The name is unchanged.
        """
        self.__init__(self)

    def same_chains(self) -> bool:
        """
        Function checks if the Disulfide is cross-chain or not.

        Returns
        -------
        bool \n
            True if the proximal and distal residues are on the same chains,
            False otherwise.
        """

        (prox, dist) = self.get_chains()
        return prox == dist

    def screenshot(
        self,
        single=True,
        style="sb",
        fname="ssbond.png",
        verbose=False,
        shadows=False,
        light="Auto",
    ) -> None:
        """
        Create and save a screenshot of the Disulfide in the given style
        and filename

        :param single: Display a single vs panel view, defaults to True
        :param style: Rendering style, one of:
        * 'sb' - split bonds
        * 'bs' - ball and stick
        * 'cpk' - CPK style
        * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
        * 'plain' - boring single color,
        :param fname: output filename,, defaults to 'ssbond.png'
        :param verbose: Verbosit, defaults to False
        :param shadows: Enable shadows, defaults to False
        """
        src = self.pdb_id
        name = self.name
        enrg = self.energy

        title = f"{src} {name}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol, Cα: {self.ca_distance:.2f} Å, Cβ: {self.cb_distance:.2f} Å, Sγ: {self.sg_distance:.2f} Å, Tors: {self.torsion_length:.2f}"

        if light == "Light":
            pv.set_plot_theme("document")
        elif light == "Dark":
            pv.set_plot_theme("dark")
        else:
            _theme = get_theme()
            if _theme == "light":
                pv.set_plot_theme("document")
            elif _theme == "dark":
                pv.set_plot_theme("dark")
            else:
                pv.set_plot_theme("document")

        if verbose:
            _logger.info("Rendering screenshot to file {fname}")

        if single:
            pl = pv.Plotter(window_size=WINSIZE)
            # pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing("msaa")
            # pl.add_camera_orientation_widget()
            self._render(
                pl,
                style=style,
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()

            pl.show(auto_close=False)  # allows for manipulation
            # Take the screenshot after ensuring the plotter is still active
            try:
                pl.screenshot(fname)
                if verbose:
                    print(f" -> display_overlay(): Saved image to: {fname}")
            except RuntimeError as e:
                _logger.error(f"Error saving screenshot: {e}")

        else:
            pl = pv.Plotter(window_size=WINSIZE, shape=(2, 2))
            pl.subplot(0, 0)

            # pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing("msaa")

            # pl.add_camera_orientation_widget()
            self._render(
                pl,
                style="cpk",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(0, 1)
            # pl.add_title(title=title, font_size=FONTSIZE)
            self._render(
                pl,
                style="pd",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(1, 0)
            # pl.add_title(title=title, font_size=FONTSIZE)
            self._render(
                pl,
                style="bs",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.subplot(1, 1)
            # pl.add_title(title=title, font_size=FONTSIZE)
            self._render(
                pl,
                style="sb",
                bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE,
                spec=SPECULARITY,
                specpow=SPEC_POWER,
            )

            pl.link_views()
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()

            # Take the screenshot after ensuring the plotter is still active
            pl.show(auto_close=False)  # allows for manipulation

            try:
                pl.screenshot(fname)
                if verbose:
                    _logger.info(f" -> display_overlay(): Saved image to: {fname}")
            except RuntimeError as e:
                _logger.error(f"Error saving screenshot: {e}")

        if verbose:
            print(f"Saved: {fname}")

    def save_meshes_as_stl(self, meshes, filename) -> None:
        """Save a list of meshes as a single STL file.

        Args:
            meshes (list): List of pyvista mesh objects to save.
            filename (str): Path to save the STL file to.
        """
        merged_mesh = pv.UnstructuredGrid()
        for mesh in meshes:
            merged_mesh += mesh
        merged_mesh.save(filename)

    def export(self, style="sb", verbose=True, fname="ssbond_plt") -> None:
        """
        Create and save a screenshot of the Disulfide in the given style and filename.

        :param single: Display a single vs panel view, defaults to True
        :param style: Rendering style, one of:
        * 'sb' - split bonds
        * 'bs' - ball and stick
        * 'cpk' - CPK style
        * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
        * 'plain' - boring single color,

        :param fname: output filename,, defaults to 'ssbond.stl'
        :param verbose: Verbosit, defaults to False
        """

        if verbose:
            print(f"-> screenshot(): Rendering screenshot to file {fname}")

        pl = pv.PolyData()

        self._plot(
            pl,
            style=style,
            bondcolor=BOND_COLOR,
            bs_scale=BS_SCALE,
            spec=SPECULARITY,
            specpow=SPEC_POWER,
        )

        self.save_meshes_as_stl(pl, fname)

        return

    def set_positions(
        self,
        n_prox: Vector3D,
        ca_prox: Vector3D,
        c_prox: Vector3D,
        o_prox: Vector3D,
        cb_prox: Vector3D,
        sg_prox: Vector3D,
        n_dist: Vector3D,
        ca_dist: Vector3D,
        c_dist: Vector3D,
        o_dist: Vector3D,
        cb_dist: Vector3D,
        sg_dist: Vector3D,
        c_prev_prox: Vector3D,
        n_next_prox: Vector3D,
        c_prev_dist: Vector3D,
        n_next_dist: Vector3D,
    ) -> None:
        """
        Set the atomic coordinates for all atoms in the Disulfide object.

        :param n_prox: Proximal N position
        :param ca_prox: Proximal Cα position
        :param c_prox: Proximal C' position
        :param o_prox: Proximal O position
        :param cb_prox: Proximal Cβ position
        :param sg_prox: Proximal Sγ position
        :param n_dist: Distal N position
        :param ca_dist: Distal Cα position
        :param c_dist: Distal C' position
        :param o_dist: Distal O position
        :param cb_dist: Distal Cβ position
        :param sg_dist: Distal Sγ position
        :param c_prev_prox: Proximal previous C'
        :param n_next_prox: Proximal next N
        :param c_prev_dist: Distal previous C'
        :param n_next_dist: Distal next N
        """

        # deep copy
        self.n_prox = n_prox.copy()
        self.ca_prox = ca_prox.copy()
        self.c_prox = c_prox.copy()
        self.o_prox = o_prox.copy()
        self.cb_prox = cb_prox.copy()
        self.sg_prox = sg_prox.copy()
        self.sg_dist = sg_dist.copy()
        self.cb_dist = cb_dist.copy()
        self.ca_dist = ca_dist.copy()
        self.n_dist = n_dist.copy()
        self.c_dist = c_dist.copy()
        self.o_dist = o_dist.copy()

        self.c_prev_prox = c_prev_prox.copy()
        self.n_next_prox = n_next_prox.copy()
        self.c_prev_dist = c_prev_dist.copy()
        self.n_next_dist = n_next_dist.copy()
        self._compute_local_coords()

    def set_name(self, namestr="Disulfide") -> None:
        """
        Set the Disulfide's name.

        :param namestr: Name, by default "Disulfide"
        """
        self.name = namestr

    def set_resnum(self, proximal: int, distal: int) -> None:
        """
        Set the proximal and residue numbers for the Disulfide.

        :param proximal: Proximal residue number
        :param distal: Distal residue number
        """
        self.proximal = proximal
        self.distal = distal

    def _compute_torsion_length(self) -> float:
        """
        Compute the 5D Euclidean length of the Disulfide object. Update the disulfide internal state.

        :return: Torsion length (Degrees)
        """
        # Use numpy array to compute element-wise square
        tors2 = np.square(self.torsion_array)

        # Compute the sum of squares using numpy's sum function
        dist = math.sqrt(np.sum(tors2))

        # Update the internal state
        self.torsion_length = dist

        return dist

    def torsion_distance(self, other) -> float:
        """
        Calculate the 5D Euclidean distance between `self` and another Disulfide
        object. This is used to compare Disulfide Bond torsion angles to
        determine their torsional similarity via a 5-Dimensional Euclidean distance metric.

        :param other: Comparison Disulfide
        :raises ProteusPyWarning: Warning if `other` is not a Disulfide object
        :return: Euclidean distance (Degrees) between `self` and `other`.
        """

        # Check length of torsion arrays
        if len(self.torsion_array) != 5 or len(other.torsion_array) != 5:
            raise ProteusPyWarning(
                "--> Torsion_Distance() requires vectors of length 5!"
            )

        # Convert to numpy arrays
        p1 = np.array(self.torsion_array)
        p2 = np.array(other.torsion_array)

        # Compute the difference and handle angle wrapping
        diff = np.abs(p1 - p2)
        diff = np.where(diff > 180, 360 - diff, diff)

        # Compute the 5D Euclidean distance using numpy's linalg.norm function
        dist = np.linalg.norm(diff)

        return dist

    def torsion_neighbors(self, others, cutoff) -> DisulfideList:
        """
        Return a list of Disulfides within the angular cutoff in the others list.
        This routine is used to find Disulfides having the same torsion length
        within the others list. This is used to find families of Disulfides with
        similar conformations. Assumes self is properly initialized.

        *NB* The routine will not distinguish between +/-
        dihedral angles. *i.e.* [-60, -60, -90, -60, -60] would have the same
        torsion length as [60, 60, 90, 60, 60], two clearly different structures.

        :param others: ```DisulfideList``` to search
        :param cutoff: Dihedral angle degree cutoff
        :return: DisulfideList within the cutoff

        Example:
        In this example we load the disulfide database subset, find the disulfides with
        the lowest and highest energies, and then find the nearest conformational neighbors.
        Finally, we display the neighbors overlaid against a common reference frame.

        >>> from proteusPy import Load_PDB_SS, DisulfideList, Disulfide
        >>> light = get_theme()
        >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)
        >>> ss_list = DisulfideList([], 'tmp')

        We point to the complete list to search for lowest and highest energies.
        >>> sslist = PDB_SS.SSList
        >>> ssmin_enrg, ssmax_enrg = PDB_SS.SSList.minmax_energy

        Make an empty list and find the nearest neighbors within 10 degrees avg RMS in
        sidechain dihedral angle space.

        >>> low_energy_neighbors = DisulfideList([],'Neighbors')
        >>> low_energy_neighbors = ssmin_enrg.torsion_neighbors(sslist, 8)

        Display the number found, and then display them overlaid onto their common reference frame.

        >>> tot = low_energy_neighbors.length
        >>> print(f'Neighbors: {tot}')
        Neighbors: 8
        >>> low_energy_neighbors.display_overlay(light="auto")

        """

        res = [ss for ss in others if self.torsion_distance(ss) <= cutoff]
        return DisulfideList(res, "neighbors")

    def translate(self, translation_vector: Vector3D):
        """Translate the Disulfide object by the given vector."""

        self.n_prox += translation_vector
        self.ca_prox += translation_vector
        self.c_prox += translation_vector
        self.o_prox += translation_vector
        self.cb_prox += translation_vector
        self.sg_prox += translation_vector
        self.sg_dist += translation_vector
        self.cb_dist += translation_vector
        self.ca_dist += translation_vector
        self.n_dist += translation_vector
        self.c_dist += translation_vector
        self.o_dist += translation_vector

        self.c_prev_prox += translation_vector
        self.n_next_prox += translation_vector
        self.c_prev_dist += translation_vector
        self.n_next_dist += translation_vector
        self._compute_local_coords()


# Class defination ends


def disulfide_energy_function(x: list) -> float:
    """
    Compute the approximate torsional energy (kcal/mpl) for the input dihedral angles.

    :param x: A list of dihedral angles: [chi1, chi2, chi3, chi4, chi5]
    :return: Energy in kcal/mol

    Example:
    >>> from proteusPy import disulfide_energy_function
    >>> dihed = [-60.0, -60.0, -90.0, -60.0, -90.0]
    >>> res = disulfide_energy_function(dihed)
    >>> float(res)
    2.5999999999999996
    """

    chi1, chi2, chi3, chi4, chi5 = x
    energy = 2.0 * (np.cos(np.deg2rad(3.0 * chi1)) + np.cos(np.deg2rad(3.0 * chi5)))
    energy += np.cos(np.deg2rad(3.0 * chi2)) + np.cos(np.deg2rad(3.0 * chi4))
    energy += (
        3.5 * np.cos(np.deg2rad(2.0 * chi3))
        + 0.6 * np.cos(np.deg2rad(3.0 * chi3))
        + 10.1
    )
    return energy


def minimize_ss_energy(inputSS: Disulfide) -> Disulfide:
    """
    Minimizes the energy of a Disulfide object using the Nelder-Mead optimization method.

    Parameters:
        inputSS (Disulfide): The Disulfide object to be minimized.

    Returns:
        Disulfide: The minimized Disulfide object.

    """

    initial_guess = inputSS.torsion_array
    result = minimize(disulfide_energy_function, initial_guess, method="Nelder-Mead")
    minimum_conformation = result.x
    modelled_min = Disulfide("minimized", minimum_conformation)
    # modelled_min.dihedrals = minimum_conformation
    # modelled_min.build_yourself()
    return modelled_min


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
