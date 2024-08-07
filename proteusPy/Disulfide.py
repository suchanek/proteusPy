"""
This module, *Disulfide*, is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
It represents the core of the current implementation of *proteusPy*.

This work is based on the original C/C++ implementation by Eric G. Suchanek. \n
Author: Eric G. Suchanek, PhD
Last revision: 2/17/2024
"""

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

import copy
import datetime
import glob
import math
import pickle
import time
from math import cos

import numpy as np
import pyvista as pv

np.set_printoptions(suppress=True)
pv.global_theme.color = "white"

from Bio.PDB import Vector, calc_dihedral

import proteusPy
from proteusPy.atoms import *
from proteusPy.DisulfideExceptions import *
from proteusPy.DisulfideList import DisulfideList
from proteusPy.ProteusGlobals import _ANG_INIT, _FLOAT_INIT, WINSIZE
from proteusPy.Residue import build_residue
from proteusPy.turtle3D import ORIENT_SIDECHAIN, Turtle3D
from proteusPy.utility import distance3d

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
    "phi_prox",
    "psi_prox",
    "phi_dist",
    "psi_dist",
    "torsion_length",
    "rho",
]


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
        permissive: bool = True,
    ) -> None:
        """
        __init__ Initialize the class to defined internal values.

        :param name: Disulfide name, by default "SSBOND"

        """
        self.name = name
        self.proximal = proximal
        self.distal = distal
        self.energy = _FLOAT_INIT
        self.proximal_chain = proximal_chain
        self.distal_chain = distal_chain
        self.pdb_id = pdb_id
        self.proximal_residue_fullid = str("")
        self.distal_residue_fullid = str("")
        self.PERMISSIVE = permissive
        self.QUIET = quiet
        self.ca_distance = _FLOAT_INIT
        self.cb_distance = _FLOAT_INIT
        self.torsion_array = np.array(
            (_ANG_INIT, _ANG_INIT, _ANG_INIT, _ANG_INIT, _ANG_INIT)
        )
        self.phiprox = _ANG_INIT
        self.psiprox = _ANG_INIT
        self.phidist = _ANG_INIT
        self.psidist = _ANG_INIT

        # global coordinates for the Disulfide, typically as
        # returned from the PDB file

        self.n_prox = Vector(0, 0, 0)
        self.ca_prox = Vector(0, 0, 0)
        self.c_prox = Vector(0, 0, 0)
        self.o_prox = Vector(0, 0, 0)
        self.cb_prox = Vector(0, 0, 0)
        self.sg_prox = Vector(0, 0, 0)
        self.sg_dist = Vector(0, 0, 0)
        self.cb_dist = Vector(0, 0, 0)
        self.ca_dist = Vector(0, 0, 0)
        self.n_dist = Vector(0, 0, 0)
        self.c_dist = Vector(0, 0, 0)
        self.o_dist = Vector(0, 0, 0)

        # set when we can't find previous or next prox or distal
        # C' or N atoms.
        self.missing_atoms = False
        self.modelled = False
        self.resolution = -1.0

        # need these to calculate backbone dihedral angles
        self.c_prev_prox = Vector(0, 0, 0)
        self.n_next_prox = Vector(0, 0, 0)
        self.c_prev_dist = Vector(0, 0, 0)
        self.n_next_dist = Vector(0, 0, 0)

        # local coordinates for the Disulfide, computed using the Turtle3D in
        # Orientation #1. these are generally private.

        self._n_prox = Vector(0, 0, 0)
        self._ca_prox = Vector(0, 0, 0)
        self._c_prox = Vector(0, 0, 0)
        self._o_prox = Vector(0, 0, 0)
        self._cb_prox = Vector(0, 0, 0)
        self._sg_prox = Vector(0, 0, 0)
        self._sg_dist = Vector(0, 0, 0)
        self._cb_dist = Vector(0, 0, 0)
        self._ca_dist = Vector(0, 0, 0)
        self._n_dist = Vector(0, 0, 0)
        self._c_dist = Vector(0, 0, 0)
        self._o_dist = Vector(0, 0, 0)

        # need these to calculate backbone dihedral angles
        self._c_prev_prox = Vector(0, 0, 0)
        self._n_next_prox = Vector(0, 0, 0)
        self._c_prev_dist = Vector(0, 0, 0)
        self._n_next_dist = Vector(0, 0, 0)

        # Dihedral angles for the disulfide bond itself, set to _ANG_INIT
        self.chi1 = _ANG_INIT
        self.chi2 = _ANG_INIT
        self.chi3 = _ANG_INIT
        self.chi4 = _ANG_INIT
        self.chi5 = _ANG_INIT
        self.rho = _ANG_INIT  # new dihedral angle: Nprox - Ca_prox - Ca_dist - N_dist

        self.torsion_length = _FLOAT_INIT

    # comparison operators, used for sorting. keyed to SS bond energy
    def __lt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy < other.energy

    def __le__(self, other):
        if isinstance(other, Disulfide):
            return self.energy <= other.energy

    def __gt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy > other.energy

    def __ge__(self, other):
        if isinstance(other, Disulfide):
            return self.energy >= other.energy

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

    def __repr__(self):
        """
        Representation for the Disulfide class
        """
        s1 = self.repr_ss_info()
        res = f"{s1}>"
        return res

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
        coords = self.internal_coords()
        missing_atoms = self.missing_atoms
        clen = coords.shape[0]

        model = self.modelled
        if model:
            all_atoms = False
        else:
            all_atoms = True

        if translate:
            cofmass = self.cofmass()
            for i in range(clen):
                coords[i] = coords[i] - cofmass

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

        # bond connection table with atoms in the specific order shown above:
        # returned by ss.get_internal_coords()

        def draw_bonds(
            pvp,
            bradius=BOND_RADIUS,
            style="sb",
            bcolor=BOND_COLOR,
            missing=True,
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

            for i in range(len(bond_conn)):
                if all_atoms:
                    if i > 10 and missing_atoms == True:  # skip missing atoms
                        continue

                bond = bond_conn[i]

                # get the indices for the origin and destination atoms
                orig = bond[0]
                dest = bond[1]

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

                bradius = _bradius

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
                    bradius = _bradius * 0.5  # make smaller to distinguish

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

        if style == "cpk":
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_CPK[atom]
                pvp.add_mesh(
                    pv.Sphere(center=coords[i], radius=rad),
                    color=ATOM_COLORS[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )
                i += 1

        elif style == "cov":
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_COVALENT[atom]
                pvp.add_mesh(
                    pv.Sphere(center=coords[i], radius=rad),
                    color=ATOM_COLORS[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )
                i += 1

        elif style == "bs":  # ball and stick
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_CPK[atom] * bs_scale
                if i > 11:
                    rad = rad * 0.75

                pvp.add_mesh(
                    pv.Sphere(center=coords[i], radius=rad),
                    color=ATOM_COLORS[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )
                i += 1
            pvp = draw_bonds(
                pvp, style="bs", missing=missing_atoms, all_atoms=all_atoms
            )

        elif style == "sb":  # splitbonds
            pvp = draw_bonds(
                pvp, style="sb", missing=missing_atoms, all_atoms=all_atoms
            )

        elif style == "pd":  # proximal-distal
            pvp = draw_bonds(
                pvp, style="pd", missing=missing_atoms, all_atoms=all_atoms
            )

        else:  # plain
            pvp = draw_bonds(
                pvp,
                style="plain",
                bcolor=bondcolor,
                missing=missing_atoms,
                all_atoms=all_atoms,
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
        coords = self.internal_coords()
        missing_atoms = self.missing_atoms
        clen = coords.shape[0]

        model = self.modelled
        if model:
            all_atoms = False
        else:
            all_atoms = True

        if translate:
            cofmass = self.cofmass()
            for i in range(clen):
                coords[i] = coords[i] - cofmass

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

        def draw_bonds(
            pvp,
            bradius=BOND_RADIUS,
            style="sb",
            bcolor=BOND_COLOR,
            missing=True,
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

            for i in range(len(bond_conn)):
                if all_atoms:
                    if i > 10 and missing_atoms == True:  # skip missing atoms
                        continue

                bond = bond_conn[i]

                # get the indices for the origin and destination atoms
                orig = bond[0]
                dest = bond[1]

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

                bradius = _bradius

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
                    bradius = _bradius * 0.5  # make smaller to distinguish

                cap1 = pv.Sphere(center=prox_pos, radius=bradius)
                cap2 = pv.Sphere(center=distal_pos, radius=bradius)

                if style == "plain":
                    cyl = pv.Cylinder(
                        origin, direction, radius=bradius, height=height * 2.0
                    )
                    # pvp.add_mesh(cyl, color=orig_col)
                    pvp.append(cyl)
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
                    # pvp.add_mesh(cyl1, color=orig_col)
                    # pvp.add_mesh(cyl2, color=dest_col)
                    pvp.append(cyl1)
                    pvp.append(cyl2)

                # pvp.add_mesh(cap1, color=orig_col)
                # pvp.add_mesh(cap2, color=dest_col)
                pvp.append(cap1)
                pvp.append(cap2)

            return pvp.copy()  # end draw_bonds

        if style == "cpk":
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_CPK[atom]
                pvp.append(pv.Sphere(center=coords[i], radius=rad))
                i += 1

        elif style == "cov":
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_COVALENT[atom]
                pvp.append(pv.Sphere(center=coords[i], radius=rad))
                i += 1

        elif style == "bs":  # ball and stick
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_CPK[atom] * bs_scale
                if i > 11:
                    rad = rad * 0.75

                pvp.append(pv.Sphere(center=coords[i]))
                i += 1
            pvp = draw_bonds(
                pvp, style="bs", missing=missing_atoms, all_atoms=all_atoms
            )

        elif style == "sb":  # splitbonds
            pvp = draw_bonds(
                pvp, style="sb", missing=missing_atoms, all_atoms=all_atoms
            )

        elif style == "pd":  # proximal-distal
            pvp = draw_bonds(
                pvp, style="pd", missing=missing_atoms, all_atoms=all_atoms
            )

        else:  # plain
            pvp = draw_bonds(
                pvp,
                style="plain",
                bcolor=bondcolor,
                missing=missing_atoms,
                all_atoms=all_atoms,
            )

        return

    def _handle_SS_exception(self, message: str):
        """
        This method catches an exception that occurs in the Disulfide
        object (if PERMISSIVE), or raises it again, this time adding the
        PDB line number to the error message. (private).

        :param message: Error message
        :raises DisulfideConstructionException: Fatal construction exception.

        """
        # message = "%s at line %i." % (message)
        message = f"{message}"

        if self.PERMISSIVE:
            # just print a warning - some residues/atoms may be missing
            warnings.warn(
                "DisulfideConstructionException: %s\n"
                "Exception ignored.\n"
                "Some atoms may be missing in the data structure." % message,
                DisulfideConstructionWarning,
            )
        else:
            # exceptions are fatal - raise again with new message (including line nr)
            raise DisulfideConstructionException(message) from None

    @property
    def dihedrals(self) -> list:
        """
        Return a list containing the dihedral angles for the disulfide.

        """
        return [self.chi1, self.chi2, self.chi3, self.chi4, self.chi5]

    @dihedrals.setter
    def dihedrals(self, dihedrals: list) -> None:
        """
        Sets the disulfide dihedral angles to the inputs specified in the list.

        :param dihedrals: list of dihedral angles.
        """
        self.chi1 = dihedrals[0]
        self.chi2 = dihedrals[1]
        self.chi3 = dihedrals[2]
        self.chi4 = dihedrals[3]
        self.chi5 = dihedrals[4]
        self.compute_torsional_energy()
        self.compute_torsion_length()

    def bounding_box(self) -> np.array:
        """
        Return the bounding box array for the given disulfide

        Returns
        -------
        :return: np.Array(3,2): Array containing the min, max for X, Y, and Z respectively.
        Does not currently take the atom's radius into account.

        """
        res = np.zeros(shape=(3, 2))
        xmin, xmax = self.compute_extents("x")
        ymin, ymax = self.compute_extents("y")
        zmin, zmax = self.compute_extents("z")

        res[0] = [xmin, xmax]
        res[1] = [ymin, ymax]
        res[2] = [zmin, zmax]

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

        self.set_dihedrals(chi1, chi2, chi3, chi4, chi5)
        self.proximal = 1
        self.distal = 2

        tmp = Turtle3D("tmp")
        tmp.Orientation = 1

        n = Vector(0, 0, 0)
        ca = Vector(0, 0, 0)
        cb = Vector(0, 0, 0)
        c = Vector(0, 0, 0)

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
        self.cb_prox = Vector(tmp._position)

        tmp.move(1.86)
        tmp.roll(self.chi2)
        tmp.yaw(103.8)
        self.sg_prox = Vector(tmp._position)

        tmp.move(2.044)
        tmp.roll(self.chi3)
        tmp.yaw(103.8)
        self.sg_dist = Vector(tmp._position)

        tmp.move(1.86)
        tmp.roll(self.chi4)
        tmp.yaw(112.8)
        self.cb_dist = Vector(tmp._position)

        tmp.move(1.53)
        tmp.roll(self.chi5)
        tmp.pitch(180.0)

        tmp.schain_to_bbone()

        n, ca, cb, c = build_residue(tmp)

        self.n_dist = n
        self.ca_dist = ca
        self.c_dist = c
        self.compute_torsional_energy()
        self.compute_local_coords()
        self.ca_distance = distance3d(self.ca_prox, self.ca_dist)
        self.cb_distance = distance3d(self.cb_prox, self.cb_dist)
        self.torsion_array = np.array(
            (self.chi1, self.chi2, self.chi3, self.chi4, self.chi5)
        )
        self.compute_torsion_length()
        self.compute_rho()
        self.missing_atoms = True
        self.modelled = True

    def cofmass(self) -> np.array:
        """
        Return the geometric center of mass for the internal coordinates of
        the given Disulfide. Missing atoms are not included.

        :return: 3D array for the geometric center of mass
        """

        res = self.internal_coords()
        return res.mean(axis=0)

    def copy(self):
        """
        Copy the Disulfide.

        :return: A copy of self.
        """
        return copy.deepcopy(self)

    def compute_extents(self, dim="z"):
        """
        Calculate the internal coordinate extents for the input axis.

        :param dim: Axis, one of 'x', 'y', 'z', by default 'z'
        :return: min, max
        """

        ic = self.internal_coords()
        # set default index to 'z'
        idx = 2

        if dim == "x":
            idx = 0
        elif dim == "y":
            idx = 1
        elif dim == "z":
            idx = 2

        _min = ic[:, idx].min()
        _max = ic[:, idx].max()
        return _min, _max

    def compute_local_coords(self) -> None:
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

        self._n_prox = Vector(turt.to_local(n))
        self._ca_prox = Vector(turt.to_local(ca))
        self._c_prox = Vector(turt.to_local(c))
        self._o_prox = Vector(turt.to_local(o))
        self._cb_prox = Vector(turt.to_local(cb))
        self._sg_prox = Vector(turt.to_local(sg))

        self._c_prev_prox = Vector(turt.to_local(cpp))
        self._n_next_prox = Vector(turt.to_local(nnp))
        self._c_prev_dist = Vector(turt.to_local(cpd))
        self._n_next_dist = Vector(turt.to_local(nnd))

        self._n_dist = Vector(turt.to_local(n2))
        self._ca_dist = Vector(turt.to_local(ca2))
        self._c_dist = Vector(turt.to_local(c2))
        self._o_dist = Vector(turt.to_local(o2))
        self._cb_dist = Vector(turt.to_local(cb2))
        self._sg_dist = Vector(turt.to_local(sg2))

    def compute_torsional_energy(self) -> float:
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

    def display(self, single=True, style="sb", light=True, shadows=False) -> None:
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

        title = f"{src}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol. Cα: {self.ca_distance:.2f} Å Cβ: {self.cb_distance:.2f} Å Tors: {self.torsion_length:.2f}°"

        if light:
            pv.set_plot_theme("document")
        else:
            pv.set_plot_theme("dark")

        if single == True:
            _pl = pv.Plotter(window_size=WINSIZE)
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
            if shadows == True:
                _pl.enable_shadows()
            _pl.show()

        else:
            pl = pv.Plotter(window_size=WINSIZE, shape=(2, 2))
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
            if shadows == True:
                pl.enable_shadows()
            pl.show()
        return

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

        if single == True:
            # _pl = pv.Plotter(window_size=WINSIZE)
            # _pl.add_title(title=title, font_size=FONTSIZE)
            pl.clear()
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
            if shadows == True:
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
            if shadows == True:
                pl.enable_shadows()
        return pl

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
        ic1 = self.internal_coords()
        ic2 = other.internal_coords()

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
        import math

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

    def get_permissive(self) -> bool:
        """
        Return the Permissive flag state. (Used in PDB parsing)

        :return: Permissive state
        """
        return self.PERMISIVE

    def get_full_id(self) -> tuple:
        """
        Return the Disulfide full IDs (Used with BIO.PDB)

        :return: Disulfide full IDs
        """
        return (self.proximal_residue_fullid, self.distal_residue_fullid)

    def initialize_disulfide_from_chain(
        self, chain1, chain2, proximal, distal, resolution, quiet=True
    ) -> bool:
        """
        Initialize a new Disulfide object with atomic coordinates from
        the proximal and distal coordinates, typically taken from a PDB file.
        This routine is primarily used internally when building the compressed
        database.

        :param chain1: list of Residues in the model, eg: chain = model['A']
        :param chain2: list of Residues in the model, eg: chain = model['A']
        :param proximal: proximal residue sequence ID
        :param distal: distal residue sequence ID
        :param resolution: structure resolution
        :param quiet: Quiet or noisy parsing, defaults to True
        :raises DisulfideConstructionWarning: Raised when not parsed correctly
        """
        id = chain1.get_full_id()[0]
        self.pdb_id = id

        chi1 = chi2 = chi3 = chi4 = chi5 = _ANG_INIT

        prox = int(proximal)
        dist = int(distal)

        prox_residue = chain1[prox]
        dist_residue = chain2[dist]

        if prox_residue.get_resname() != "CYS" or dist_residue.get_resname() != "CYS":
            print(
                f"build_disulfide() requires CYS at both residues: {prox} {prox_residue.get_resname()} {dist} {dist_residue.get_resname()} Chain: {prox_residue.get_segid()}"
            )

        # set the objects proximal and distal values
        self.set_resnum(proximal, distal)

        if resolution is not None:
            self.resolution = resolution

        self.proximal_chain = chain1.get_id()
        self.distal_chain = chain2.get_id()

        self.proximal_residue_fullid = prox_residue.get_full_id()
        self.distal_residue_fullid = dist_residue.get_full_id()

        if quiet:
            warnings.filterwarnings("ignore", category=DisulfideConstructionWarning)
        else:
            warnings.simplefilter("always")

        # grab the coordinates for the proximal and distal residues as vectors
        # so we can do math on them later
        # proximal residue

        try:
            n1 = prox_residue["N"].get_vector()
            ca1 = prox_residue["CA"].get_vector()
            c1 = prox_residue["C"].get_vector()
            o1 = prox_residue["O"].get_vector()
            cb1 = prox_residue["CB"].get_vector()
            sg1 = prox_residue["SG"].get_vector()

        except Exception:
            print(f"Invalid or missing coordinates for proximal residue {proximal}")
            return False

        # distal residue
        try:
            n2 = dist_residue["N"].get_vector()
            ca2 = dist_residue["CA"].get_vector()
            c2 = dist_residue["C"].get_vector()
            o2 = dist_residue["O"].get_vector()
            cb2 = dist_residue["CB"].get_vector()
            sg2 = dist_residue["SG"].get_vector()

        except Exception:
            print(f"Invalid or missing coordinates for distal residue {distal}")
            return False

        # previous residue and next residue - optional, used for phi, psi calculations
        try:
            prevprox = chain1[prox - 1]
            nextprox = chain1[prox + 1]

            prevdist = chain2[dist - 1]
            nextdist = chain2[dist + 1]

            cprev_prox = prevprox["C"].get_vector()
            nnext_prox = nextprox["N"].get_vector()

            cprev_dist = prevdist["C"].get_vector()
            nnext_dist = nextdist["N"].get_vector()

            # compute phi, psi for prox and distal
            self.phiprox = np.degrees(calc_dihedral(cprev_prox, n1, ca1, c1))
            self.psiprox = np.degrees(calc_dihedral(n1, ca1, c1, nnext_prox))
            self.phidist = np.degrees(calc_dihedral(cprev_dist, n2, ca2, c2))
            self.psidist = np.degrees(calc_dihedral(n2, ca2, c2, nnext_dist))

        except Exception:
            mess = f"Missing coords for: {id} {prox-1} or {dist+1} for SS {proximal}-{distal}"
            cprev_prox = nnext_prox = cprev_dist = nnext_dist = Vector(-1.0, -1.0, -1.0)
            self.missing_atoms = True
            # print(f"{mess}")

        # update the positions and conformation
        self.set_positions(
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
        self.chi1 = np.degrees(calc_dihedral(n1, ca1, cb1, sg1))
        self.chi2 = np.degrees(calc_dihedral(ca1, cb1, sg1, sg2))
        self.chi3 = np.degrees(calc_dihedral(cb1, sg1, sg2, cb2))
        self.chi4 = np.degrees(calc_dihedral(sg1, sg2, cb2, ca2))
        self.chi5 = np.degrees(calc_dihedral(sg2, cb2, ca2, n2))
        self.rho = np.degrees(calc_dihedral(n1, ca1, ca2, n2))

        self.ca_distance = distance3d(self.ca_prox, self.ca_dist)
        self.cb_distance = distance3d(self.cb_prox, self.cb_dist)
        self.torsion_array = np.array(
            (self.chi1, self.chi2, self.chi3, self.chi4, self.chi5)
        )
        self.compute_torsion_length()

        # calculate and set the SS bond torsional energy
        self.compute_torsional_energy()

        # compute and set the local coordinates
        self.compute_local_coords()
        return True

    def internal_coords(self) -> np.array:
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

    def internal_coords_res(self, resnumb) -> np.array:
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
            mess = f"-> Disulfide.internal_coords(): Invalid argument. \
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
        # pl.add_camera_orientation_widget()
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

    def pprint(self) -> None:
        """
        Pretty print general info for the Disulfide
        """
        s1 = self.repr_ss_info()
        s2 = self.repr_ss_ca_dist()
        s3 = self.repr_ss_conformation()
        s4 = self.repr_ss_torsion_length()
        res = f"{s1} \n{s3} \n{s2} \n{s4}>"
        print(res)

    def pprint_all(self) -> None:
        """
        Pretty print all info for the Disulfide
        """
        s1 = self.repr_ss_info() + "\n"
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s5 = self.repr_chain_ids()
        s6 = self.repr_ss_ca_dist()
        s7 = self.repr_ss_torsion_length()

        res = f"{s1} {s5} {s2} {s3} {s4}\n {s6}\n {s7}>"

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

    def repr_ss_chain_ids(self) -> str:
        """
        Representation for Disulfide chain IDs
        """
        return f"Proximal Chain fullID: <{self.proximal_residue_fullid}> Distal Chain fullID: <{self.distal_residue_fullid}>"

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

    def repr_ss_torsion_length(self) -> str:
        """
        Representation for Disulfide torsion length
        """
        s1 = f"Torsion length: {self.torsion_length:.2f} deg"
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
        s5 = self.repr_chain_ids()
        s6 = self.repr_ss_ca_dist()
        s8 = self.repr_ss_cb_dist()
        s7 = self.repr_ss_torsion_length()

        res = f"{s1} {s5} {s2} {s3} {s4} {s6} {s7} {s8}>"
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
        return f"{self.repr_ss_chain_ids()}"

    def compute_rho(self) -> float:
        self.rho = calc_dihedral(self.n_prox, self.ca_prox, self.ca_dist, self.n_dist)
        return self.rho

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
        light=True,
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

        title = f"{src} {name}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol, Cα: {self.ca_distance:.2f} Å, Cβ: {self.cb_distance:.2f} Å, Tors: {self.torsion_length:.2f}"

        if light:
            pv.set_plot_theme("document")
        else:
            pv.set_plot_theme("dark")

        if verbose:
            print(f"-> screenshot(): Rendering screenshot to file {fname}")

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

            pl.show(auto_close=False)
            pl.screenshot(fname)
            pl.clear()

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

            pl.show(auto_close=False)
            pl.screenshot(fname)

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

    def set_permissive(self, perm: bool) -> None:
        """
        Set PERMISSIVE flag for Disulfide parsing.

        :return: None
        """

        self.PERMISSIVE = perm

    def set_positions(
        self,
        n_prox: Vector,
        ca_prox: Vector,
        c_prox: Vector,
        o_prox: Vector,
        cb_prox: Vector,
        sg_prox: Vector,
        n_dist: Vector,
        ca_dist: Vector,
        c_dist: Vector,
        o_dist: Vector,
        cb_dist: Vector,
        sg_dist: Vector,
        c_prev_prox: Vector,
        n_next_prox: Vector,
        c_prev_dist: Vector,
        n_next_dist: Vector,
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

    def set_dihedrals(
        self, chi1: float, chi2: float, chi3: float, chi4: float, chi5: float
    ) -> None:
        """
        Set the disulfide's dihedral angles, Chi1-Chi5. -180 - 180 degrees.

        :param chi1: Chi1
        :param chi2: Chi2
        :param chi3: Chi3
        :param chi4: Chi4
        :param chi5: Chi5
        """
        self.chi1 = chi1
        self.chi2 = chi2
        self.chi3 = chi3
        self.chi4 = chi4
        self.chi5 = chi5
        self.torsion_array = np.array([chi1, chi2, chi3, chi4, chi5])
        self.compute_torsional_energy()
        self.compute_torsion_length()

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

    def compute_torsion_length(self) -> float:
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

    def Torsion_Distance(self, other) -> float:
        """
        Calculate the 5D Euclidean distance between ```self``` and another Disulfide
        object. This is used to compare Disulfide Bond torsion angles to
        determine their torsional similarity via a 5-Dimensional Euclidean distance metric.

        :param other: Comparison Disulfide
        :raises ProteusPyWarning: Warning if ```other``` is not a Disulfide object
        :return: Euclidean distance (Degrees) between ```self``` and ```other```.
        """

        from proteusPy.ProteusPyWarning import ProteusPyWarning

        # Check length of torsion arrays
        if len(self.torsion_array) != 5 or len(other.torsion_array) != 5:
            raise ProteusPyWarning(
                "--> Torsion_Distance() requires vectors of length 5!"
            )

        # Convert to numpy arrays and add 180 to each element
        p1 = np.array(self.torsion_array) + 180.0
        p2 = np.array(other.torsion_array) + 180.0

        # Compute the 5D Euclidean distance using numpy's linalg.norm function
        dist = np.linalg.norm(p1 - p2)

        return dist

    def Torsion_neighbors(self, others, cutoff) -> DisulfideList:
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
        >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)
        >>> ss_list = DisulfideList([], 'tmp')

        We point to the complete list to search for lowest and highest energies.
        >>> sslist = PDB_SS.SSList
        >>> ssmin_enrg, ssmax_enrg = PDB_SS.SSList.minmax_energy

        Make an empty list and find the nearest neighbors within 10 degrees avg RMS in
        sidechain dihedral angle space.

        >>> low_energy_neighbors = DisulfideList([],'Neighbors')
        >>> low_energy_neighbors = ssmin_enrg.Torsion_neighbors(sslist, 10)

        Display the number found, and then display them overlaid onto their common reference frame.

        >>> tot = low_energy_neighbors.length
        >>> print(f'Neighbors: {tot}')
        Neighbors: 2
        >>> low_energy_neighbors.display_overlay()

        """
        from proteusPy import DisulfideList

        res = [ss for ss in others if self.Torsion_Distance(ss) <= cutoff]
        return DisulfideList(res, "neighbors")

    def torsion_to_sixclass(self) -> str:
        """
        Return the sextant class string for ``self``.

        :raises DisulfideIOException: _description_
        :return: Sextant string
        """
        from proteusPy.DisulfideClasses import get_sixth_quadrant

        tors = self.torsion_array
        res = [get_sixth_quadrant(x) for x in tors]
        return "".join([str(r) for r in res])


# Class defination ends


def Disulfide_Energy_Function(x: list) -> float:
    """
    Compute the approximate torsional energy (kcal/mpl) for the input dihedral angles.

    :param x: A list of dihedral angles: [chi1, chi2, chi3, chi4, chi5]
    :return: Energy in kcal/mol

    Example:
    >>> from proteusPy import Disulfide_Energy_Function
    >>> dihed = [-60.0, -60.0, -90.0, -60.0, -90.0]
    >>> res = Disulfide_Energy_Function(dihed)
    >>> float(res)
    2.5999999999999996
    """
    import numpy as np

    chi1, chi2, chi3, chi4, chi5 = x
    energy = 2.0 * (np.cos(np.deg2rad(3.0 * chi1)) + np.cos(np.deg2rad(3.0 * chi5)))
    energy += np.cos(np.deg2rad(3.0 * chi2)) + np.cos(np.deg2rad(3.0 * chi4))
    energy += (
        3.5 * np.cos(np.deg2rad(2.0 * chi3))
        + 0.6 * np.cos(np.deg2rad(3.0 * chi3))
        + 10.1
    )
    return energy


def Minimize(inputSS: Disulfide) -> Disulfide:
    """
    Minimizes the energy of a Disulfide object using the Nelder-Mead optimization method.

    Parameters:
        inputSS (Disulfide): The Disulfide object to be minimized.

    Returns:
        Disulfide: The minimized Disulfide object.

    """
    from scipy.optimize import minimize

    from proteusPy import Disulfide, Disulfide_Energy_Function

    initial_guess = inputSS.torsion_array
    result = minimize(Disulfide_Energy_Function, initial_guess, method="Nelder-Mead")
    minimum_conformation = result.x
    modelled_min = Disulfide("minimized")
    modelled_min.dihedrals = minimum_conformation
    modelled_min.build_yourself()
    return modelled_min


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
