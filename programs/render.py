"""
This module provides functionality for rendering disulfide bonds using PyVista,
and is part of the ProteusPy package. The module requires the proteusPy package
to function.

The module includes the `DisulfideBondRenderer` class, which offers methods to render
disulfide bonds in various styles such as ball-and-stick, CPK, covalent, split bonds,
and plain styles. The renderer can handle missing atoms and translate the disulfide bond
to its geometric center of mass for better visualization.

Classes:
    DisulfideBondRenderer: A class for rendering disulfide bonds using PyVista.

Functions:
    main(): Main program function to load a PDB file, create a DisulfideBondRenderer object,
            and display the visualization in different styles.

Attributes:
    ATOM_COLORS (dict): Dictionary mapping atom types to their respective colors.
    ATOM_RADII_COVALENT (dict): Dictionary mapping atom types to their covalent radii.
    ATOM_RADII_CPK (dict): Dictionary mapping atom types to their CPK radii.
    BOND_COLOR (str): Default color for bonds.
    BOND_RADIUS (float): Default radius for bonds.
    BS_SCALE (float): Default scale factor for ball-and-stick rendering.
    SPEC_POWER (int): Default specular power for rendering.
    SPECULARITY (float): Default specularity for rendering.
    WINSIZE (tuple): Default window size for the PyVista plotter.

Usage:
    To use this module, create a `DisulfideBondRenderer` object with a `Disulfide` object
    and call the `display` method to visualize the disulfide bond in the desired style.

Example:
    from proteusPy import Load_PDB_SS
    from render import DisulfideBondRenderer

    pdb = Load_PDB_SS(subset=False, verbose=True)
    ss1 = pdb[0]

    renderer = DisulfideBondRenderer(ss=ss1)
    renderer.display(background_color="white", style="sb", res=50)
"""

import math

import numpy as np
import pyvista as pv

from proteusPy import WINSIZE, Disulfide, Load_PDB_SS
from proteusPy.atoms import (
    ATOM_COLORS,
    ATOM_RADII_COVALENT,
    ATOM_RADII_CPK,
    BOND_COLOR,
    BOND_RADIUS,
    BS_SCALE,
    SPEC_POWER,
    SPECULARITY,
)


class DisulfideBondRenderer:
    """
    This class provides methods to render disulfide bonds in various styles using PyVista.
    It supports rendering atoms and bonds with different styles, including ball-and-stick,
    CPK, covalent, split bonds, and plain styles. The renderer can handle missing atoms
    and translate the disulfide bond to its geometric center of mass for better visualization.

    Attributes:
        _internal_coords (np.ndarray): Internal coordinates of the disulfide bond.
        modelled (bool): Flag indicating if the disulfide bond is modeled.
        missing_atoms (bool): Flag indicating if there are missing atoms.
        cofmass (np.ndarray): Center of mass of the disulfide bond.

    Methods:
        __init__(self, ss: Disulfide):
            Initialize the DisulfideBondRenderer with a Disulfide object.

        _draw_bonds(self, pvp, coords, bradius=BOND_RADIUS, style="sb", bcolor=BOND_COLOR,
                    missing=True, all_atoms=True, res=100):
            Generate the appropriate PyVista cylinder objects to represent a particular
            disulfide bond.

        _render_atoms(self, pvp: pv.Plotter, coords: np.ndarray, style: str, bs_scale: float,
                      spec: float, specpow: int, res: int):
            Render the atoms as spheres based on the selected style.

        _render(self, pvplot: pv.Plotter, style="bs", plain=False, bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE, spec=SPECULARITY, specpow=SPEC_POWER, translate=True,
                bond_radius=BOND_RADIUS, res=100):
            Update the passed PyVista Plotter object with the mesh data for the input
            Disulfide Bond.

        display(self, background_color="white", style="bs", plain=False, bondcolor=BOND_COLOR,
                bs_scale=BS_SCALE, spec=SPECULARITY, specpow=SPEC_POWER, translate=True,
                bond_radius=BOND_RADIUS, res=100):
            Create a PyVista Plotter with specified window size, render the disulfide bond,
            and display the visualization.
    """

    # Static atoms array corresponding to each coordinate row
    ATOMS = [
        "N",  # 0
        "C",  # 1
        "C",  # 2
        "O",  # 3
        "C",  # 4
        "SG",  # 5
        "N",  # 6
        "C",  # 7
        "C",  # 8
        "O",  # 9
        "C",  # 10
        "SG",  # 11
        "C",  # 12
        "N",  # 13
        "C",  # 14
        "N",  # 15
    ]

    def __init__(self, ss: Disulfide):
        """
        Initialize the DisulfideBondRenderer.

        :param ss: A Disulfide object containing atom coordinates and metadata.
        """

        # Initialize attributes
        self._internal_coords = ss.internal_coords_array.copy()
        self.modelled = ss.modelled
        self.missing_atoms = ss.missing_atoms
        self.cofmass = ss.cofmass

    def _draw_bonds(
        self,
        pvp,
        coords,
        bradius=BOND_RADIUS,
        style="sb",
        bcolor=BOND_COLOR,
        missing=False,
        all_atoms=True,
        res=100,
    ):
        """
        Generate the appropriate PyVista cylinder objects to represent
        a particular disulfide bond.

        :param pvp: Input plotter object to be updated.
        :param coords: Coordinates of the atoms.
        :param bradius: Bond radius, by default BOND_RADIUS.
        :param style: Bond style. One of 'sb', 'plain', 'pd'.
        :param bcolor: Bond color for simple bonds, by default BOND_COLOR.
        :param missing: True if atoms are missing, False otherwise, by default True.
        :param all_atoms: True if rendering all atoms including side chains, False if only
            backbone rendered, by default True.
        :param res: Resolution for cylinders, by default 100.

        :return: Updated Plotter object.
        :rtype: pv.Plotter
        """
        # Define bond connections
        _bond_conn = np.array(
            [
                [0, 1],  # N-Ca
                [1, 2],  # Ca-C
                [2, 3],  # C-O
                [1, 4],  # Ca-Cb
                [4, 5],  # Cb-SG
                [6, 7],  # N-Ca (next residue)
                [7, 8],  # Ca-C
                [8, 9],  # C-O
                [7, 10],  # Ca-Cb
                [10, 11],  # Cb-SG
                [5, 11],  # SG-SG
                [12, 0],  # C_prev_prox-N
                [2, 13],  # C-N_next_prox
                [14, 6],  # C_prev_dist-N_dist
                [8, 15],  # C-N_next_dist
            ]
        )

        _bond_conn_backbone = np.array(
            [
                [0, 1],  # N-Ca
                [1, 2],  # Ca-C
                [1, 4],  # Ca-Cb
                [4, 5],  # Cb-SG
                [6, 7],  # N-Ca
                [7, 8],  # Ca-C
                [7, 10],  # Ca-Cb
                [10, 11],  # Cb-SG
                [5, 11],  # SG-SG
            ]
        )

        # Define bond colors
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
                # Prev and next C-N bonds - color by atom type
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

        # Select bond connections and colors based on all_atoms flag
        if all_atoms:
            bond_conn = _bond_conn
            bond_split_colors = _bond_split_colors
        else:
            bond_conn = _bond_conn_backbone
            bond_split_colors = _bond_split_colors_backbone

        for i, bond in enumerate(bond_conn):
            if all_atoms:
                # Skip bonds involving missing atoms if necessary
                if missing and i >= 11:
                    continue

            # Get the indices for the origin and destination atoms
            orig, dest = bond

            # Get the bond color based on atom types
            if i < len(bond_split_colors):
                col = bond_split_colors[i]
                orig_col = ATOM_COLORS.get(col[0], bcolor)
                dest_col = ATOM_COLORS.get(col[1], bcolor)
            else:
                # Default colors if out of range
                orig_col = dest_col = bcolor

            # Get the coordinates
            prox_pos = coords[orig]
            distal_pos = coords[dest]

            # Compute the direction vector and height
            direction = distal_pos - prox_pos
            height = math.dist(prox_pos, distal_pos) / 2.0

            origin = prox_pos + 0.5 * direction  # for a single plain bond

            # Compute split bond origins
            origin1 = prox_pos + 0.25 * direction
            origin2 = prox_pos + 0.75 * direction

            # Adjust bond radius for previous and next residue bonds
            if i >= 12:
                current_bradius = bradius * 0.5  # Make smaller to distinguish
            else:
                current_bradius = bradius

            if style == "plain":
                # Single cylinder for plain style
                orig_col = dest_col = bcolor
                cyl = pv.Cylinder(
                    center=prox_pos + 0.5 * direction,
                    direction=direction,
                    radius=bradius,
                    height=height * 2.0,
                    resolution=res,
                    capping=True,
                )
                pvp.add_mesh(
                    cyl,
                    color=bcolor,
                    smooth_shading=True,
                    specular=SPECULARITY,
                    specular_power=SPEC_POWER,
                )
            elif style == "sb":
                # Split bonds into two cylinders
                cyl1 = pv.Cylinder(
                    center=origin1,
                    direction=direction,
                    radius=current_bradius,
                    height=height,
                    resolution=res,
                    capping=False,
                )
                cyl2 = pv.Cylinder(
                    center=origin2,
                    direction=direction,
                    radius=current_bradius,
                    height=height,
                    resolution=res,
                    capping=False,
                )
                pvp.add_mesh(
                    cyl1,
                    color=orig_col,
                    smooth_shading=True,
                    specular=SPECULARITY,
                    specular_power=SPEC_POWER,
                )
                pvp.add_mesh(
                    cyl2,
                    color=dest_col,
                    smooth_shading=True,
                    specular=SPECULARITY,
                    specular_power=SPEC_POWER,
                )
            elif style == "pd":
                # proximal-distal red/green coloring
                if i <= 5:
                    orig_col = dest_col = "red"
                elif i > 5 and i <= 10:
                    orig_col = dest_col = "green"
                else:
                    orig_col = dest_col = "yellow"
            else:
                orig_col = ATOM_COLORS[col[0]]
                dest_col = ATOM_COLORS[col[1]]

            # Create and add caps
            cap1 = pv.Sphere(
                center=prox_pos,
                radius=current_bradius,
                theta_resolution=res // 2,
                phi_resolution=res // 2,
            )
            cap2 = pv.Sphere(
                center=distal_pos,
                radius=current_bradius,
                theta_resolution=res // 2,
                phi_resolution=res // 2,
            )

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
            pvp.add_mesh(
                cap1,
                color=orig_col,
                smooth_shading=True,
                specular=SPECULARITY,
                specular_power=SPEC_POWER,
            )
            pvp.add_mesh(
                cap2,
                color=dest_col,
                smooth_shading=True,
                specular=SPECULARITY,
                specular_power=SPEC_POWER,
            )

        return pvp  # end draw_bonds

    def _render_atoms(
        self,
        pvp: pv.Plotter,
        coords: np.ndarray,
        style: str,
        bs_scale: float,
        spec: float,
        specpow: int,
        res: int,
    ):
        """
        Render the atoms as spheres based on the selected style.

        :param pvp: PyVista Plotter object.
        :param coords: Coordinates of the atoms.
        :param style: Rendering style.
        :param bs_scale: Scale factor for ball-and-stick.
        :param spec: Specularity.
        :param specpow: Specular power.
        :param res: Resolution for spheres.

        :return: None
        """
        for i, atom in enumerate(self.ATOMS):
            if self.missing_atoms and i > 11:
                continue

            if style in ["cpk", "cov", "bs"]:
                if style == "cpk":
                    rad = ATOM_RADII_CPK.get(atom, 0.5)
                elif style == "cov":
                    rad = ATOM_RADII_COVALENT.get(atom, 0.5)
                elif style == "bs":
                    rad = ATOM_RADII_CPK.get(atom, 0.5) * bs_scale
                    if i > 11:
                        rad *= 0.75

                sphere = pv.Sphere(
                    center=coords[i],
                    radius=rad,
                    theta_resolution=res // 2,
                    phi_resolution=res // 2,
                )
                atom_color = ATOM_COLORS.get(atom, "white")
                pvp.add_mesh(
                    sphere,
                    color=atom_color,
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )

    def _render(
        self,
        pvplot: pv.Plotter,
        style="bs",
        bondcolor=BOND_COLOR,
        bs_scale=BS_SCALE,
        spec=SPECULARITY,
        specpow=SPEC_POWER,
        translate=True,
        bond_radius=BOND_RADIUS,
        res=100,
    ):
        """
        Update the passed PyVista Plotter object with the mesh data for the
        input Disulfide Bond.

        :param pvplot: PyVista Plotter object.
        :param style: Rendering style, by default 'bs'. One of 'bs', 'cpk', 'cov', 'sb', 'pd',
        'plain'.
        :param plain: Used internally, by default False.
        :param bondcolor: Bond color for simple bonds, by default BOND_COLOR.
        :param bs_scale: Scale factor (0-1) to reduce the atom sizes for ball and stick, by
        default BS_SCALE.
        :param spec: Specularity (0-1), where 1 is totally smooth and 0 is rough, by default
        SPECULARITY.
        :param specpow: Exponent used for specularity calculations, by default SPEC_POWER.
        :param translate: Flag used internally to indicate if we should translate
            the disulfide to its geometric center of mass, by default True.
        :param bond_radius: Bond radius, by default BOND_RADIUS.
        :param res: Resolution for spheres and cylinders, by default 100.

        :return: Updated PyVista Plotter object with atoms and bonds.
        :rtype: pv.Plotter
        """
        bradius = bond_radius
        coords = self._internal_coords.copy()
        missing_atoms = self.missing_atoms

        all_atoms = not self.missing_atoms

        if translate:
            cofmass = self.cofmass
            coords = coords - cofmass

        pvp = pvplot

        # Render atoms based on the selected style
        if style in ["cpk", "cov", "bs"]:
            self._render_atoms(pvp, coords, style, bs_scale, spec, specpow, res)

        # Render bonds based on the selected style
        if style in ["bs", "sb", "pd", "plain"]:
            pvp = self._draw_bonds(
                pvp,
                coords=coords,
                bradius=bradius,
                style=style,
                bcolor=bondcolor,
                missing=missing_atoms,
                all_atoms=all_atoms,
                res=res,
            )

        return pvp  # end _render

    def display(
        self,
        background_color="white",
        style="bs",
        bondcolor=BOND_COLOR,
        bs_scale=BS_SCALE,
        spec=SPECULARITY,
        specpow=SPEC_POWER,
        translate=True,
        bond_radius=BOND_RADIUS,
        res=100,
    ):
        """
        Create a PyVista Plotter with specified window size, render the disulfide bond,
        and display the visualization.

        :param background_color: Background color of the plotter, by default 'white'.
        :param style: Rendering style, by default 'bs'. One of 'bs', 'cpk', 'cov', 'sb',
        'pd', 'plain'.
        :param plain: Used internally, by default False.
        :param bondcolor: Bond color for simple bonds, by default BOND_COLOR.
        :param bs_scale: Scale factor (0-1) to reduce the atom sizes for ball and stick,
        by default BS_SCALE.
        :param spec: Specularity (0-1), where 1 is totally smooth and 0 is rough, by
        default SPECULARITY.
        :param specpow: Exponent used for specularity calculations, by default SPEC_POWER.
        :param translate: Flag used internally to indicate if we should translate
            the disulfide to its geometric center of mass, by default True.
        :param bond_radius: Bond radius, by default BOND_RADIUS.
        :param res: Resolution for spheres and cylinders, by default 100.

        :return: None
        """
        # Initialize the PyVista Plotter with specified window size
        plotter = pv.Plotter(window_size=WINSIZE)
        plotter.set_background(background_color)

        # Render the disulfide bond
        plotter = self._render(
            pvplot=plotter,
            style=style,
            bondcolor=bondcolor,
            bs_scale=bs_scale,
            spec=spec,
            specpow=specpow,
            translate=translate,
            bond_radius=bond_radius,
            res=res,
        )

        # Set camera position for better visualization
        try:
            plotter.camera_position = "iso"
        except AttributeError as e:
            print(f"Error setting camera position: {e}")
            print("Plotter object might not be properly initialized.")
            return

        # Display the plot
        plotter.show()


def main():
    """Main program"""

    pdb = Load_PDB_SS(subset=True, verbose=True)
    ss1 = pdb[0]

    renderer = DisulfideBondRenderer(ss=ss1)

    # Display the visualization
    renderer.display(
        background_color="white",  # Background color
        style="sb",
        res=50,  # Resolution for spheres and cylinders
    )

    renderer.display(
        background_color="white",  # Background color
        style="cpk",
        res=50,  # Resolution for spheres and cylinders
    )
    renderer.display(
        background_color="white",  # Background color
        style="pd",
        res=50,  # Resolution for spheres and cylinders
    )

    renderer.display(
        background_color="white",  # Background color
        style="plain",
        res=50,  # Resolution for spheres and cylinders
    )

    # ss1.display()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    main()

# End of file
