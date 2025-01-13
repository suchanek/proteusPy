"""
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

The module provides the implmentation and interface for the [DisulfideList](#DisulfideList)
object, used extensively by Disulfide class.

Author: Eric G. Suchanek, PhD
Last revision: 2025-01-03 17:03:02 -egs-
"""

# pylint: disable=c0103
# pylint: disable=c0301
# pylint: disable=c0302
# pylint: disable=c0415
# pylint: disable=w0212

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

import copy
import logging
import os
from collections import UserList
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyvista as pv
from plotly.subplots import make_subplots

import proteusPy
from proteusPy import Disulfide
from proteusPy.atoms import BOND_RADIUS, FONTSIZE
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import MODEL_DIR, PBAR_COLS, PDB_DIR, WINSIZE
from proteusPy.utility import get_jet_colormap, get_theme, grid_dimensions

_logger = create_logger(__name__)


# Set the figure sizes and axis limits.
DPI = 220
WIDTH = 6.0
HEIGHT = 6.0
TORMIN = -179.9
TORMAX = 180.0


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

Distance_DF_Cols = [
    "source",
    "ss_id",
    "proximal",
    "distal",
    "energy",
    "ca_distance",
    "cb_distance",
    "sg_distance",
]


class DisulfideList(UserList):
    """
    The class provides a sortable list for Disulfide objects.
    Indexing and slicing are supported, as well as typical list operations like
    ``.insert()``, ``.append()`` and ``.extend().`` The DisulfideList object must be initialized
    with an iterable (tuple, list) and a name. Sorting is keyed by torsional energy.

    The class can also render Disulfides to a pyVista window using the
    [display()](#DisulfideList.display) and [display_overlay()](#DisulfideList.display_overlay)methods.
    See below for examples.\n

    Examples:
    >>> from proteusPy import Disulfide, DisulfideLoader, DisulfideList, Load_PDB_SS

    Instantiate some variables. Note: the list is initialifzed with an iterable and a name (optional)

    >>> SS = Disulfide('tmp')

    The list is initialized with an iterable, a name and resolution. Name and resolution
    are optional.
    >>> SSlist = DisulfideList([],'ss', -1.0)

    Load the database.
    >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)

    Get the first disulfide via indexing.
    >>> SS = PDB_SS[0]

    # assert str(SS) == "<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>"

    >>> SS4yys = PDB_SS['4yys']

    # assert str(SS4yys) == "[<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>]"

    Make some empty disulfides.
    >>> ss1 = Disulfide('ss1')
    >>> ss2 = Disulfide('ss2')

    Make a DisulfideList containing ss1, named 'tmp'
    >>> sslist = DisulfideList([ss1], 'tmp')
    >>> sslist.append(ss2)

    Extract the first disulfide
    >>> ss1 = PDB_SS[0]

    # assert str(ss1.pprint_all()) == "<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å\n Proximal Chain fullID: <('4yys', 0, 'A', (' ', 22, ' '))> Distal Chain fullID: <('4yys', 0, 'A', (' ', 65, ' '))>\nProximal Coordinates:\n   N: <Vector -2.36, -20.48, 5.21>\n   Cα: <Vector -2.10, -19.89, 3.90>\n   C: <Vector -1.12, -18.78, 4.12>\n   O: <Vector -1.30, -17.96, 5.03>\n   Cβ: <Vector -3.38, -19.31, 3.32>\n   Sγ: <Vector -3.24, -18.40, 1.76>\n   Cprev <Vector -2.67, -21.75, 5.36>\n   Nnext: <Vector -0.02, -18.76, 3.36>\n Distal Coordinates:\n   N: <Vector -0.60, -18.71, -1.62>\n   Cα: <Vector -0.48, -19.10, -0.22>\n   C: <Vector 0.92, -19.52, 0.18>\n   O: <Vector 1.10, -20.09, 1.25>\n   Cβ: <Vector -1.48, -20.23, 0.08>\n   Sγ: <Vector -3.22, -19.69, 0.18>\n   Cprev <Vector -0.73, -17.44, -2.01>\n   Nnext: <Vector 1.92, -19.18, -0.63>\n<BLANKLINE>\n Proximal Internal Coords:\n   N: <Vector -0.41, 1.40, -0.00>\n   Cα: <Vector 0.00, 0.00, 0.00>\n   C: <Vector 1.50, 0.00, 0.00>\n   O: <Vector 2.12, 0.71, -0.80>\n   Cβ: <Vector -0.50, -0.70, -1.25>\n   Sγ: <Vector 0.04, -2.41, -1.50>\n   Cprev <Vector -2.67, -21.75, 5.36>\n   Nnext: <Vector -0.02, -18.76, 3.36>\nDistal Internal Coords:\n   N: <Vector 1.04, -5.63, 1.17>\n   Cα: <Vector 1.04, -4.18, 1.31>\n   C: <Vector 1.72, -3.68, 2.57>\n   O: <Vector 1.57, -2.51, 2.92>\n   Cβ: <Vector -0.41, -3.66, 1.24>\n   Sγ: <Vector -1.14, -3.69, -0.43>\n   Cprev <Vector -0.73, -17.44, -2.01>\n   Nnext: <Vector 1.92, -19.18, -0.63>\n Χ1-Χ5: 174.63°, 82.52°, -83.32°, -62.52° -73.83°, 138.89°, 1.70 kcal/mol\n Cα Distance: 4.50 Å\n Torsion length: 231.53 deg>"

    Get a list of disulfides via slicing
    >>> subset = DisulfideList(PDB_SS[0:10],'subset')

    Display the subset disulfides overlaid onto the same coordinate frame,
    (proximal N, Ca, C').

    The disulfides are colored individually to facilitate inspection.

    >>> subset.display_overlay()
    """

    def __init__(self, iterable, pid: str = "nil", res=-1.0, quiet=True, fast=False):
        """
        Initialize the DisulfideList.

        :param iterable: An iterable of disulfide bonds.
        :type iterable: iterable
        :param pid: Name for the list, default is "nil".
        :type pid: str
        :param res: Resolution, default is -1.0. If -1, the average resolution is used.
        :type res: float
        :param quiet: If True, suppress output, default is True.
        :type quiet: bool
        :param fast: If True, enable fast mode, default is False.
        :type fast: bool

        Example:
        >>> from proteusPy import DisulfideList, Disulfide

        Initialize some empty disulfides.
        >>> ss1 = Disulfide('ss1')
        >>> ss2 = Disulfide('ss2')
        >>> ss3 = Disulfide('ss3')

        Make a list containing the disulfides.
        >>> sslist = DisulfideList([ss1, ss2], 'sslist')
        >>> sslist
        [<Disulfide ss1, Source: 1egs, Resolution: -1.0 Å>, <Disulfide ss2, Source: 1egs, Resolution: -1.0 Å>]
        >>> sslist.append(ss3)
        >>> sslist
        [<Disulfide ss1, Source: 1egs, Resolution: -1.0 Å>, <Disulfide ss2, Source: 1egs, Resolution: -1.0 Å>, <Disulfide ss3, Source: 1egs, Resolution: -1.0 Å>]
        """

        super().__init__(self.validate_ss(item) for item in iterable)

        self.pdb_id = pid
        self.quiet = quiet

        if not fast:
            if res == -1:
                self._res = self.average_resolution
            else:
                self._res = res
        else:
            self._res = res

    def __getitem__(self, item):
        """
        Retrieve a disulfide from the list. Internal only.

        :param item: Index or slice
        :return: Sublist
        """
        if isinstance(item, slice):
            indices = range(*item.indices(len(self.data)))
            ind_list = list(indices)
            first_ind = ind_list[0]
            last_ind = ind_list[-1]
            name = (
                self.data[first_ind].pdb_id
                + f"_slice[{first_ind}:{last_ind+1}]_{self.data[last_ind].pdb_id}"
            )
            sublist = [self.data[i] for i in indices]
            return DisulfideList(sublist, name)
        return UserList.__getitem__(self, item)

    def __setitem__(self, index, item):
        self.data[index] = self.validate_ss(item)

    # Rendering engine calculates and instantiates all bond
    # cylinders and atomic sphere meshes. Called by all high level routines

    def _render(self, pl, style, res=100) -> pv.Plotter:
        """
        Display a window showing the list of disulfides in the given style.
        :param style: one of 'cpk', 'bs', 'sb', 'plain', 'cov', 'pd'
        :return: Window in the relevant style
        """
        ssList = self.data
        tot_ss = len(ssList)  # number off ssbonds
        rows, cols = grid_dimensions(tot_ss)
        res = 100

        if tot_ss > 30:
            res = 60
        if tot_ss > 60:
            res = 30
        if tot_ss > 90:
            res = 12

        total_plots = rows * cols
        for idx in range(min(tot_ss, total_plots)):
            if not self.quiet:
                if idx % 5 == 0:
                    _logger.info("Rendering %d of %d bonds.", idx + 1, tot_ss)

            r = idx // cols
            c = idx % cols
            pl.subplot(r, c)

            ss = ssList[idx]
            src = ss.pdb_id
            enrg = ss.energy
            title = f"{src} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: E: {enrg:.2f}, Cα: {ss.ca_distance:.2f} Å, Tors: {ss.torsion_length:.2f}°"
            pl.add_title(title=title, font_size=FONTSIZE)
            ss._render(
                pl,
                style=style,
                res=res,
            )

        return pl

    @property
    def average_ca_distance(self):
        """
        Return the Average energy (kcal/mol) for the Disulfides in the list.

        :return: Average energy (kcal/mol) between all atoms in the list
        """
        sslist = self.data
        tot = len(sslist)
        if tot == 0:
            return 0.0

        total_dist = sum(ss.ca_distance for ss in sslist)
        return total_dist / tot

    @property
    def average_distance(self):
        """
        Return the Average distance (Å) between the atoms in the list.

        :return: Average distance (Å) between all atoms in the list

        """
        sslist = self.data
        cnt = 1

        total = 0.0
        for ss1 in sslist:
            for ss2 in sslist:
                if ss2 == ss1:
                    continue
                total += ss1.Distance_RMS(ss2)
                cnt += 1

        return total / cnt

    @property
    def average_energy(self):
        """
        Return the Average energy (kcal/mol) for the Disulfides in the list.

        :return: Average energy (kcal/mol) between all atoms in the list
        """
        sslist = self.data
        tot = len(sslist)
        if tot == 0:
            return 0.0

        total_energy = sum(ss.energy for ss in sslist)
        return total_energy / tot

    @property
    def average_conformation(self):
        """
        Return the average conformation for the disulfides in the list.

        :return: Average conformation: [x1, x2, x3, x4, x5]
        """
        sslist = self.data
        res = np.mean([ss.torsion_array for ss in sslist], axis=0)
        return res

    def append(self, item):
        """
        Append the list with item

        :param item: Disulfide to add
        :type item: Disulfide
        """
        self.data.append(self.validate_ss(item))

    @property
    def average_resolution(self) -> float:
        """
        Compute and return the average structure resolution for the given list.

        :return: Average resolution (A)
        """
        resolutions = [ss.resolution for ss in self.data if ss.resolution != -1.0]
        return sum(resolutions) / len(resolutions) if resolutions else -1.0

    @property
    def resolution(self) -> float:
        """
        Compute and return the average structure resolution for the given list.

        :return: Average resolution (A)
        """
        return self._res

    @resolution.setter
    def resolution(self, value: float):
        """
        Set the average structure resolution for the given list.

        :param value: The new resolution value to set.
        :type value: float
        """
        if not isinstance(value, float):
            raise TypeError("Resolution must be a float.")
        self._res = value

    @property
    def average_torsion_distance(self):
        """
        Return the average distance in torsion space (degrees), between all pairs in the
        DisulfideList

        :return: Torsion Distance (degrees)
        """
        sslist = self.data
        total = 0
        cnt = 0

        for ss1, ss2 in combinations(sslist, 2):
            total += ss1.torsion_distance(ss2)
            cnt += 1

        return float(total / cnt) if cnt > 0 else 0

    def build_distance_df(self) -> pd.DataFrame:
        """
        Create a dataframe containing the input DisulfideList Cα-Cα distance, energy.
        This can take several minutes for the entire database.

        :return: DataFrame containing Ca distances
        :rtype: pd.DataFrame
        """
        # create a list to collect rows as dictionaries
        rows = []
        i = 0
        sslist = self.data
        total_length = len(sslist)
        update_interval = max(1, total_length // 20)  # 5% of the list length

        if self.quiet:
            pbar = sslist
        else:
            pbar = tqdm(sslist, ncols=PBAR_COLS, leave=False)

        for ss in pbar:
            new_row = {
                "source": ss.pdb_id,
                "ss_id": ss.name,
                "proximal": ss.proximal,
                "distal": ss.distal,
                "energy": ss.energy,
                "ca_distance": ss.ca_distance,
                "cb_distance": ss.cb_distance,
                "sg_distance": ss.sg_distance,
            }
            rows.append(new_row)
            i += 1

            if not self.quiet:
                if i % update_interval == 0 or i == total_length - 1:
                    pbar.update(update_interval)

        # create the dataframe from the list of dictionaries
        SS_df = pd.DataFrame(rows, columns=Distance_DF_Cols)

        return SS_df

    # here we build a dataframe containing the torsional parameters

    def build_torsion_df(self) -> pd.DataFrame:
        """
        Create a dataframe containing the input DisulfideList torsional parameters,
        Cα-Cα distance, energy, and phi-psi angles. This can take several minutes for the
        entire database.

        :return: pd.DataFrame containing the torsions
        """
        # create a list to collect rows as dictionaries
        rows = []
        i = 0
        total_length = len(self.data)
        update_interval = max(1, total_length // 20)  # 5% of the list length

        sslist = self.data
        if self.quiet:
            pbar = sslist
        else:
            pbar = tqdm(sslist, ncols=PBAR_COLS, leave=False)

        for ss in pbar:
            new_row = {
                "source": ss.pdb_id,
                "ss_id": ss.name,
                "proximal": ss.proximal,
                "distal": ss.distal,
                "chi1": ss.chi1,
                "chi2": ss.chi2,
                "chi3": ss.chi3,
                "chi4": ss.chi4,
                "chi5": ss.chi5,
                "energy": ss.energy,
                "ca_distance": ss.ca_distance,
                "cb_distance": ss.cb_distance,
                "sg_distance": ss.sg_distance,
                "psi_prox": ss.psiprox,
                "phi_prox": ss.phiprox,
                "phi_dist": ss.phidist,
                "psi_dist": ss.psidist,
                "torsion_length": ss.torsion_length,
                "rho": ss.rho,
            }
            rows.append(new_row)
            i += 1

            if not self.quiet:
                if i % update_interval == 0 or i == total_length - 1:
                    pbar.update(update_interval)

        if not self.quiet:
            pbar.close()

        # create the dataframe from the list of dictionaries
        SS_df = pd.DataFrame(rows, columns=Torsion_DF_Cols)

        return SS_df

    def by_chain(self, chain: str):
        """
        Return a DisulfideList from the input chain identifier.

        :param chain: chain identifier, 'A', 'B, etc
        :return: DisulfideList containing disulfides within that chain.
        """

        reslist = DisulfideList([], chain)
        sslist = self.data

        for ss in sslist:
            pchain = ss.proximal_chain
            dchain = ss.distal_chain
            if pchain == dchain:
                if pchain == chain:
                    reslist.append(ss)
            else:
                print(f"Cross chain SS: {ss.repr_compact}:")
        return reslist

    @property
    def center_of_mass(self):
        """
        Calculate the center of mass for the Disulfide list
        """
        sslist = self.data
        tot = len(sslist)
        if tot == 0:
            return 0.0

        total_cofmass = sum(ss.cofmass for ss in sslist)
        return total_cofmass / tot

    def describe(self):
        """
        Prints out relevant attributes of the given disulfideList.

        :param disulfideList: A list of disulfide objects.
        :param list_name: The name of the list.
        """
        name = self.pdb_id
        avg_distance = self.average_ca_distance
        avg_energy = self.average_energy
        avg_resolution = self.average_resolution
        list_length = len(self.data)

        if list_length == 0:
            avg_bondangle = 0
            avg_bondlength = 0
        else:
            total_bondangle = 0
            total_bondlength = 0

            for ss in self.data:
                total_bondangle += ss.bond_angle_ideality
                total_bondlength += ss.bond_length_ideality

            avg_bondangle = total_bondangle / list_length
            avg_bondlength = total_bondlength / list_length

        print(f"DisulfideList: {name}")
        print(f"Length: {list_length}")
        print(f"Average energy: {avg_energy:.2f} kcal/mol")
        print(f"Average CA distance: {avg_distance:.2f} Å")
        print(f"Average Resolution: {avg_resolution:.2f} Å")
        print(f"Bond angle deviation: {avg_bondangle:.2f}°")
        print(f"Bond length deviation: {avg_bondlength:.2f} Å")

    def display(self, style="sb", light="Auto", panelsize=512):
        """
        Display the Disulfide list in the specific rendering style.

        :param single: Display the bond in a single panel in the specific style.
        :param style:  Rendering style: One of:\n
            - 'sb' - split bonds
            - 'bs' - ball and stick
            - 'cpk' - CPK style
            - 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            - 'plain' - boring single color
        :light: If True, light background, if False, dark
        """
        # from proteusPy.utility import get_theme

        pid = self.pdb_id
        ssbonds = self.data
        tot_ss = len(ssbonds)  # number off ssbonds
        rows, cols = grid_dimensions(tot_ss)
        winsize = (panelsize * cols, panelsize * rows)

        avg_enrg = self.average_energy
        avg_dist = self.average_distance
        resolution = self.average_resolution

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

        title = f"<{pid}> {resolution:.2f} Å: ({tot_ss} SS), Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"

        pl = pv.Plotter(window_size=winsize, shape=(rows, cols))
        pl = self._render(pl, style)
        pl.enable_anti_aliasing("msaa")
        pl.add_title(title=title, font_size=FONTSIZE)
        pl.link_views()
        pl.reset_camera()
        pl.show()

    def display_torsion_statistics(
        self,
        display=True,
        save=False,
        fname="ss_torsions.png",
        stats=False,
        light="Auto",
    ):
        """
        Display torsion and distance statistics for a given Disulfide list.

        :param display: Whether to display the plot in the notebook. Default is True.
        :type display: bool
        :param save: Whether to save the plot as an image file. Default is False.
        :type save: bool
        :param fname: The name of the image file to save. Default is 'ss_torsions.png'.
        :type fname: str
        :param stats: Whether to return the DataFrame representing the statistics for `self`. Default is False.
        :type stats: bool
        :param light: Whether to use the 'plotly_light' or 'plotly_dark' template. Default is True.
        :type light: bool
        :return: None
        """
        title = f"{self.id}: {self.length} members"

        df = self.torsion_df
        df_subset = df.iloc[:, 4:]
        df_stats = df_subset.describe()

        tor_vals, dist_vals = calculate_torsion_statistics(self)

        tor_mean_vals = tor_vals.loc["mean"]
        tor_std_vals = tor_vals.loc["std"]

        dist_mean_vals = dist_vals.loc["mean"]
        dist_std_vals = dist_vals.loc["std"]

        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.125, column_widths=[1, 1]
        )
        fig.update_layout(template="plotly" if light else "plotly_dark")

        fig.update_layout(
            title={
                "text": title,
                "xanchor": "center",
                # 'y':.9,
                "x": 0.5,
                "yanchor": "top",
            },
            width=1024,
            height=1024,
        )

        fig.add_trace(
            go.Bar(
                x=["X1", "X2", "X3", "X4", "X5"],
                y=tor_mean_vals[:5],
                name="Torsion Angle,(°) ",
                error_y=dict(type="data", array=tor_std_vals, visible=True),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=["rho"],
                y=[dist_mean_vals[4]],
                name="ρ, (°)",
                error_y=dict(type="data", array=[dist_std_vals[4]], visible=True),
            ),
            row=1,
            col=1,
        )

        # Update the layout of the subplot
        # Cα N, Cα, Cβ, C', Sγ Å °

        fig.update_yaxes(
            title_text="Dihedral Angle (°)", range=[-200, 200], row=1, col=1
        )
        fig.update_yaxes(range=[0, 320], row=2, col=2)

        # Add another subplot for the mean values of energy
        fig.add_trace(
            go.Bar(
                x=["Strain Energy (kcal/mol)"],
                y=[dist_mean_vals[3]],
                name="Energy (kcal/mol)",
                error_y=dict(
                    type="data",
                    array=[dist_std_vals[3].tolist()],
                    width=0.25,
                    visible=True,
                ),
            ),
            row=1,
            col=2,
        )
        fig.update_traces(width=0.25, row=1, col=2)

        # Update the layout of the subplot
        # fig.update_xaxes(title_text="Energy", row=1, col=2)
        fig.update_yaxes(
            title_text="kcal/mol", range=[0, 20], row=1, col=2
        )  # max possible DSE

        # Add another subplot for the mean values of ca_distance
        fig.add_trace(
            go.Bar(
                x=["Cα Distance, (Å)", "Cβ Distance, (Å)", "Sγ Distance, (Å)"],
                y=[dist_mean_vals[0], dist_mean_vals[1], dist_mean_vals[2]],
                name="Distances (Å)",
                error_y=dict(
                    type="data",
                    array=[
                        dist_std_vals[0].tolist(),
                        dist_std_vals[1].tolist(),
                        dist_std_vals[2].tolist(),
                    ],
                    width=0.25,
                    visible=True,
                ),
            ),
            row=2,
            col=1,
        )
        # Update the layout of the subplot
        fig.update_yaxes(title_text="Distance (A)", range=[0, 10], row=2, col=1)  #
        fig.update_traces(width=0.25, row=2, col=1)

        # Add a scatter subplot for torsion length column
        fig.add_trace(
            go.Bar(
                x=["Torsion Length, (Å)"],
                y=[tor_mean_vals[5]],
                name="Torsion Length, (Å)",
                error_y=dict(
                    type="data", array=[tor_std_vals[5]], width=0.25, visible=True
                ),
            ),
            row=2,
            col=2,
        )
        # Update the layout of the subplot
        fig.update_yaxes(title_text="Torsion Length", range=[0, 350], row=2, col=2)
        fig.update_traces(width=0.25, row=2, col=2)

        # Update the error bars
        fig.update_traces(
            error_y_thickness=2,
            error_y_color="gray",
            texttemplate="%{y:.2f} ± %{error_y.array:.2f}",
            textposition="outside",
        )  # , row=1, col=1)

        if display:
            fig.show()
        if save:
            fig.write_image(Path(fname))

        if stats:
            return df_stats

        return

    @property
    def distance_df(self) -> pd.DataFrame:
        """
        Build and return the distance dataframe for the input list.
        This can take considerable time for the entire list.

        :return: Dataframe containing the Cα-Cα distances for the given list.

        Example:
        >>> from proteusPy import Disulfide, Load_PDB_SS, DisulfideList
        >>> PDB_SS = Load_PDB_SS()

        """
        return self.build_distance_df()

    def display_overlay(
        self,
        screenshot=False,
        movie=False,
        verbose=False,
        fname="ss_overlay.png",
        light="Auto",
    ):
        """
        Display all disulfides in the list overlaid in stick mode against
        a common coordinate frames. This allows us to see all of the disulfides
        at one time in a single view. Colors vary smoothy between bonds.

        :param screenshot: Save a screenshot, defaults to False
        :param movie: Save a movie, defaults to False
        :param verbose: Verbosity, defaults to True
        :param fname: Filename to save for the movie or screenshot, defaults to 'ss_overlay.png'
        :param light: Background color, defaults to True for White. False for Dark.
        """

        # from proteusPy.utility import get_theme

        pid = self.pdb_id

        ssbonds = self.data
        tot_ss = len(ssbonds)  # number off ssbonds
        avg_enrg = self.average_energy
        avg_dist = self.average_distance
        resolution = self.average_resolution

        res = 64

        if tot_ss > 30:
            res = 48
        if tot_ss > 60:
            res = 16
        if tot_ss > 90:
            res = 8

        title = f"<{pid}> {resolution:.2f} Å: ({tot_ss} SS), Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"

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

        if movie:
            pl = pv.Plotter(window_size=WINSIZE, off_screen=True)
        else:
            pl = pv.Plotter(window_size=WINSIZE, off_screen=False)

        pl.add_title(title=title, font_size=FONTSIZE)
        pl.enable_anti_aliasing("msaa")
        # pl.add_camera_orientation_widget()
        pl.add_axes()

        mycol = np.zeros(shape=(tot_ss, 3))
        mycol = get_jet_colormap(tot_ss)

        # scale the overlay bond radii down so that we can see the individual elements better
        # maximum 90% reduction

        brad = BOND_RADIUS if tot_ss < 10 else BOND_RADIUS * 0.75
        brad = brad if tot_ss < 25 else brad * 0.8
        brad = brad if tot_ss < 50 else brad * 0.8
        brad = brad if tot_ss < 100 else brad * 0.6

        # print(f'Brad: {brad}')
        if verbose:
            pbar = tqdm(range(tot_ss), ncols=PBAR_COLS)
        else:
            pbar = range(tot_ss)

        for i, ss in zip(pbar, ssbonds):
            color = [int(mycol[i][0]), int(mycol[i][1]), int(mycol[i][2])]
            ss._render(
                pl,
                style="plain",
                bondcolor=color,
                translate=False,
                bond_radius=brad,
                res=res,
            )

        pl.reset_camera()

        if screenshot:
            pl.show(auto_close=False)  # allows for manipulation
            # Take the screenshot after ensuring the plotter is still active
            try:
                pl.screenshot(fname)
                if verbose:
                    print(f" -> display_overlay(): Saved image to: {fname}")
            except RuntimeError as e:
                _logger.error(f"Error saving screenshot: {e}")

        elif movie:
            if verbose:
                print(f" -> display_overlay(): Saving mp4 animation to: {fname}")

            pl.open_movie(fname)
            path = pl.generate_orbital_path(n_points=360)
            pl.orbit_on_path(path, write_frames=True)
            pl.close()

            if verbose:
                print(f" -> display_overlay(): Saved mp4 animation to: {fname}")
        else:
            pl.show()

        return

    def extend(self, other):
        """
        Extend the Disulfide list with other.

        :param other: extension
        :type item: DisulfideList
        """

        if isinstance(other, type(self)):
            self.data.extend(other)
        else:
            self.data.extend(self.validate_ss(item) for item in other)

    def filter_by_distance(self, distance: float = -1.0, minimum: float = 2.0):
        """
        Return a DisulfideList filtered by to between the maxium Ca distance and
        the minimum, which defaults to 2.0A.

        :param distance: Distance in Å
        :param minimum: Distance in Å
        :return: DisulfideList containing disulfides with the given distance.
        """

        reslist = []
        sslist = self.data

        # if distance is -1.0, return the entire list
        if distance == -1.0:
            return sslist.copy()

        reslist = [
            ss
            for ss in sslist
            if ss.ca_distance < distance and ss.ca_distance > minimum
        ]

        return DisulfideList(reslist, f"filtered by distance < {distance:.2f}")

    def filter_by_sg_distance(self, distance: float = -1.0, minimum: float = 1.0):
        """
        Return a DisulfideList filtered by to between the maxium Sg distance and
        the minimum, which defaults to 1.0A.

        :param distance: Distance in Å
        :param minimum: Distance in Å
        :return: DisulfideList containing disulfides with the given distance.
        """

        reslist = []
        sslist = self.data

        # if distance is -1.0, return the entire list
        if distance == -1.0:
            return sslist.copy()

        reslist = [
            ss
            for ss in sslist
            if ss.sg_distance < distance and ss.sg_distance > minimum
        ]

        return DisulfideList(reslist, f"filtered by Sγ distance < {distance:.2f}")

    def filter_by_bond_ideality(self, angle: float = -1.0):
        """
        Return a DisulfideList filtered by bond angle ideality between the maxium angle
        and the minimum, which defaults to 0.0°.

        :param angle: Angle in degrees
        :param minimum: Angle in degrees
        :return: DisulfideList containing disulfides with the given angle.
        """

        reslist = []
        sslist = self.data

        # if angle is -1.0, return the entire list
        if angle == -1.0:
            return sslist.copy()

        reslist = [ss for ss in sslist if ss.bond_angle_ideality < angle]

        return DisulfideList(reslist, f"filtered by bond angle < {angle:.2f}")

    def get_by_name(self, name):
        """
        Returns the Disulfide with the given name from the list.
        """
        for ss in self.data:
            if ss.name == name:
                return ss.copy()  # or ss.copy() !!!
        return None

    def get_chains(self):
        """
        Return the chain IDs for chains within the given Disulfide.
        :return: Chain IDs for given Disulfide
        """

        res_dict = {"xxx"}
        sslist = self.data

        for ss in sslist:
            pchain = ss.proximal_chain
            dchain = ss.distal_chain
            res_dict.update(pchain)
            res_dict.update(dchain)

        res_dict.remove("xxx")

        return res_dict

    def get_torsion_array(self):
        """
        Return a 2D NumPy array representing the dihedral angles in the given disulfide list.

        :return: A 2D NumPy array of shape (n, 5), where n is the number of disulfide bonds in the list. Each row
                of the array represents the dihedral angles of a disulfide bond, in the following order:
                [X1, X, X3, X4, X5], where i is the index of the disulfide bond in the list.
        """
        return np.array([ss.torsion_array for ss in self.data])

    def has_chain(self, chain) -> bool:
        """
        Returns True if given chain contained in Disulfide, False otherwise.
        :return: Returns True if given chain contained in Disulfide, False otherwise.
        """

        chns = {"xxx"}
        chns = self.get_chains()
        if chain in chns:
            return True
        else:
            return False

    @property
    def id(self):
        """
        PDB ID of the list
        """
        return self.pdb_id

    @id.setter
    def id(self, value):
        """
        Set the DisulfideList ID

        Parameters
        ----------
        value : str
            List ID
        """
        self.pdb_id = value

    def TorsionGraph(
        self, display=True, save=False, fname="ss_torsions.png", light="Auto"
    ):
        """
        Generate and optionally display or save a torsion graph.

        This method generates a torsion graph based on the torsion statistics
        of disulfide bonds. It can display the graph, save it to a file, or both.

        :param display: If True, the torsion graph will be displayed. Default is True.
        :type display: bool
        :param save: If True, the torsion graph will be saved to a file. Default is False.
        :type save: bool
        :param fname: The filename to save the torsion graph. Default is "ss_torsions.png".
        :type fname: str
        :param light: If True, a light theme will be used for the graph. Default is True.
        :type light: bool

        :return: None
        """
        # tor_stats, dist_stats = self.calculate_torsion_statistics()
        self.display_torsion_statistics(
            display=display, save=save, fname=fname, light=light
        )

    def translate(self, translation_vector) -> None:
        """
        Translate the DisulfideList by the given translation vector.
        Note: The translation is a vector SUBTRACTION, not addition.
        This is used primarily to move a list to its geometric center of mass
        and is a destructive operation, in the sense that it updates the list in place.

        :param translation_vector: The translation vector to apply.
        :type translation_vector: Vector3D
        """
        for ss in self.data:
            ss.translate(translation_vector)

    def insert(self, index, item):
        """
        Insert a Disulfide into the list at the specified index

        :param index: insertion point
        :type index: int
        :param item: Disulfide to insert
        :type item: Disulfide
        """
        self.data.insert(index, self.validate_ss(item))

    @property
    def length(self):
        """Return the length of the list"""
        return len(self.data)

    @property
    def min(self) -> Disulfide:
        """
        Return Disulfide from the list with the minimum energy

        :return: Disulfide with the minimum energy.
        """
        sslist = sorted(self.data)
        return sslist[0]

    @property
    def max(self) -> Disulfide:
        """
        Return Disulfide from the list with the maximum energy

        :return: Disulfide with the maximum energy. This assumes that
        the comparison is based on the energy attribute.
        """
        sslist = sorted(self.data)
        return sslist[-1]

    def minmax_distance(self):
        """
        Return the Disulfides with the minimum and
        maximum Cα distances in the list.

        :return: SSmin, SSmax
        """
        sslist = self.data

        if not sslist:
            return None, None

        ssmin = min(sslist, key=lambda ss: ss.ca_distance)
        ssmax = max(sslist, key=lambda ss: ss.ca_distance)

        return ssmin, ssmax

    @property
    def minmax_energy(self):
        """
        Return the Disulfides with the minimum and maximum energies
        from the DisulfideList.

        :return: Disulfides with minimum and maximum energies
        """
        sslist = self.data

        if not sslist:
            return None, None

        sslist = sorted(sslist, key=lambda ss: ss.energy)
        return sslist[0], sslist[-1]

    def nearest_neighbors(self, cutoff: float, *args):
        """
        Return all Disulfides within the given angle cutoff of the input Disulfide.

        :param cutoff: Distance cutoff, degrees
        :param args: Either 5 individual angles (chi1, chi2, chi3, chi4, chi5) or a list of 5 angles
        :return: DisulfideList of neighbors within the cutoff
        """
        if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 5:
            chi1, chi2, chi3, chi4, chi5 = args[0]
        elif len(args) == 5:
            chi1, chi2, chi3, chi4, chi5 = args
        else:
            raise ValueError(
                "You must provide either 5 individual angles or a list of 5 angles."
            )

        sslist = self.data
        modelss = proteusPy.Disulfide("model", torsions=[chi1, chi2, chi3, chi4, chi5])
        res = modelss.torsion_neighbors(sslist, cutoff)

        resname = f"Neighbors within {cutoff:.2f}° of [{', '.join(f'{angle:.2f}' for angle in modelss.dihedrals)}]"
        res.pdb_id = resname

        return res

    def nearest_neighbors_ss(self, ss, cutoff: float):
        """
        Return the list of Disulfides within the torsional cutoff
        of the input Disulfide.

        :param ss: Disulfide to compare to
        :param cutoff: Distance cutoff, degrees
        :return: DisulfideList of neighbors
        """

        sslist = self.data
        res = ss.torsion_neighbors(sslist, cutoff)

        resname = f"{ss.name} neighbors within {cutoff}°"
        res.pdb_id = resname

        return res

    def pprint(self):
        """
        Pretty print self.
        """
        sslist = self.data
        for ss in sslist:
            ss.pprint()

    def pprint_all(self):
        """
        Pretty print full disulfide descriptions in self.
        """
        sslist = self.data
        for ss in sslist:
            ss.pprint_all()

    @property
    def torsion_df(self):
        """Return the Torsion DataFrame for the DisulfideList"""
        return self.build_torsion_df()

    @property
    def torsion_array(self):
        """Return the Torsions as an Array"""
        return self.get_torsion_array()

    def validate_ss(self, value):
        """Return the Disulfide object if it is a Disulfide, otherwise raise an error"""
        from proteusPy.Disulfide import Disulfide

        if value is None:
            raise ValueError("The value cannot be None.")

        if not isinstance(value, Disulfide):
            raise TypeError("The value must be an instance of Disulfide.")
        return value

    def create_deviation_dataframe(self):
        """
        Create a DataFrame with columns PDB_ID, SS_Name, Angle_Deviation, Distance_Deviation,
        Ca Distance from a list of disulfides.

        :return: DataFrame containing the disulfide information.
        :rtype: pd.DataFrame
        """
        disulfide_list = self.data
        data = {
            "PDB_ID": [],
            "Resolution": [],
            "SS_Name": [],
            "Angle_Deviation": [],
            "Bondlength_Deviation": [],
            "Ca_Distance": [],
            "Sg_Distance": [],
        }

        # Collect data in batches
        pdb_ids = []
        resolutions = []
        ss_names = []
        angle_deviations = []
        bondlength_deviations = []
        ca_distances = []
        sg_distances = []

        for ss in tqdm(disulfide_list, desc="Processing..."):
            pdb_ids.append(ss.pdb_id)
            resolutions.append(ss.resolution)
            ss_names.append(ss.name)
            angle_deviations.append(ss.bond_angle_ideality)
            bondlength_deviations.append(ss.bond_length_ideality)
            ca_distances.append(ss.ca_distance)
            sg_distances.append(ss.sg_distance)

        # Extend the data dictionary in one go
        data["PDB_ID"].extend(pdb_ids)
        data["Resolution"].extend(resolutions)
        data["SS_Name"].extend(ss_names)
        data["Angle_Deviation"].extend(angle_deviations)
        data["Bondlength_Deviation"].extend(bondlength_deviations)
        data["Ca_Distance"].extend(ca_distances)
        data["Sg_Distance"].extend(sg_distances)

        df = pd.DataFrame(data)
        return df

    # class ends


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
    :param pdb_dir: Path to the PDB files, defaults to MODEL_DIR. This is: PDB_DIR/good and are
                    the pre-parsed PDB files that have been scanned by the DisulfideDownloader program.
    :param verbose: Print info while parsing.
    :param quiet: Suppress non-error logging output.
    :param dbg: Enable debug logging.
    :param cutoff: Distance cutoff for filtering disulfides.
    :param sg_cutoff: SG distance cutoff for filtering disulfides.
    :return: A DisulfideList of Disulfide objects initialized from the file.

    Example:

    PDB_DIR defaults to os.getenv('PDB').
    To load the Disulfides from the PDB ID 5rsa we'd use the following:

    >>> from proteusPy.DisulfideList import DisulfideList, load_disulfides_from_id
    >>> from proteusPy.ProteusGlobals import DATA_DIR
    >>> SSlist = DisulfideList([],'5rsa')
    >>> SSlist = load_disulfides_from_id('5rsa', pdb_dir=DATA_DIR, verbose=False)
    >>> SSlist
    [<Disulfide 5rsa_26A_84A, Source: 5rsa, Resolution: 2.0 Å>, <Disulfide 5rsa_40A_95A, Source: 5rsa, Resolution: 2.0 Å>, <Disulfide 5rsa_58A_110A, Source: 5rsa, Resolution: 2.0 Å>, <Disulfide 5rsa_65A_72A, Source: 5rsa, Resolution: 2.0 Å>]
    """

    from proteusPy.Disulfide import Initialize_Disulfide_From_Coords
    from proteusPy.ssparser import extract_ssbonds_and_atoms

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


def Ocalculate_torsion_statistics(sslist: DisulfideList):
    """
    Calculate and return the torsion and distance statistics for the DisulfideList.

    This method builds a DataFrame containing torsional parameters, Cα-Cα distance,
    energy, and phi-psi angles for the DisulfideList. It then calculates the mean
    and standard deviation for the torsional and distance parameters.

    :return: A tuple containing two DataFrames:
            - tor_stats: DataFrame with mean and standard deviation for torsional parameters.
            - dist_stats: DataFrame with mean and standard deviation for distance parameters.
    :rtype: tuple (pd.DataFrame, pd.DataFrame)
    """
    df = sslist.torsion_df

    tor_cols = ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
    dist_cols = ["ca_distance", "cb_distance", "sg_distance", "energy", "rho"]
    tor_stats = {}
    dist_stats = {}

    for col in tor_cols:
        tor_stats[col] = {"mean": (df[col]).mean(), "std": (df[col]).std()}

    for col in dist_cols:
        dist_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

    tor_stats = pd.DataFrame(tor_stats, columns=tor_cols)
    dist_stats = pd.DataFrame(dist_stats, columns=dist_cols)

    return tor_stats, dist_stats


def calculate_torsion_statistics(sslist):
    """
    Calculate and return the torsion and distance statistics for the DisulfideList.

    This method builds a DataFrame containing torsional parameters, Cα-Cα distance,
    energy, and phi-psi angles for the DisulfideList. It then calculates the mean
    and standard deviation for the torsional and distance parameters.

    :return: A tuple containing two DataFrames:
            - tor_stats: DataFrame with mean and standard deviation for torsional parameters.
            - dist_stats: DataFrame with mean and standard deviation for distance parameters.
    :rtype: tuple (pd.DataFrame, pd.DataFrame)
    """
    df = sslist.torsion_df

    tor_cols = ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
    dist_cols = ["ca_distance", "cb_distance", "sg_distance", "energy", "rho"]
    tor_stats = {}
    dist_stats = {}

    def circular_mean(series):
        """
        Calculate the circular mean of a series of angles.

        This function converts the input series of angles from degrees to radians,
        computes the mean of the sine and cosine of these angles, and then converts
        the result back to degrees.

        :param series: A sequence of angles in degrees.
        :type series: array-like
        :return: The circular mean of the input angles in degrees.
        :rtype: float
        """
        radians = np.deg2rad(series)
        sin_mean = np.sin(radians).mean()
        cos_mean = np.cos(radians).mean()
        return np.rad2deg(np.arctan2(sin_mean, cos_mean))

    for col in tor_cols[:5]:
        tor_stats[col] = {"mean": circular_mean(df[col]), "std": df[col].std()}

    tor_stats["torsion_length"] = {
        "mean": df["torsion_length"].mean(),
        "std": df["torsion_length"].std(),
    }

    for col in dist_cols:
        dist_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

    tor_stats = pd.DataFrame(tor_stats, columns=tor_cols)
    dist_stats = pd.DataFrame(dist_stats, columns=dist_cols)

    return tor_stats, dist_stats


def extract_disulfide(
    pdb_filename: str, verbose=False, quiet=True, pdbdir=PDB_DIR
) -> DisulfideList:
    """
    Read the PDB file represented by `pdb_filename` and return a `DisulfideList`
    containing the Disulfide bonds found.

    :param pdb_filename:   The filename of the PDB file to read.
    :param verbose:        Display more messages (default: False).
    :param quiet:          Turn off DisulfideConstruction warnings (default: True).
    :param pdbdir:         Path to PDB files (default: PDB_DIR).
    :return:               A `DisulfideList` containing the Disulfide bonds found.
    :rtype:                DisulfideList
    """

    def extract_id_from_filename(filename: str) -> str:
        """
        Extract the ID from a filename formatted as 'pdb{id}.ent'.

        :param filename: The filename to extract the ID from.
        :type filename: str
        :return: The extracted ID.
        :rtype: str
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
