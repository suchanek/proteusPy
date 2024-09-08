"""
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

The module provides the implmentation and interface for the [DisulfideList](#DisulfideList)
object, used extensively by Disulfide class.

Author: Eric G. Suchanek, PhD
Last revision: 7/12/2024 -egs-
"""

# pylint: disable=c0103
# pylint: disable=c0301

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
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyvista as pv
from plotly.subplots import make_subplots

import proteusPy
from proteusPy import Disulfide
from proteusPy.atoms import *
from proteusPy.logger_config import get_logger
from proteusPy.ProteusGlobals import MODEL_DIR, PBAR_COLS, WINSIZE
from proteusPy.utility import get_jet_colormap, grid_dimensions

_logger = get_logger(__name__)


# Set the figure sizes and axis limits.
DPI = 220
WIDTH = 6.0
HEIGHT = 6.0
TORMIN = -179.0
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

    Instantiate some variables. Note: the list is initialized with an iterable and a name (optional)

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

    def __init__(self, iterable, pid: str, res=-1.0, quiet=True):
        """
        Initialize the DisulfideList

        :param iterable: an iterable e.g. []
        :type iterable: iterable
        :param id: Name for the list
        :type id: str

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
        total = 0
        count = 0

        if res == -1:
            for ss in iterable:
                if ss.resolution is not None:
                    total += ss.resolution
                    count += 1
            if count != 0:
                self.res = total / count
            else:
                self.res = -1.0
        else:
            self.res = res

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

    def _render(self, style, panelsize=256) -> pv.Plotter:
        """
        Display a window showing the list of disulfides in the given style.
        :param style: one of 'cpk', 'bs', 'sb', 'plain', 'cov', 'pd'
        :return: Window in the relevant style
        """
        ssList = self.data
        tot_ss = len(ssList)  # number off ssbonds
        rows, cols = grid_dimensions(tot_ss)
        winsize = (panelsize * cols, panelsize * rows)

        pl = pv.Plotter(window_size=winsize, shape=(rows, cols))
        i = 0

        for r in range(rows):
            for c in range(cols):
                pl.subplot(r, c)
                if i < tot_ss:
                    # ss = Disulfide()
                    ss = ssList[i]
                    src = ss.pdb_id
                    enrg = ss.energy
                    title = f"{src} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: E: {enrg:.2f}, Cα: {ss.ca_distance:.2f} Å, Tors: {ss.torsion_length:.2f}°"
                    pl.add_title(title=title, font_size=FONTSIZE)
                    ss._render(
                        pl,
                        style=style,
                        bondcolor=BOND_COLOR,
                        bs_scale=BS_SCALE,
                        spec=SPECULARITY,
                        specpow=SPEC_POWER,
                    )
                i += 1
        return pl

    @property
    def Average_Distance(self):
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
    def Average_Energy(self):
        """
        Return the Average energy (kcal/mol) for the Disulfides in the list.

        :return: Average energy (kcal/mol) between all atoms in the list

        """
        sslist = self.data
        tot = len(sslist)
        if tot == 0:
            return 0.0

        total = 0.0
        for ss1 in sslist:
            total += ss1.energy

        return total / tot

    @property
    def Average_Conformation(self):
        """
        Return the Average conformation for the Disulfides in the list.

        :return: Average conformation: [x1, x2, x3, x4, x5]
        """

        sslist = self.data
        tot = len(sslist)
        res = np.zeros(5)

        for ss, i in zip(sslist, range(tot)):
            res += ss.torsion_array

        return res / tot

    def append(self, item):
        """
        Append the list with item

        :param item: Disulfide to add
        :type item: Disulfide
        """
        self.data.append(self.validate_ss(item))

    @property
    def Average_Resolution(self) -> float:
        """
        Compute and return the average structure resolution for the given list.

        :return: Average resolution (A)
        """
        res = 0.0
        cnt = 1

        for ss in self.data:
            _res = ss.resolution
            if _res is not None and _res != -1.0:
                res += _res
                cnt += 1
        return res / cnt if cnt else -1.0

    @property
    def Average_Torsion_Distance(self):
        """
        Return the average distance in torsion space (degrees), between all pairs in the
        DisulfideList

        :return: Torsion Distance (degrees)
        """
        sslist = self.data
        total = 0
        cnt = 1

        for ss1 in sslist:
            for ss2 in sslist:
                if ss2 == ss1:
                    continue
                total += ss1.Torsion_Distance(ss2)
                cnt += 1

        return total / cnt

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
            }
            rows.append(new_row)
            i += 1

            if i % update_interval == 0 or i == total_length - 1:
                pbar.update(update_interval)

        pbar.close()

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
                "psiprox": ss.psiprox,
                "phiprox": ss.phiprox,
                "phidist": ss.phidist,
                "psidist": ss.psidist,
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

    def calculate_torsion_statistics(self):
        df = self.build_torsion_df()

        # df_subset = df.iloc[:, 4:]
        # df_stats = df_subset.describe()

        # print(df_stats.head())

        # mean_vals = df_stats.loc["mean"].values
        # std_vals = df_stats.loc["std"].values

        tor_cols = ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
        dist_cols = ["ca_distance", "cb_distance", "energy"]
        tor_stats = {}
        dist_stats = {}

        for col in tor_cols:
            tor_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

        for col in dist_cols:
            dist_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

        tor_stats = pd.DataFrame(tor_stats, columns=tor_cols)
        dist_stats = pd.DataFrame(dist_stats, columns=dist_cols)

        return tor_stats, dist_stats

    def display(self, style="sb", light=True, panelsize=512):
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
        pid = self.pdb_id
        ssbonds = self.data
        tot_ss = len(ssbonds)  # number off ssbonds
        avg_enrg = self.Average_Energy
        avg_dist = self.Average_Distance
        resolution = self.resolution

        if light:
            pv.set_plot_theme("document")
        else:
            pv.set_plot_theme("dark")

        title = f"<{pid}> {resolution:.2f} Å: ({tot_ss} SS), Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"

        pl = pv.Plotter()
        pl = self._render(style, panelsize)
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
        light=True,
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

        # print(df_stats.head())

        mean_vals = df_stats.loc["mean"].values
        std_vals = df_stats.loc["std"].values

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
                y=mean_vals[:5],
                name="Torsion Angle,(°) ",
                error_y=dict(type="data", array=std_vals[:5], visible=True),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=["rho"],
                y=[mean_vals[13]],
                name="ρ, (°)",
                error_y=dict(type="data", array=[std_vals[13]], visible=True),
            ),
            row=1,
            col=1,
        )

        # Update the layout of the subplot
        # Cα N, Cα, Cβ, C', Sγ Å °

        fig.update_yaxes(
            title_text="Torsion Angle (°)", range=[-200, 200], row=1, col=1
        )
        fig.update_yaxes(range=[0, 320], row=2, col=2)

        # Add another subplot for the mean values of energy
        fig.add_trace(
            go.Bar(
                x=["Strain Energy (kcal/mol)"],
                y=[mean_vals[5]],
                name="Energy (kcal/mol)",
                error_y=dict(
                    type="data",
                    array=[std_vals[5].tolist()],
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
                x=["Cα Distance, (Å)", "Cβ Distance, (Å)"],
                y=[mean_vals[6], mean_vals[7]],
                name="Cβ Distance (Å)",
                error_y=dict(
                    type="data",
                    array=[std_vals[6].tolist(), std_vals[7].tolist()],
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
                y=[mean_vals[12]],
                name="Torsion Length, (Å)",
                error_y=dict(
                    type="data", array=[std_vals[12]], width=0.25, visible=True
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
        verbose=True,
        fname="ss_overlay.png",
        light=True,
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

        pid = self.pdb_id
        ssbonds = self.data
        tot_ss = len(ssbonds)  # number off ssbonds
        avg_enrg = self.Average_Energy
        avg_dist = self.Average_Distance
        resolution = self.resolution

        res = 100

        if tot_ss > 100:
            res = 60
        if tot_ss > 200:
            res = 30
        if tot_ss > 300:
            res = 8

        title = f"<{pid}> {resolution:.2f} Å: ({tot_ss} SS), Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"

        if light:
            pv.set_plot_theme("document")
        else:
            pv.set_plot_theme("dark")

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
        pbar = tqdm(range(tot_ss), ncols=PBAR_COLS)

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
            pl.screenshot(fname)

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

    def filter_by_distance(self, distance: float, minimum: float = 2.0):
        """
        Return a DisulfideList filtered by to between the maxium Ca distance and
        the minimum, which defaults to 2.0A.

        :param distance: Distance in Å
        :param minimum: Distance in Å
        :return: DisulfideList containing disulfides with the given distance.
        """

        reslist = DisulfideList([], f"filtered by distance < {distance:.2f}")
        sslist = self.data

        # if distance is -1.0, return the entire list
        if distance == -1.0:
            return sslist.copy()

        reslist = [
            ss
            for ss in sslist
            if ss.ca_distance < distance and ss.ca_distance > minimum
        ]

        return reslist

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
        Returns a 2D NumPy array representing the dihedral angles in the given disulfide list.

        :return: A 2D NumPy array of shape (n, 5), where n is the number of disulfide bonds in the list. Each row
                of the array represents the dihedral angles of a disulfide bond, in the following order:
                [X1_i, X2_i, X3_i, X4_i, X5_i], where i is the index of the disulfide bond in the list.
        """
        sslist = self.data
        tot = len(sslist)
        res = np.zeros(shape=(tot, 5))

        for idx, ss in enumerate(sslist):
            row = ss.torsion_array
            res[idx, :] = row

        return res

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

    @property
    def resolution(self):
        """
        Resolution of the parent sturcture (A)
        """
        return self.res

    @resolution.setter
    def resolution(self, value):
        """
        Set the resolution of the list

        :param value: Resolution (A)
        """
        self.res = value

    def TorsionGraph(
        self, display=True, save=False, fname="ss_torsions.png", light=True
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

        :return: Disulfide with the maximum energy.
        """
        sslist = sorted(self.data)
        return sslist[-1]

    def minmax_distance(self):
        """
        Return the Disulfides with the minimum and
        maximum Cα distances in the list.

        :return: SSmin, SSmax
        """

        _min = 99999.9
        _max = -99999.9

        sslist = self.data
        ssmin = 0
        ssmax = 0
        idx = 0

        pbar = tqdm(sslist, ncols=PBAR_COLS)
        for ss in pbar:
            dist = ss.ca_distance

            if dist >= _max:
                ssmax = idx
                _max = dist

            if dist <= _min:
                ssmin = idx
                _min = dist
            idx += 1

        return sslist[ssmin], sslist[ssmax]

    @property
    def minmax_energy(self):
        """
        Return the Disulfides with the minimum and maximum energies
        from the DisulfideList.

        :return: Disulfide with the given ID
        """
        sslist = sorted(self.data)
        return sslist[0], sslist[-1]

    def nearest_neighbors(
        self,
        chi1: float,
        chi2: float,
        chi3: float,
        chi4: float,
        chi5: float,
        cutoff: float,
    ):
        """
        Given a torsional array of chi1-chi5,

        :param chi1: Chi1 (degrees)
        :param chi2: Chi2 (degrees)
        :param chi3: Chi3 (degrees)
        :param chi4: Chi4 (degrees)
        :param chi5: Chi5 (degrees)
        :param cutoff: Distance cutoff, degrees
        :return: DisulfideList of neighbors
        """

        sslist = self.data
        modelss = proteusPy.Disulfide("model")

        modelss.build_model(chi1, chi2, chi3, chi4, chi5)
        res = DisulfideList([], "neighbors")
        res = modelss.Torsion_neighbors(sslist, cutoff)

        return res

    def nearest_neighbors_ss(self, ss, cutoff: float):
        """
        Given an input Disulfide and overall torsional cutoff, return
        the list of Disulfides within the cutoff

        :param ss: Disulfide to compare to
        :param cutoff: Distance cutoff, degrees
        :return: DisulfideList of neighbors
        """

        chi1 = ss.chi1
        chi2 = ss.chi2
        chi3 = ss.chi3
        chi4 = ss.chi4
        chi5 = ss.chi5

        sslist = self.data
        modelss = proteusPy.Disulfide("model")

        modelss.build_model(chi1, chi2, chi3, chi4, chi5)
        res = DisulfideList([], "neighbors")
        res = modelss.Torsion_neighbors(sslist, cutoff)

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
        return self.build_torsion_df()

    @property
    def torsion_array(self):
        return self.get_torsion_array()

    def validate_ss(self, value):
        return value

    # class ends


def load_disulfides_from_id(
    pdb_id: str,
    pdb_dir=MODEL_DIR,
    model_numb=0,
    verbose=False,
    quiet=True,
    dbg=False,
    cutoff=-1.0,
) -> DisulfideList:
    """
    Loads the Disulfides by PDB ID and returns a ```DisulfideList``` of Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.

    *NB:* Requires EGS-Modified BIO.parse_pdb_header.py from https://github.com/suchanek/biopython

    :param pdb_id: the name of the PDB entry.
    :param pdb_dir: path to the PDB files, defaults to MODEL_DIR - this is: PDB_DIR/good and are
    the pre-parsed PDB files that have been scanned by the DisulfideDownloader program.
    :param model_numb: model number to use, defaults to 0 for single structure files.
    :param verbose: print info while parsing
    :return: a list of Disulfide objects initialized from the file.

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
    errors = 0
    resolution = -1.0

    structure_fname = os.path.join(pdb_dir, f"pdb{pdb_id}.ent")
    # model = structure[model_numb]

    if verbose:
        mess = f"-> load_disulfide_from_id() - Parsing structure: {pdb_id}:"
        _logger.info(mess)

    SSList = DisulfideList([], pdb_id, resolution)

    # list of tuples with (proximal distal chaina chainb)
    # ssbonds = parse_ssbond_header_rec(ssbond_dict)

    ssbond_atom_list, num_ssbonds, errors = extract_ssbonds_and_atoms(
        structure_fname, verbose=verbose
    )

    if num_ssbonds == 0:
        if verbose:
            mess = f"-> load_disulfides_from_id(): {pdb_id} has no SSBonds."
            print(mess)
        _logger.warning(mess)
        return None

    if verbose:
        mess = f"-> load_disulfides_from_id(): {pdb_id} has {num_ssbonds} SSBonds, found: {errors} errors"
        _logger.info(mess)

    # with warnings.catch_warnings():
    if quiet:
        _logger.setLevel(logging.ERROR)

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
            _logger.info(mess)

        proximal_int = int(proximal)
        distal_int = int(distal)

        if proximal == distal:
            if verbose:
                mess = f"-> load_disulfides_from_id(): SSBond record has (proximal == distal):\
                {pdb_id} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}."
                _logger.info(mess)

        if verbose:
            mess = f"-> load_disulfides_from_id(): SSBond: {i}: {pdb_id}: {proximal} {chain1_id} - {distal} {chain2_id}"
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
        if verbose:
            _logger.info("New SS: %s", new_ss)

        if new_ss is not None:
            SSList.append(new_ss)
            if verbose:
                mess = f"-> load_disulfides_from_id(): Initialized Disulfide: {pdb_id} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}."
                _logger.info(mess)
        else:
            mess = f"-> load_disulfides_from_id(): Cannot initialize Disulfide: {pdb_id} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}."
            _logger.ERROR(mess)

        i += 1

    if quiet:
        _logger.setLevel(logging.WARNING)

    if cutoff > 0:
        SSList = SSList.filter_by_distance(cutoff)

    return copy.deepcopy(SSList)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
