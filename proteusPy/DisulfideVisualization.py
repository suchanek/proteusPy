"""
This module provides visualization functionality for disulfide bonds in the proteusPy package.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-12
"""

# pylint: disable=C0301
# pylint: disable=C0103
# pylint: disable=W0212

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyvista as pv
from plotly.subplots import make_subplots
from scipy import stats
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from proteusPy.atoms import BOND_RADIUS
from proteusPy.DisulfideClassManager import DisulfideClassManager
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import NBINS, PBAR_COLS, WINSIZE
from proteusPy.utility import (
    calculate_fontsize,
    get_jet_colormap,
    grid_dimensions,
    set_plotly_theme,
    set_pyvista_theme,
)

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        tqdm = tqdm_notebook
except NameError:
    pass  # Use default tqdm import

_logger = create_logger(__name__)


class DisulfideVisualization:
    """Provides visualization methods for Disulfide bonds, including 3D rendering,
    statistical plots, and overlay displays."""

    @staticmethod
    def enumerate_class_fromlist(tclass: DisulfideClassManager, sslist, base=8):
        """Enumerate the classes from a list of class IDs.

        :param tclass: DisulfideClassManager instance
        :param sslist: List of class IDs to enumerate
        :param base: Base for class IDs (2 or 8)
        :return: DataFrame with class IDs and counts
        """
        x = []
        y = []

        for cls in sslist:
            if cls is not None:
                _y = tclass.sslist_from_classid(cls, base=base)
                # it's possible to have 0 SS in a class
                if _y is not None:
                    # only append if we have both.
                    x.append(cls)
                    y.append(len(_y))

        sslist_df = pd.DataFrame(columns=["class_id", "count"])
        sslist_df["class_id"] = x
        sslist_df["count"] = y
        return sslist_df

    @staticmethod
    def plot_classes_vs_cutoff(
        tclass: DisulfideClassManager,
        cutoff: float,
        steps: int = 50,
        base=8,
        theme="auto",
        verbose=False,
    ) -> None:
        """Plot the total percentage and number of members for each octant class against the cutoff value.

        :param tclass: DisulfideClassManager instance
        :param cutoff: Percent cutoff value for filtering the classes
        :param steps: Number of steps to take in the cutoff
        :param base: The base class to use, 6 or 8
        :param theme: The theme to use for the plot ('auto', 'light', or 'dark')
        :param verbose: Whether to display verbose output
        :return: None
        """
        _cutoff = np.linspace(0, cutoff, steps)
        tot_list = []
        members_list = []
        base_str = "Octant" if base == 8 else "Binary"

        set_plotly_theme(theme)

        for c in _cutoff:
            class_df = tclass.filter_class_by_percentage(c, base=base)
            tot = class_df["percentage"].sum()
            tot_list.append(tot)
            members_list.append(class_df.shape[0])
            if verbose:
                print(
                    f"Cutoff: {c:5.3} accounts for {tot:7.2f}% and is {class_df.shape[0]:5} members long."
                )

        fig = go.Figure()

        # Add total percentage trace
        fig.add_trace(
            go.Scatter(
                x=_cutoff,
                y=tot_list,
                mode="lines+markers",
                name="Total percentage",
                yaxis="y1",
                line=dict(color="blue"),
            )
        )

        # Add number of members trace
        fig.add_trace(
            go.Scatter(
                x=_cutoff,
                y=members_list,
                mode="lines+markers",
                name="Number of members",
                yaxis="y2",
                line=dict(color="red"),
            )
        )

        # Update layout
        fig.update_layout(
            title={
                "text": f"{base_str} Classes vs Cutoff, ({cutoff}%)",
                "x": 0.5,
                "yanchor": "top",
                "xanchor": "center",
            },
            xaxis=dict(title="Cutoff"),
            yaxis=dict(
                title="Total percentage",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
            ),
            yaxis2=dict(
                title="Number of members",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                overlaying="y",
                side="right",
                type="log",
            ),
            legend=dict(x=0.75, y=1.16),
        )

        fig.show()

    @staticmethod
    def plot_binary_to_eightclass_incidence(
        tclass: DisulfideClassManager,
        theme="light",
        save=False,
        savedir=".",
        verbose=False,
    ):
        """Plot the incidence of all octant Disulfide classes for a given binary class.

        :param tclass: DisulfideClassManager instance
        :param theme: The theme to use for the plot
        :param save: Whether to save the plots
        :param savedir: Directory to save plots to
        :param verbose: Whether to display verbose output
        """
        if verbose:
            _logger.setLevel("INFO")

        clslist = tclass.binaryclass_df["class_id"]
        for cls in clslist:
            eightcls = tclass.binary_to_class(cls, 8)
            df = DisulfideVisualization.enumerate_class_fromlist(
                tclass, eightcls, base=8
            )
            DisulfideVisualization.plot_count_vs_class_df(
                df,
                title=cls,
                theme=theme,
                save=save,
                savedir=savedir,
                base=8,
                verbose=verbose,
            )
        if verbose:
            _logger.info("Graph generation complete.")
            _logger.setLevel("WARNING")

    @staticmethod
    def plot_count_vs_class_df(
        df,
        title="title",
        theme="auto",
        save=False,
        savedir=".",
        base=8,
        verbose=False,
        log=True,
    ):
        """Plot a line graph of count vs class ID using Plotly for the given disulfide class.

        :param df: DataFrame containing class data
        :param title: Title for the plot
        :param theme: Theme to use for the plot
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param base: Base for class IDs (2 or 8)
        :param verbose: Whether to display verbose output
        :param log: Whether to use log scale for y-axis
        """
        set_plotly_theme(theme)

        _title = f"Binary Class: {title}"
        _labels = {}
        _prefix = "None"
        if base == 8:
            _labels = {"class_id": "Octant Class ID", "count": "Count"}
            _prefix = "Octant"
        elif base == 2:
            _labels = {"class_id": "Binary Class ID", "count": "Count"}
            _prefix = "Binary"
        else:
            raise ValueError("Invalid base. Must be 2 or 8.")

        fig = px.line(
            df,
            x="class_id",
            y="count",
            title=f"{_title}",
            labels=_labels,
        )

        fig.update_layout(
            showlegend=True,
            title_x=0.5,
            title_font=dict(size=20),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            autosize=True,
            yaxis_type="log" if log else "linear",
        )
        fig.update_layout(autosize=True)

        if save:
            fname = Path(savedir) / f"{title}_{_prefix}.png"

            if verbose:
                _logger.info("Saving %s plot to %s", title, fname)
            fig.write_image(fname, "png")
        else:
            fig.show()

    @staticmethod
    def plot_count_vs_class_df_sampled(
        df,
        title="title",
        theme="auto",
        save=False,
        savedir=".",
        base=8,
        verbose=False,
        log=True,
        sample_size=1000,
    ):
        """Plot a line graph of count vs class ID using Plotly with sampling.

        :param df: DataFrame containing class data
        :param title: Title for the plot
        :param theme: Theme to use for the plot
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param base: Base for class IDs (2 or 8)
        :param verbose: Whether to display verbose output
        :param log: Whether to use log scale for y-axis
        :param sample_size: Number of items to sample
        """
        set_plotly_theme(theme)

        _title = f"Binary Class: {title}"
        _labels = {}
        _prefix = "None"
        if base == 8:
            _labels = {"class_id": "Octant Class ID", "count": "Count"}
            _prefix = "Octant"
        elif base == 2:
            _labels = {"class_id": "Binary Class ID", "count": "Count"}
            _prefix = "Binary"
        else:
            raise ValueError("Invalid base. Must be 2 or 8.")

        df_sampled = df.sample(n=sample_size)

        fig = px.line(
            df_sampled,
            x="class_id",
            y="count",
            title=f"{_title} (Sampled {sample_size} items)",
            labels=_labels,
        )

        fig.update_layout(
            showlegend=True,
            title_x=0.5,
            title_font=dict(size=20),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            autosize=True,
            yaxis_type="log" if log else "linear",
        )
        fig.update_layout(autosize=True)

        if save:
            fname = Path(savedir) / f"{title}_{_prefix}_sampled.png"

            if verbose:
                _logger.info("Saving %s plot to %s", title, fname)
            fig.write_image(fname, "png")
        else:
            fig.show()

    @staticmethod
    def plot_count_vs_class_df_paginated(
        df,
        title="title",
        theme="auto",
        save=False,
        savedir=".",
        base=8,
        verbose=False,
        log=True,
        page_size=200,
    ):
        """Plot a line graph of count vs class ID using Plotly with pagination.

        :param df: DataFrame containing class data
        :param title: Title for the plot
        :param theme: Theme to use for the plot
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param base: Base for class IDs (2 or 8)
        :param verbose: Whether to display verbose output
        :param log: Whether to use log scale for y-axis
        :param page_size: Number of items per page
        """
        set_plotly_theme(theme)

        _title = f"Binary Class: {title}"
        _labels = {}
        _prefix = "None"
        if base == 8:
            _labels = {"class_id": "Octant Class ID", "count": "Count"}
            _prefix = "Octant"
        elif base == 2:
            _labels = {"class_id": "Binary Class ID", "count": "Count"}
            _prefix = "Binary"
        else:
            raise ValueError("Invalid base. Must be 2 or 8.")

        total_pages = (len(df) + page_size - 1) // page_size

        for page in range(total_pages):
            start = page * page_size
            end = start + page_size
            df_page = df.iloc[start:end]

            fig = px.line(
                df_page,
                x="class_id",
                y="count",
                title=f"{_title} (Page {page + 1}/{total_pages})",
                labels=_labels,
            )

            fig.update_layout(
                showlegend=True,
                title_x=0.5,
                title_font=dict(size=20),
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                autosize=True,
                yaxis_type="log" if log else "linear",
            )
            fig.update_layout(autosize=True)

            if save:
                fname = Path(savedir) / f"{title}_{_prefix}_page_{page + 1}.png"

                if verbose:
                    _logger.info("Saving %s plot to %s", title, fname)
                fig.write_image(fname, "png")
            else:
                fig.show()

    @staticmethod
    def plot_count_vs_classid(
        tclass: DisulfideClassManager, cls=None, theme="auto", base=8, log=True
    ):
        """Plot a line graph of count vs class ID using Plotly.

        :param tclass: DisulfideClassManager instance
        :param cls: Specific class to plot (optional)
        :param theme: Theme to use for the plot
        :param base: Base for class IDs (2 or 8)
        :param log: Whether to use log scale for y-axis
        """
        set_plotly_theme(theme)

        _title = None

        match base:
            case 8:
                _title = "Octant Class Distribution"
            case 2:
                _title = "Binary Class Distribution"
            case _:
                raise ValueError("Invalid base. Must be 2 or 8")

        df = tclass.binaryclass_df if base == 2 else tclass.eightclass_df

        if cls is None:
            fig = px.line(df, x="class_id", y="count", title=_title)
        else:
            subset = df[df["class_id"] == cls]
            fig = px.line(subset, x="class_id", y="count", title=_title)

        fig.update_layout(
            xaxis_title="Class ID",
            yaxis_title="Count",
            showlegend=True,
            title_x=0.5,
            autosize=True,
            yaxis_type="log" if log else "linear",
        )

        fig.show()

    @staticmethod
    def plot_disulfides_vs_pdbid(ssdict, cutoff=1):
        """Plot the number of disulfides versus PDB ID.

        :param ssdict: Dictionary mapping PDB IDs to disulfide indices
        :param cutoff: Minimum number of disulfides required for inclusion
        :return: Tuple of (pdb_ids, num_disulfides)
        """
        pdbids = []
        num_disulfides = []

        for pdbid, disulfides in ssdict.items():
            if len(disulfides) > cutoff:
                pdbids.append(pdbid)
                num_disulfides.append(len(disulfides))

        # Create a DataFrame
        df = pd.DataFrame({"PDB ID": pdbids, "Number of Disulfides": num_disulfides})
        fig = px.bar(
            df,
            x="PDB ID",
            y="Number of Disulfides",
            title=f"Disulfides vs PDB ID with cutoff: {cutoff}, {len(pdbids)} PDB IDs",
        )
        fig.update_layout(
            xaxis_title="PDB ID",
            yaxis_title="Number of Disulfides",
            xaxis_tickangle=-90,
        )
        fig.show()

        return pdbids, num_disulfides

    @staticmethod
    def plot_classes(
        tclass: DisulfideClassManager,
        class_string,
        base=8,
        theme="auto",
        log=False,
        page_size=200,
    ):
        """Plot the distribution of classes for the given class string.

        :param tclass: DisulfideClassManager instance
        :param class_string: The class string to plot
        :param base: Base for class IDs (2 or 8)
        :param theme: Theme to use for the plot
        :param log: Whether to use log scale for y-axis
        :param page_size: Number of items per page
        """
        classlist = tclass.binary_to_class(class_string, base)
        df = DisulfideVisualization.enumerate_class_fromlist(
            tclass, classlist, base=base
        )
        DisulfideVisualization.plot_count_vs_class_df_paginated(
            df, title=class_string, theme=theme, base=base, log=log, page_size=page_size
        )

    @staticmethod
    def display(sslist, style="sb", light="auto", panelsize=512):
        """Display the Disulfide list in the specific rendering style.

        :param sslist: List of Disulfide objects
        :param style: Rendering style: One of:
            - 'sb' - split bonds
            - 'bs' - ball and stick
            - 'cpk' - CPK style
            - 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            - 'plain' - boring single color
        :param light: If True, light background, if False, dark
        :param panelsize: Size of each panel in pixels
        """
        ssbonds = sslist
        tot_ss = len(ssbonds)  # number of ssbonds
        rows, cols = grid_dimensions(tot_ss)
        winsize = (panelsize * cols, panelsize * rows)

        set_pyvista_theme(light)

        pl = pv.Plotter(window_size=winsize, shape=(rows, cols))
        pl = DisulfideVisualization._render(pl, sslist, style, panelsize=panelsize)
        pl.enable_anti_aliasing("msaa")

        pl.link_views()
        pl.reset_camera()
        pl.show()

    @staticmethod
    def display_overlay(
        sslist,
        screenshot=False,
        movie=False,
        verbose=False,
        fname="ss_overlay.png",
        light="auto",
        winsize=WINSIZE,
    ):
        """Display all disulfides in the list overlaid in stick mode against
        a common coordinate frame.

        :param sslist: List of Disulfide objects
        :param screenshot: Save a screenshot
        :param movie: Save a movie
        :param verbose: Verbosity
        :param fname: Filename to save for the movie or screenshot
        :param light: Background color
        :param winsize: Window size tuple (width, height)
        """
        pid = sslist.id
        ssbonds = sslist
        tot_ss = len(ssbonds)
        avg_enrg = sslist.average_energy
        avg_dist = sslist.average_distance
        resolution = sslist.average_resolution

        res = 64
        if tot_ss > 30:
            res = 48
        if tot_ss > 60:
            res = 16
        if tot_ss > 90:
            res = 8

        title = f"<{pid}> {resolution:.2f} Å: ({tot_ss} SS), E: {avg_enrg:.2f} kcal/mol, Dist: {avg_dist:.2f} Å"
        fontsize = calculate_fontsize(title, winsize[0])

        set_pyvista_theme(light)

        if movie:
            pl = pv.Plotter(window_size=winsize, off_screen=True)
        else:
            pl = pv.Plotter(window_size=winsize, off_screen=False)

        pl.add_title(title=title, font_size=fontsize)
        pl.enable_anti_aliasing("msaa")
        pl.add_axes()

        mycol = get_jet_colormap(tot_ss)

        brad = BOND_RADIUS if tot_ss < 10 else BOND_RADIUS * 0.75
        brad = brad if tot_ss < 25 else brad * 0.8
        brad = brad if tot_ss < 50 else brad * 0.8
        brad = brad if tot_ss < 100 else brad * 0.6

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
            pl.show(auto_close=False)
            try:
                pl.screenshot(fname)
                if verbose:
                    print(f" -> display_overlay(): Saved image to: {fname}")
            except RuntimeError as e:
                _logger.error("Error saving screenshot: %s", e)

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

    @staticmethod
    def display_torsion_statistics(
        sslist,
        display=True,
        save=False,
        fname="ss_torsions.png",
        theme="auto",
    ):
        """Display torsion and distance statistics for a given Disulfide list.

        :param sslist: List of Disulfide objects
        :param display: Whether to display the plot in the notebook
        :param save: Whether to save the plot as an image file
        :param fname: The name of the image file to save
        :param theme: The theme to use for the plot
        """
        if len(sslist) == 0:
            _logger.warning("Empty DisulfideList. Nothing to display.")
            return

        set_plotly_theme(theme)
        title = f"{sslist.id}: {len(sslist)} members"

        tor_vals, dist_vals = sslist.calculate_torsion_statistics()

        tor_mean_vals = tor_vals.loc["mean"]
        tor_std_vals = tor_vals.loc["std"]

        dist_mean_vals = dist_vals.loc["mean"]
        dist_std_vals = dist_vals.loc["std"]

        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.125, column_widths=[1, 1]
        )

        fig.update_layout(
            title={
                "text": title,
                "xanchor": "center",
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
                name="Torsion Angle (°) ",
                error_y=dict(type="data", array=tor_std_vals, visible=True),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=["rho"],
                y=[dist_mean_vals[4]],
                name="ρ (°)",
                error_y=dict(type="data", array=[dist_std_vals[4]], visible=True),
            ),
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text="Dihedral Angle (°)", range=[-200, 200], row=1, col=1
        )
        fig.update_yaxes(range=[0, 320], row=2, col=2)

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

        fig.update_yaxes(title_text="kcal/mol", range=[0, 8], row=1, col=2)

        fig.add_trace(
            go.Bar(
                x=["Cα Distance (Å)", "Cβ Distance (Å)", "Sγ Distance (Å)"],
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
        fig.update_yaxes(title_text="Distance (A)", range=[0, 8], row=2, col=1)
        fig.update_traces(width=0.25, row=2, col=1)

        fig.add_trace(
            go.Bar(
                x=["Torsion Length (Å)"],
                y=[tor_mean_vals[5]],
                name="Torsion Length (Å)",
                error_y=dict(
                    type="data", array=[tor_std_vals[5]], width=0.25, visible=True
                ),
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="Torsion Length", range=[0, 350], row=2, col=2)
        fig.update_traces(width=0.25, row=2, col=2)

        fig.update_traces(
            error_y_thickness=2,
            error_y_color="gray",
            texttemplate="%{y:.2f} ± %{error_y.array:.2f}",
            textposition="outside",
        )

        if display:
            fig.show()

        if save:
            fig.write_image(fname)

    @staticmethod
    def plot_distances(
        distances,
        distance_type="sg",
        cutoff=-1,
        comparison="less",
        theme="auto",
        log=True,
    ):
        """Plot the distance values as a histogram.

        :param distances: List of distance values
        :param distance_type: Type of distance to plot ('sg' or 'ca')
        :param cutoff: Cutoff value for the x-axis title
        :param comparison: If 'less', show distances less than cutoff
        :param theme: The plotly theme to use
        :param log: Whether to use a logarithmic scale for the y-axis
        """
        set_plotly_theme(theme)
        yaxis_type = "log" if log else "linear"
        flip = False if comparison == "less" else True

        match distance_type:
            case "sg":
                column_name = "SG Distance"
                title = "Sγ Distance Distribution"
                if cutoff == -1.0:
                    xtitle = "Sγ-Sγ Distances, (no cutoff)"
                else:
                    xtitle = (
                        f"Sγ Distance < {cutoff} Å"
                        if not flip
                        else f"Sγ-Sγ Distance >= {cutoff} Å"
                    )
            case "ca":
                column_name = "Ca Distance"
                title = "Cα Distance Distribution"
                if cutoff == -1.0:
                    xtitle = "Cα-Cα Distances, (no cutoff)"
                else:
                    xtitle = (
                        f"Cα Distance < {cutoff} Å"
                        if not flip
                        else f"Cα-Cα Distance >= {cutoff} Å"
                    )
            case _:
                raise ValueError("Invalid distance_type. Must be 'sg' or 'ca'.")

        df = pd.DataFrame(distances, columns=[column_name])

        fig = px.histogram(
            df,
            x=column_name,
            nbins=NBINS,
            title=title,
        )
        fig.update_layout(
            title={"text": "Distance Distribution", "x": 0.5, "xanchor": "center"},
            xaxis_title=xtitle,
            yaxis_title="Frequency",
            yaxis_type=yaxis_type,
            bargap=0.2,
        )
        fig.show()

    @staticmethod
    def plot_deviation_scatterplots(df: pd.DataFrame, theme="auto") -> go.Figure:
        """Plot scatter plots for Bondlength_Deviation, Angle_Deviation, Ca_Distance, and Sg_Distance.

        :param df: DataFrame containing the deviation information
        :param theme: The theme to use for the plot
        :return: Figure containing the scatter plots
        """
        set_plotly_theme(theme)
        dotsize = 2

        fig = px.scatter(
            df, x=df.index, y="Bondlength_Deviation", title="Bondlength Deviation"
        )
        fig.update_layout(xaxis_title="Row Index", yaxis_title="Bondlength Deviation")
        fig.update_traces(marker=dict(size=dotsize))
        fig.show()

        fig = px.scatter(df, x=df.index, y="Angle_Deviation", title="Angle Deviation")
        fig.update_layout(xaxis_title="Row Index", yaxis_title="Angle Deviation")
        fig.update_traces(marker=dict(size=dotsize))
        fig.show()

        fig = px.scatter(df, x=df.index, y="Ca_Distance", title="Cα Distance")
        fig.update_layout(xaxis_title="Row Index", yaxis_title="Cα Distance")
        fig.update_traces(marker=dict(size=dotsize))
        fig.show()

        fig = px.scatter(df, x=df.index, y="Sg_Distance", title="Sg Distance")
        fig.update_layout(xaxis_title="Row Index", yaxis_title="Sg Distance")
        fig.update_traces(marker=dict(size=dotsize))
        fig.show()

        return fig

    @staticmethod
    def plot_deviation_histograms(df: pd.DataFrame, theme="auto", log=True) -> None:
        """Plot histograms for Bondlength_Deviation and Angle_Deviation with normal distribution overlay.

        :param df: DataFrame containing the deviation information
        :param theme: The plotly theme to use
        :param log: Whether to use a logarithmic scale for the y-axis
        """
        set_plotly_theme(theme)
        yaxis_type = "log" if log else "linear"

        def create_histogram_with_normal(data, column_name, title, x_label):
            fig = px.histogram(
                data,
                x=column_name,
                nbins=NBINS,
                title=title,
            )

            mean = np.mean(data[column_name])
            std = np.std(data[column_name])
            x = np.linspace(data[column_name].min(), data[column_name].max(), 100)
            y = stats.norm.pdf(x, mean, std)

            hist, _ = np.histogram(data[column_name], bins=NBINS)
            scaling_factor = np.max(hist) / np.max(y)
            y_scaled = y * scaling_factor

            fig.add_trace(
                go.Scatter(x=x, y=y_scaled, mode="lines", name="Normal Distribution")
            )

            fig.update_layout(
                title={"text": title, "x": 0.5, "xanchor": "center"},
                xaxis_title=x_label,
                yaxis_title="Frequency",
                yaxis_type=yaxis_type,
            )

            if log:
                fig.update_yaxes(range=[0, np.log10(np.max(hist))])
            else:
                fig.update_yaxes(range=[0, np.max(hist)])

            return fig

        fig_bond_length = create_histogram_with_normal(
            df,
            "Bondlength_Deviation",
            "Bond Length Deviation",
            "Bond Length Deviation (Å)",
        )
        fig_bond_length.show()

        fig_bond_angle = create_histogram_with_normal(
            df, "Angle_Deviation", "Bond Angle Deviation", "Bond Angle Deviation (°)"
        )
        fig_bond_angle.show()

    @staticmethod
    def _render(pl, sslist, style, res=100, panelsize=WINSIZE):
        """Internal rendering engine that calculates and instantiates all bond
        cylinders and atomic sphere meshes.

        :param pl: PyVista plotter object
        :param sslist: List of Disulfide objects
        :param style: Rendering style
        :param res: Resolution for rendering
        :param panelsize: Size of each panel
        :return: Updated plotter object
        """
        ssList = sslist
        tot_ss = len(ssList)
        rows, cols = grid_dimensions(tot_ss)

        if tot_ss > 30:
            res = 60
        if tot_ss > 60:
            res = 30
        if tot_ss > 90:
            res = 12

        total_plots = rows * cols
        for idx in range(min(tot_ss, total_plots)):
            r = idx // cols
            c = idx % cols
            pl.subplot(r, c)

            ss = ssList[idx]
            src = ss.pdb_id
            enrg = ss.energy
            title = f"{src} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: E: {enrg:.2f}, Cα: {ss.ca_distance:.2f} Å, Tors: {ss.torsion_length:.2f}°"
            fontsize = calculate_fontsize(title, panelsize)
            pl.add_title(title=title, font_size=fontsize)
            ss._render(
                pl,
                style=style,
                res=res,
            )

        return pl
