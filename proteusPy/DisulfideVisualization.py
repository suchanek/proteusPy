"""
This module provides visualization functionality for disulfide bonds in the proteusPy package.

Author: Eric G. Suchanek, PhD
Last revision: 2025-03-13 18:04:49
"""

# pylint: disable=C0301
# pylint: disable=C0302
# pylint: disable=C0103
# pylint: disable=W0212

import logging
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyvista as pv
from plotly.subplots import make_subplots
from scipy import stats
from tqdm import tqdm

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
from proteusPy.DisulfideClassManager import DisulfideClassManager
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import FONTSIZE, NBINS, PBAR_COLS, WINSIZE
from proteusPy.utility import (
    calculate_fontsize,
    dpi_adjusted_fontsize,
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

set_pyvista_theme("auto")

# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

_logger = create_logger(__name__)


class DisulfideVisualization:
    """Provides visualization methods for Disulfide bonds, including 3D rendering,
    statistical plots, and overlay displays."""

    @staticmethod
    def enumerate_class_fromlist(
        tclass: DisulfideClassManager, clslist, base=8
    ) -> pd.DataFrame:
        """Enumerate the classes from a list of class IDs.

        :param tclass: DisulfideClassManager instance
        :param clslist: List of class IDs to enumerate
        :param base: Base for class IDs (2 or 8)
        :return: DataFrame with class IDs and counts
        """
        x = []
        y = []

        for cls in clslist:
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
        log=False,
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
                log=log,
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
        total_classes = df.shape[0]
        total_disulfides = df["count"].sum()

        _title = f"Class: {title}, Classes: {total_classes}, SS: {total_disulfides}"
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
        total_classes = df.shape[0]

        _title = f"Class: {title}b (Total: {total_classes})"
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
        total_classes = df.shape[0]

        _title = f"Class: {title}b (Total: {total_classes})"
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

        # Calculate global min/max values from the entire dataset before pagination
        # This ensures consistent y-axis scaling across all pages
        global_y_min = df["count"].min() if len(df) > 0 else 0
        global_y_max = df["count"].max() if len(df) > 0 else 1

        # Add some padding to the max value for better visualization
        global_y_max = global_y_max * 1.1  # 10% padding
        # For log scale, ensure minimum is at least 1
        if log and global_y_min <= 0:
            global_y_min = 1

        if verbose:
            _logger.info(
                "Global y-axis range: [%s, %s]",
                math.log10(global_y_min) if log else global_y_min,
                math.log10(global_y_max) if log else global_y_max,
            )

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
                # Set consistent y-axis range for all pages using the global values
                yaxis=dict(
                    range=[
                        math.log10(global_y_min) if log else global_y_min,
                        math.log10(global_y_max) if log else global_y_max,
                    ]
                ),
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
        class_string: str = None,
        base: int = 8,
        theme: str = "auto",
        log: bool = False,
        page_size: int = 200,
        paginated: bool = False,
    ):
        """Plot the distribution of classes for the given binary class string.

        :param tclass: DisulfideClassManager instance
        :param class_string: The binary class string to plot
        :param base: Base for class IDs (2 or 8)
        :param theme: Theme to use for the plot
        :param log: Whether to use log scale for y-axis
        :param page_size: Number of items per page
        :param paginated: Whether to paginate the output
        """
        classlist = tclass.binary_to_class(class_string, base)
        df = DisulfideVisualization.enumerate_class_fromlist(
            tclass, classlist, base=base
        )
        if paginated:
            DisulfideVisualization.plot_count_vs_class_df_paginated(
                df,
                title=class_string,
                theme=theme,
                base=base,
                log=log,
                page_size=page_size,
            )
        else:
            DisulfideVisualization.plot_count_vs_class_df(
                df, title=class_string, theme=theme, base=base, log=log
            )

    @staticmethod
    def display_sslist(sslist, style="sb", light="auto", panelsize=512):
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
        pl = DisulfideVisualization._render_sslist(
            pl, sslist, style, panelsize=panelsize
        )
        pl.enable_anti_aliasing("msaa")

        pl.link_views()
        pl.reset_camera()
        pl.show()

    @staticmethod
    def simulate_orbit_on_path(plotter, steps=360, step_size=None, sleep_time=0.02):
        """
        Simulate the orbit_on_path function by manually updating the camera position
        over a number of steps.

        :param plotter: The PyVista plotter object
        :type plotter: pv.Plotter
        :param steps: Number of steps for one complete rotation, defaults to 360
        :type steps: int, optional
        :param step_size: Step size for the animation, defaults to 1.0/steps if None
        :type step_size: float, optional
        :param sleep_time: Time to sleep between steps in seconds, defaults to 0.02
        :type sleep_time: float, optional
        """

        # Get the current camera focal point and position
        focal_point = np.array(plotter.camera.focal_point)
        current_position = np.array(plotter.camera.position)

        # Compute radius as the distance from camera to focal point
        radius = np.linalg.norm(current_position - focal_point)

        # Calculate the initial angle in the XY plane
        initial_angle = np.arctan2(
            current_position[1] - focal_point[1], current_position[0] - focal_point[0]
        )

        # Calculate step size if not provided
        if step_size is None:
            step_size = 1.0 / steps

        # Calculate the total number of steps based on step_size
        total_steps = int(1.0 / step_size) if step_size > 0 else steps

        # Start the orbit animation
        for step in range(total_steps):
            # Compute the angle for this step in radians
            angle = initial_angle + 2 * np.pi * step / total_steps

            # Create a new position orbiting in the XY plane (keeping Z constant)
            new_position = focal_point + np.array(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    current_position[2],  # maintain the original Z level
                ]
            )

            # Update the camera: new position, focal point, and an up vector
            plotter.camera_position = (
                new_position.tolist(),
                focal_point.tolist(),
                [0, 0, 1],
            )
            plotter.render()

            # Pause briefly to control animation speed
            time.sleep(sleep_time)

        return

    @staticmethod
    def display_overlay(
        sslist=None,
        pl=None,
        screenshot=False,
        movie=False,
        verbose=False,
        fname="ss_overlay.png",
        light="auto",
        winsize=WINSIZE,
        spin=False,
        steps=360,
        step_size=None,
        dpi=300,
    ) -> pv.Plotter:
        """Display all disulfides in the list overlaid in stick mode against
        a common coordinate frame.

        :param sslist: List of Disulfide objects
        :type sslist: DisulfideList
        :param screenshot: Save a screenshot
        :type screenshot: bool
        :param movie: Save a movie
        :type movie: bool
        :param verbose: Verbosity
        :type verbose: bool
        :param fname: Filename to save for the movie or screenshot
        :type fname: str
        :param light: Background color
        :type light: str
        :param winsize: Window size tuple (width, height)
        :type winsize: tuple
        :param spin: Whether to spin the plot
        :type spin: bool
        :param steps: Number of steps for spinning
        :type steps: int
        :param step_size: Step size for spinning
        :type step_size: float
        :param dpi: DPI for the saved image
        :type dpi: int
        :return: pyvista.Plotter instance
        :rtype: pv.Plotter
        """
        pid = sslist.pdb_id
        ssbonds = sslist
        tot_ss = len(ssbonds)
        avg_enrg = sslist.average_energy
        avg_dist = sslist.average_distance
        resolution = sslist.average_resolution
        scale = dpi / 300

        res = 32
        if tot_ss > 10:
            res = 24
        if tot_ss > 15:
            res = 18
        if tot_ss > 20:
            res = 12

        title = f"<{pid}> {resolution:.2f} Å: ({tot_ss} SS), E: {avg_enrg:.2f} kcal/mol, Dist: {avg_dist:.2f} Å"
        fontsize = calculate_fontsize(title, winsize[0])
        fontsize = dpi_adjusted_fontsize(fontsize)

        set_pyvista_theme(light)

        if movie:
            if not pl:
                pl = pv.Plotter(window_size=winsize, off_screen=True)
        else:
            if not pl:
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
            DisulfideVisualization._render_ss(
                ss,
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
                pl.screenshot(fname, scale=scale)
                if verbose:
                    print(f" -> display_overlay(): Saved image to: {fname}")
            except RuntimeError as e:
                _logger.error("Error saving screenshot: %s", e)

        elif movie:
            if verbose:
                print(f" -> display_overlay(): Saving mp4 animation to: {fname}")
            _logger.debug("Saving mp4 animation to: %s", fname)

            pl.open_movie(fname)
            path = pl.generate_orbital_path(n_points=360)
            pl.orbit_on_path(path, write_frames=True)
            pl.close()

            if verbose:
                print(f" -> display_overlay(): Saved mp4 animation to: {fname}")
        elif spin:
            _logger.debug("Spinning the plot.")
            # Reset camera before generating the path
            # pl.reset_camera()

            # Generate the orbital path
            # path = pl.generate_orbital_path(n_points=steps)

            # Calculate step size if not provided
            if step_size is None:
                step_size = 1.0 / steps

            if verbose:
                print(
                    " -> display_overlay(): Spinning the plot. Window will remain open after spinning."
                )

            # First show the plot to initialize the window
            pl.show(auto_close=False)

            # Then apply the orbit path - this will spin the view and keep the window open
            try:
                # Use our custom simulation function instead of orbit_on_path
                DisulfideVisualization.simulate_orbit_on_path(
                    pl, steps=steps, step_size=step_size
                )
            except ValueError as e:
                _logger.error("Error during simulate_orbit_on_path: %s", e)

        else:
            pl.show()
        return pl

    @staticmethod
    def display_torsion_statistics(
        sslist,
        display=True,
        save=False,
        fname="ss_torsions.png",
        theme="auto",
        dpi=300,
        figure_size=(4, 3),
    ):
        """Display torsion and distance statistics for a given Disulfide list.

        :param sslist: List of Disulfide objects
        :param display: Whether to display the plot in the notebook
        :param save: Whether to save the plot as an image file
        :param fname: The name of the image file to save
        :param theme: The theme to use for the plot
        :param dpi: DPI (dots per inch) for the saved image, controls the resolution (default: 300)
        :param figure_size: Tuple of (width, height) in inches for the figure size (default: (4, 3))
        """
        if len(sslist) == 0:
            _logger.warning("Empty DisulfideList. Nothing to display.")
            return

        # Calculate pixel dimensions
        _width = figure_size[0] * dpi
        _height = figure_size[1] * dpi

        # Calculate scale factor based on DPI (300 DPI is the reference)
        scale_factor = dpi / 300

        # Scale font sizes based on DPI
        title_font_size = int(20 * scale_factor)
        axis_font_size = int(14 * scale_factor)
        tick_font_size = int(12 * scale_factor)
        text_font_size = int(10 * scale_factor)
        legend_font_size = int(10 * scale_factor)

        set_plotly_theme(theme)
        title = f"{sslist.pdb_id}: {len(sslist)} members"

        tor_vals, dist_vals = sslist.calculate_torsion_statistics()

        tor_mean_vals = tor_vals.loc["mean"]
        tor_std_vals = tor_vals.loc["std"]

        dist_mean_vals = dist_vals.loc["mean"]
        dist_std_vals = dist_vals.loc["std"]

        # Adjust vertical spacing based on scale factor
        vertical_spacing = 0.125 * (1 + 0.2 * (scale_factor - 1))

        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=vertical_spacing, column_widths=[1, 1]
        )

        fig.update_layout(
            title={
                "text": title,
                "xanchor": "center",
                "x": 0.5,
                "yanchor": "top",
                "font": {"size": title_font_size},
            },
            width=_width,
            height=_height,
            margin=dict(t=50 * scale_factor, b=50 * scale_factor),
            legend=dict(
                font=dict(size=legend_font_size),
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="right",
                x=1,
                tracegroupgap=5,
                itemsizing="constant",
            ),
        )

        fig.add_trace(
            go.Bar(
                x=["X1", "X2", "X3", "X4", "X5"],
                y=tor_mean_vals[:5],
                name="Torsion Angle (°) ",
                error_y=dict(
                    type="data",
                    array=tor_std_vals,
                    width=4 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[
                    f"{val:.2f} ± {std:.2f}"
                    for val, std in zip(tor_mean_vals[:5], tor_std_vals[:5])
                ],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="torsion",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=["rho"],
                y=[dist_mean_vals[4] * 100],
                name="ρ (°) * 100",
                error_y=dict(
                    type="data",
                    array=[dist_std_vals[4]],
                    width=4.0 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[f"{dist_mean_vals[4] * 100:.2f} ± {dist_std_vals[4]:.2f}"],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="rho",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text="Dihedral Angle (°)",
            range=[-200, 200],
            row=1,
            col=1,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )
        fig.update_yaxes(
            range=[0, 320],
            row=2,
            col=2,
            tickfont=dict(size=tick_font_size),
        )

        fig.add_trace(
            go.Bar(
                x=["Strain Energy (kcal/mol)"],
                y=[dist_mean_vals[3]],
                name="Energy (kcal/mol)",
                error_y=dict(
                    type="data",
                    array=[dist_std_vals[3].tolist()],
                    width=4.0 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[f"{dist_mean_vals[3]:.2f} ± {dist_std_vals[3]:.2f}"],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="energy",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.update_traces(width=0.5 * scale_factor, row=1, col=2)

        fig.update_yaxes(
            title_text="kcal/mol",
            range=[0, 8],
            row=1,
            col=2,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )

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
                    width=4 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[
                    f"{dist_mean_vals[0]:.2f} ± {dist_std_vals[0]:.2f}",
                    f"{dist_mean_vals[1]:.2f} ± {dist_std_vals[1]:.2f}",
                    f"{dist_mean_vals[2]:.2f} ± {dist_std_vals[2]:.2f}",
                ],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="distances",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Distance (Å)",
            range=[0, 8],
            row=2,
            col=1,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )
        fig.update_traces(width=0.5 * scale_factor, row=2, col=1)

        fig.add_trace(
            go.Bar(
                x=["Torsion Length (Å)"],
                y=[tor_mean_vals[5]],
                name="Torsion Length (Å)",
                error_y=dict(
                    type="data",
                    array=[tor_std_vals[5]],
                    width=4.0 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[f"{tor_mean_vals[5]:.2f} ± {tor_std_vals[5]:.2f}"],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="torsion_length",
                showlegend=True,
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(
            title_text="Torsion Length",
            range=[0, 350],
            row=2,
            col=2,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )
        fig.update_traces(width=0.5 * scale_factor, row=2, col=2)

        # Update x-axis fonts
        fig.update_xaxes(tickfont=dict(size=tick_font_size))

        if display:
            fig.show()

        if save:
            # Convert DPI to scale factor (300 DPI is considered standard, so scale = dpi/300)
            scale = dpi / 300
            fig.write_image(fname, scale=scale)

    @staticmethod
    def plot_energies(
        energies,
        theme="auto",
        log=True,
    ):
        """Plot the energies as a histogram.

        :param energies: List of energies
        :param theme: The plotly theme to use
        :param log: Whether to use a logarithmic scale for the y-axis
        """
        set_plotly_theme(theme)
        yaxis_type = "log" if log else "linear"

        column_name = "Energy"
        title = "Disulfide Torsional Energy Distribution"
        xtitle = "Energy (kcal/mol)"

        df = pd.DataFrame(energies, columns=[column_name])

        fig = px.histogram(
            df,
            x=column_name,
            nbins=NBINS,
            title=title,
        )
        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            xaxis_title=xtitle,
            yaxis_title="Frequency",
            yaxis_type=yaxis_type,
            bargap=0.2,
        )
        fig.show()

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
    def _render_sslist(pl, sslist, style, res=64, panelsize=512):
        """Internal rendering engine that calculates and instantiates all bond
        cylinders and atomic sphere meshes.

        :param pl: PyVista plotter object
        :param sslist: List of Disulfide objects
        :param style: Rendering style
        :param res: Resolution for rendering
        :param panelsize: Size of each panel
        :return: Updated plotter object
        """
        _ssList = sslist
        tot_ss = len(_ssList)
        rows, cols = grid_dimensions(tot_ss)
        _res = res
        if tot_ss > 10:
            _res = 18
        elif tot_ss > 15:
            _res = 12
        elif tot_ss > 20:
            _res = 8
        else:
            _res = 32

        total_plots = rows * cols
        for idx in range(min(tot_ss, total_plots)):
            r = idx // cols
            c = idx % cols
            pl.subplot(r, c)

            ss = _ssList[idx]
            src = ss.pdb_id
            enrg = ss.energy
            title = f"{src} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: E: {enrg:.2f}, Ca: {ss.ca_distance:.2f} Å, Tors: {ss.torsion_length:.2f}°"
            fontsize = calculate_fontsize(title, panelsize)
            fontsize = dpi_adjusted_fontsize(fontsize) - 2
            pl.add_title(title=title, font_size=fontsize)
            DisulfideVisualization._render_ss(
                ss,
                pl,
                style=style,
                res=_res,
                translate=True,
            )
            pl.reset_camera()

        pl.link_views()
        pl.reset_camera()
        return pl

    @staticmethod
    def _render_ss(
        ss,
        pvplot: pv.Plotter,
        style="bs",
        bondcolor=BOND_COLOR,
        bs_scale=BS_SCALE,
        spec=SPECULARITY,
        specpow=SPEC_POWER,
        translate=False,
        bond_radius=BOND_RADIUS,
        res=64,
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
                    pv.Sphere(
                        center=coords[i],
                        radius=rad,
                        theta_resolution=res,
                        phi_resolution=res,
                    ),
                    color=colors[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )

        def draw_bonds(
            ss,
            pvp,
            coords,
            bond_radius=BOND_RADIUS,
            style="sb",
            bcolor=BOND_COLOR,
            all_atoms=True,
            res=res,
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

            missing = ss.missing_atoms
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

                cap1 = pv.Sphere(
                    center=prox_pos,
                    radius=bradius,
                    theta_resolution=res,
                    phi_resolution=res,
                )
                cap2 = pv.Sphere(
                    center=distal_pos,
                    radius=bradius,
                    theta_resolution=res,
                    phi_resolution=res,
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

                pvp.add_mesh(cap1, color=orig_col)
                pvp.add_mesh(cap2, color=dest_col)

            return pvp  # end draw_bonds

        model = ss.modelled
        coords = ss.internal_coords
        if translate:
            coords -= ss.cofmass

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
            pvp = draw_bonds(
                ss,
                pvp,
                coords,
                style="bs",
                all_atoms=all_atoms,
                bond_radius=bond_radius,
            )
        elif style in ["sb", "pd", "plain"]:
            pvp = draw_bonds(
                ss,
                pvp,
                coords,
                style=style,
                all_atoms=all_atoms,
                bond_radius=bond_radius,
                bcolor=bondcolor if style == "plain" else None,
            )

        return pvp

    @staticmethod
    def display_ss(
        ss, single=True, style="sb", light="auto", shadows=False, winsize=WINSIZE
    ) -> None:
        """
        Display the Disulfide bond in the specific rendering style.
        :param ss: Disulfide object
        :param single: Display the bond in a single panel in the specific style.
        :param style:  Rendering style: One of:
            * 'sb' - split bonds
            * 'bs' - ball and stick
            * 'cpk' - CPK style
            * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            * 'plain' - boring single color
        :param light: If True, light background, if False, dark

        Example:
        >>> import proteusPy as pp

        >>> PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)
        >>> ss = PDB_SS[0]
        >>> ss.display(style='cpk', light="auto")
        >>> ss.screenshot(style='bs', fname='proteus_logo_sb.png')
        """

        src = ss.pdb_id
        enrg = ss.energy

        title = f"{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol. Ca: {ss.ca_distance:.2f} Å, Tors: {ss.torsion_length:.2f}°"

        set_pyvista_theme(light)
        fontsize = dpi_adjusted_fontsize(FONTSIZE)

        if single:
            _pl = pv.Plotter(window_size=winsize)
            _pl.add_title(title=title, font_size=fontsize)
            _pl.enable_anti_aliasing("msaa")

            DisulfideVisualization._render_ss(
                ss,
                _pl,
                style=style,
                translate=True,
            )
            _pl.reset_camera()
            if shadows:
                _pl.enable_shadows()
            _pl.show()

        else:
            pl = pv.Plotter(window_size=winsize, shape=(2, 2))
            pl.subplot(0, 0)

            pl.add_title(title=title, font_size=fontsize)
            pl.enable_anti_aliasing("msaa")

            # pl.add_camera_orientation_widget()

            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="cpk",
                translate=True,
            )

            pl.subplot(0, 1)
            pl.add_title(title=title, font_size=fontsize)

            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="bs",
                translate=True,
            )

            pl.subplot(1, 0)
            pl.add_title(title=title, font_size=fontsize)

            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="sb",
                translate=True,
            )

            pl.subplot(1, 1)
            pl.add_title(title=title, font_size=fontsize)

            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="pd",
                translate=True,
            )

            pl.link_views()
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()
            pl.show()
        return

    @staticmethod
    def make_movie(
        ss, style="sb", fname="ssbond.mp4", verbose=False, steps=360
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

        # src = self.pdb_id
        # name = self.name
        # enrg = self.energy
        # title = f"{src} {name}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol, Cα: {self.ca_distance:.2f} Å, Tors: {self.torsion_length:.2f}"

        if verbose:
            print(f"Rendering animation to {fname}...")
        set_pyvista_theme("auto")

        pl = pv.Plotter(window_size=WINSIZE, off_screen=True)
        pl.open_movie(fname)
        path = pl.generate_orbital_path(n_points=steps)

        #
        # pl.add_title(title=title, font_size=FONTSIZE)
        pl.enable_anti_aliasing("msaa")
        pl = DisulfideVisualization._render_ss(
            ss,
            pl,
            style=style,
            translate=True,
        )
        pl.reset_camera()
        pl.orbit_on_path(path, write_frames=True)
        pl.close()

        if verbose:
            print(f"Saved mp4 animation to: {fname}")

    @staticmethod
    def spin(
        ss, style="sb", pl=None, verbose=False, steps=360, theme="auto"
    ) -> pv.Plotter:
        """
        Spin the object by rotating it one revolution about the Y axis in the given style.

        :param style: Rendering style, defaults to 'sb', one of:
            * 'sb' - split bonds
            * 'bs' - ball and stick
            * 'cpk' - CPK style
            * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            * 'plain' - boring single color

        :param verbose: Verbosity, defaults to False
        :param steps: Number of steps for one complete rotation, defaults to 360.
        """

        src = ss.pdb_id
        enrg = ss.energy

        title = f"{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol, Ca: {ss.ca_distance:.2f} Å, Tors: {ss.torsion_length:.2f}"

        set_pyvista_theme(theme)

        if verbose:
            _logger.info("Spinning object: %d steps...", steps)

        # Create a Plotter instance
        if not pl:
            pl = pv.Plotter(window_size=WINSIZE, off_screen=False)

        pl.add_title(title=title, font_size=dpi_adjusted_fontsize(FONTSIZE))

        # Enable anti-aliasing for smoother rendering
        pl.enable_anti_aliasing("msaa")

        # Generate an orbital path for spinning
        # path = pl.generate_orbital_path(n_points=steps)

        # Render the object in the specified style
        pl = DisulfideVisualization._render_ss(ss, pl, style=style, translate=True)

        pl.reset_camera()
        pl.show(auto_close=False)

        step_size = 1 / steps

        try:
            # Use our custom simulation function instead of orbit_on_path
            DisulfideVisualization.simulate_orbit_on_path(
                pl, steps=steps, step_size=step_size
            )
        except ValueError as e:
            _logger.error("Error during simulate_orbit_on_path: %s", e)

        # Orbit the camera along the generated path

        if verbose:
            print("Spinning completed.")

        return pl

    @staticmethod
    def screenshot(
        ss,
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

        src = ss.pdb_id
        enrg = ss.energy
        title = f"{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol, Ca: {ss.ca_distance:.2f} Å, Sg: {ss.sg_distance:.2f} Å, Tors: {ss.torsion_length:.2f}"

        set_pyvista_theme(light)

        if verbose:
            _logger.info("Rendering screenshot to file %s", fname)

        if single:
            pl = pv.Plotter(window_size=WINSIZE, off_screen=True)
            pl.add_title(title=title, font_size=dpi_adjusted_fontsize(FONTSIZE))
            pl.enable_anti_aliasing("msaa")
            DisulfideVisualization._render_ss(
                ss,
                pl,
                style=style,
                translate=True,
            )
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()

            # pl.show(auto_close=True)  # allows for manipulation
            # Take the screenshot after ensuring the plotter is still active
            try:
                pl.screenshot(fname)
                pl.close()
            except RuntimeError as e:
                _logger.error("Error saving screenshot: %s", e)

        else:
            pl = pv.Plotter(window_size=WINSIZE, shape=(2, 2), off_screen=True)
            pl.subplot(0, 0)

            pl.add_title(title=title, font_size=dpi_adjusted_fontsize(FONTSIZE))
            pl.enable_anti_aliasing("msaa")

            # pl.add_camera_orientation_widget()
            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="cpk",
                translate=True,
            )

            pl.subplot(0, 1)
            pl.add_title(title=title, font_size=dpi_adjusted_fontsize(FONTSIZE))
            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="pd",
                translate=True,
            )

            pl.subplot(1, 0)
            pl.add_title(title=title, font_size=dpi_adjusted_fontsize(FONTSIZE))
            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="bs",
                translate=True,
            )

            pl.subplot(1, 1)
            pl.add_title(title=title, font_size=dpi_adjusted_fontsize(FONTSIZE))
            DisulfideVisualization._render_ss(
                ss,
                pl,
                style="sb",
                translate=True,
            )

            pl.link_views()
            pl.reset_camera()
            if shadows:
                pl.enable_shadows()

            # Take the screenshot after ensuring the plotter is still active
            # pl.show(auto_close=True)  # allows for manipulation

            try:
                pl.screenshot(fname)
                pl.close()
            except RuntimeError as e:
                _logger.error("Error saving screenshot: %s", e)

        if verbose:
            print(f"Screenshot saved as: {fname}")

    @staticmethod
    def display_worst_structures(df, top_n=10, sample_percent=10):
        """
        Highlight the worst structures for distance and angle deviations and annotate their names.
        Also, add a subplot showing the worst structures aggregated by PDB_ID.

        :param top_n: Number of worst structures to highlight.
        :type top_n: int
        """
        rows = df.shape[0]
        samplesize = int(rows * sample_percent / 100)

        # Identify the worst structures for Bond Length Deviation
        worst_distance = df.nlargest(top_n, "Bondlength_Deviation")

        # Identify the worst structures for angle deviation
        worst_angle = df.nlargest(top_n, "Angle_Deviation")

        # Identify the worst structures for Cα distance
        worst_ca = df.nlargest(top_n, "Ca_Distance")

        # Combine the worst structures
        worst_structures = pd.concat(
            [worst_distance, worst_angle, worst_ca]
        ).drop_duplicates()

        # Aggregate worst structures by PDB_ID
        worst_structures_agg = (
            worst_structures.groupby("PDB_ID").size().reset_index(name="Count")
        )

        # Scatter plot for all structures
        fig = px.scatter(
            df.sample(samplesize),
            x="Bondlength_Deviation",
            y="Angle_Deviation",
            title="Bond Length Deviation vs. Angle Deviation",
            hover_data=["PDB_ID", "Bondlength_Deviation", "Angle_Deviation"],
        )

        fig.update_traces(
            hovertemplate="<b>PDB ID: %{customdata[0]}</b><br>Bondlength Deviation: %{customdata[1]:.2f}<br>Angle Deviation: %{customdata[2]:.2f}<extra></extra>"
        )

        fig.add_scatter(
            x=worst_structures["Bondlength_Deviation"],
            y=worst_structures["Angle_Deviation"],
            mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            customdata=worst_structures[
                ["PDB_ID", "Bondlength_Deviation", "Angle_Deviation"]
            ],
            hovertemplate="<b>PDB ID: %{customdata[0]}</b><br>Bondlength Deviation: %{customdata[1]:.2f}<br>Angle Deviation: %{customdata[2]:.2f}<extra></extra>",
            name="Worst Structures",
        )
        for _, row in worst_structures.iterrows():
            fig.add_annotation(
                x=row["Bondlength_Deviation"],
                y=row["Angle_Deviation"],
                text=row["SS_Name"],
                showarrow=False,
                arrowhead=1,
                font=dict(size=6),  # Adjust the font size as needed
                xshift=0,
                yshift=10,
            )
        fig.show()

        # Bar plot for worst structures aggregated by PDB_ID
        fig = px.bar(
            worst_structures_agg,
            x="PDB_ID",
            y="Count",
            title="Worst Structures Aggregated by PDB_ID",
        )
        fig.update_layout(xaxis_title="PDB_ID", yaxis_title="Count")
        fig.show()

    @staticmethod
    def plot_percentile_cutoffs(
        pdb_full,
        percentile_range=(80, 99),
        num_steps=20,
        figsize=(12, 10),  # Increased height for two subplots
        save_path=None,
        verbose=False,
    ):
        """
        Generate a graph showing Ca and Sg distance cutoffs as well as bondlength and bondangle deviation cutoffs as a function of percentile.

        :param pdb_full: The PDB object containing the SSList attribute
        :type pdb_full: object
        :param percentile_range: Tuple containing the min and max percentiles to plot (inclusive)
        :type percentile_range: tuple
        :param num_steps: Number of percentile steps to calculate
        :type num_steps: int
        :param figsize: Figure size as (width, height) in inches
        :type figsize: tuple
        :param save_path: Path to save the figure, if None the figure is displayed but not saved
        :type save_path: str or None
        :param verbose: Whether to print the results, defaults to False
        :type verbose: bool
        :return: The figure and axes objects
        :rtype: tuple
        """
        # pylint: disable=C0415

        from proteusPy import DisulfideStats

        # Create deviation dataframe
        dev_df = DisulfideStats.create_deviation_dataframe(
            pdb_full.SSList, verbose=verbose
        )

        # Generate percentile values
        percentiles = np.linspace(percentile_range[0], percentile_range[1], num_steps)

        # Initialize arrays to store cutoff values
        ca_cutoffs = []
        sg_cutoffs = []
        bondlength_cutoffs = []
        angle_cutoffs = []

        # Calculate cutoffs for each percentile
        for p in percentiles:
            ca_cutoff = DisulfideStats.calculate_percentile_cutoff(
                dev_df, "Ca_Distance", percentile=p
            )
            sg_cutoff = DisulfideStats.calculate_percentile_cutoff(
                dev_df, "Sg_Distance", percentile=p
            )
            bondlength_cutoff = DisulfideStats.calculate_percentile_cutoff(
                dev_df, "Bondlength_Deviation", percentile=p
            )
            angle_cutoff = DisulfideStats.calculate_percentile_cutoff(
                dev_df, "Angle_Deviation", percentile=p
            )

            ca_cutoffs.append(ca_cutoff)
            sg_cutoffs.append(sg_cutoff)
            bondlength_cutoffs.append(bondlength_cutoff)
            angle_cutoffs.append(angle_cutoff)

        # Create the plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot distances in the first subplot
        ax1.plot(percentiles, ca_cutoffs, "b-", marker="o", label="Cα-Cα Distance")
        ax1.plot(percentiles, sg_cutoffs, "r-", marker="s", label="Sγ-Sγ Distance")

        # Add labels and title for first subplot
        ax1.set_ylabel("Distance Cutoff (Å)")
        ax1.set_title("Disulfide Bond Distance Cutoffs vs Percentile")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend()

        # Annotate key percentiles for distances
        for i, p in enumerate(percentiles):
            if i % (num_steps // 4) == 0 or p == percentiles[-1]:
                ax1.annotate(
                    f"{p:.0f}%: {ca_cutoffs[i]:.2f}Å",
                    xy=(p, ca_cutoffs[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
                ax1.annotate(
                    f"{p:.0f}%: {sg_cutoffs[i]:.2f}Å",
                    xy=(p, sg_cutoffs[i]),
                    xytext=(5, -15),
                    textcoords="offset points",
                )

        # Plot deviations in the second subplot
        ax2.plot(
            percentiles,
            bondlength_cutoffs,
            "g-",
            marker="^",
            label="Bondlength Deviation",
        )
        ax2.plot(percentiles, angle_cutoffs, "m-", marker="d", label="Angle Deviation")

        # Add labels for second subplot
        ax2.set_xlabel("Percentile")
        ax2.set_ylabel("Deviation Cutoff")
        ax2.set_title("Disulfide Bond Deviation Cutoffs vs Percentile")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend()

        # Annotate key percentiles for deviations
        for i, p in enumerate(percentiles):
            if i % (num_steps // 4) == 0 or p == percentiles[-1]:
                ax2.annotate(
                    f"{p:.0f}%: {bondlength_cutoffs[i]:.2f}",
                    xy=(p, bondlength_cutoffs[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
                ax2.annotate(
                    f"{p:.0f}%: {angle_cutoffs[i]:.2f}°",
                    xy=(p, angle_cutoffs[i]),
                    xytext=(5, -15),
                    textcoords="offset points",
                )

        # Add a main title for the entire figure
        fig.suptitle("Disulfide Bond Metrics vs Percentile", fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Adjust to make room for the suptitle

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig, (ax1, ax2)

    @staticmethod
    def plot_energy_by_class(
        metrics_df: pd.DataFrame,
        title: str = "Energy Distribution by Class",
        theme: str = "auto",
        save: bool = False,
        savedir: str = ".",
        verbose: bool = False,
        split: bool = False,
        max_classes_per_plot: int = 32,
        dpi: int = 300,
        suffix: str = "png",
    ) -> None:
        """
        Create a box plot showing energy distribution by class_id using plotly_express.

        :param metrics_df: DataFrame containing the metrics data with class_id and energy columns
        :param title: Title for the plot
        :param theme: Theme to use for the plot ('auto', 'light', or 'dark')
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param verbose: Whether to display verbose output
        :param split: Whether to split the plot into multiple plots if there are many classes
        :param max_classes_per_plot: Maximum number of classes to include in each plot when splitting
        :param dpi: DPI (dots per inch) for the saved image, controls the resolution (default: 300)
        """
        # Set the plotly theme
        set_plotly_theme(theme)

        # If not splitting or few classes, create a single plot
        if not split or len(metrics_df["class"].unique()) <= max_classes_per_plot:
            # Create box plot using plotly_express
            fig = px.box(
                metrics_df,
                x="class",  # x-axis: class IDs
                y="energy",  # y-axis: energy values
                title=title,
                labels={"class": "Class ID", "energy": "Energy (kcal/mol)"},
            )

            # Update layout for consistent styling
            fig.update_layout(
                showlegend=True,
                title_x=0.5,
                title_font=dict(size=20),
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                autosize=True,
            )

            # Save or display the plot
            if save:
                fname = Path(savedir) / f"{title.lower().replace(' ', '_')}.png"
                if verbose:
                    _logger.info("Saving plot to %s with DPI %d", fname, dpi)
                # Convert DPI to scale factor (300 DPI is considered standard, so scale = dpi/300)
                scale = dpi / 300
                fig.write_image(fname, suffix, scale=scale)
            else:
                fig.show()
        else:
            # Split into multiple plots
            unique_classes = metrics_df["class"].unique()
            num_plots = (
                len(unique_classes) + max_classes_per_plot - 1
            ) // max_classes_per_plot

            if verbose:
                _logger.info(
                    "Splitting into %d plots with up to %d classes each",
                    num_plots,
                    max_classes_per_plot,
                )

            for i in range(num_plots):
                start_idx = i * max_classes_per_plot
                end_idx = min((i + 1) * max_classes_per_plot, len(unique_classes))
                classes_subset = unique_classes[start_idx:end_idx]

                # Filter DataFrame for current subset of classes
                subset_df = metrics_df[metrics_df["class"].isin(classes_subset)]

                # Create plot title for this subset
                subset_title = f"{title} (Part {i+1}/{num_plots})"

                # Create box plot for this subset
                fig = px.box(
                    subset_df,
                    x="class",
                    y="energy",
                    title=subset_title,
                    labels={"class": "Class ID", "energy": "Energy (kcal/mol)"},
                )

                # Update layout for consistent styling
                fig.update_layout(
                    showlegend=True,
                    title_x=0.5,
                    title_font=dict(size=20),
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    autosize=True,
                )

                # Save or display the plot
                if save:
                    fname = (
                        Path(savedir)
                        / f"{title.lower().replace(' ', '_')}_part_{i+1}.{suffix}"
                    )
                    if verbose:
                        _logger.info(
                            "Saving plot part %d to %s with DPI %d", i + 1, fname, dpi
                        )
                    # Convert DPI to scale factor (300 DPI is considered standard, so scale = dpi/300)
                    scale = dpi / 300
                    fig.write_image(fname, suffix, scale=scale)
                else:
                    fig.show()

    @staticmethod
    def plot_torsion_distance_by_class(
        metrics_df: pd.DataFrame,
        title: str = "Torsion Distance by Class",
        theme: str = "auto",
        save: bool = False,
        savedir: str = ".",
        verbose: bool = False,
        dpi: int = 300,
        suffix: str = "png",
        fname_prefix: str = "",
        window_size: tuple = (1024, 1024),
    ) -> None:
        """
        Create a bar chart showing torsion distance by class_id using plotly_express.

        :param metrics_df: DataFrame containing the metrics data with class_id and avg_torsion_distance columns
        :param title: Title for the plot
        :param theme: Theme to use for the plot ('auto', 'light', or 'dark')
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param verbose: Whether to display verbose output
        :param dpi: DPI (dots per inch) for the saved image, controls the resolution (default: 300)
        :param suffix: File format for saved images (default: "png")
        """
        # Set the plotly theme
        set_plotly_theme(theme)

        # Check if avg_torsion_distance column exists
        if "avg_torsion_distance" not in metrics_df.columns:
            _logger.warning("avg_torsion_distance column not found in the DataFrame.")
            return

        # Calculate threshold for high occupancy classes (mean + 1 std dev)
        count_threshold = metrics_df["count"].mean() + metrics_df["count"].std()

        # Create color array based on threshold
        colors = [
            (
                "rgba(65, 105, 225, 0.7)"
                if x < count_threshold
                else "rgba(220, 20, 60, 0.7)"
            )
            for x in metrics_df["count"]
        ]

        # Create bar chart using plotly_express
        fig = px.bar(
            metrics_df,
            x="class",  # x-axis: class IDs
            y="avg_torsion_distance",  # y-axis: torsion distance values
            title=title,
            labels={
                "class": "Class ID",
                "avg_torsion_distance": "Average Torsion Distance (°)",
                "count": "Count",
            },
            hover_data=["count"],  # Add count to hover information
        )

        # Update marker colors based on threshold
        fig.update_traces(marker_color=colors)

        # Update layout for consistent styling
        fig.update_layout(
            showlegend=False,  # No legend needed for a single bar series
            title_x=0.5,
            title_font=dict(size=20),
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            autosize=True,
            width=window_size[0],
            height=window_size[1],
            xaxis=dict(
                tickangle=-45,  # Angle the x-axis labels for better readability
                tickmode="auto",
            ),
        )

        # Save or display the plot
        if save:
            fname = Path(savedir) / f"{fname_prefix}.{suffix}"

            if verbose:
                _logger.info("Saving plot to %s with DPI %d", fname, dpi)
            # Convert DPI to scale factor (300 DPI is considered standard, so scale = dpi/300)
            scale = dpi / 300
            fig.write_image(fname, suffix, scale=scale)
        else:
            fig.show()

    @staticmethod
    def display_torsion_class_df(
        torsion_df: pd.DataFrame,
        class_id: str,
        display: bool = True,
        save: bool = False,
        fname: str = "ss_torsions.png",
        theme: str = "auto",
        dpi: int = 300,
        figure_size: tuple = (4, 3),
    ) -> None:
        """
        Display torsion and distance statistics for a given class ID using a TorsionDF dataframe.

        :param torsion_df: The TorsionDF dataframe containing the torsion data
        :param class_id: The class ID to display statistics for (e.g. '11111b' for binary or '11111o' for octant)
        :param display: Whether to display the plot in the notebook
        :param save: Whether to save the plot as an image file
        :param fname: The name of the image file to save
        :param theme: The theme to use for the plot ('auto', 'light', or 'dark')
        :param dpi: DPI (dots per inch) for the saved image, controls the resolution
        :param figure_size: Tuple of (width, height) in inches for the figure size
        """
        # Determine if binary or octant class based on suffix
        if class_id.endswith("b"):
            class_column = "binary_class_string"
            class_str = class_id[:-1]  # Remove 'b' suffix
        elif class_id.endswith("o"):
            class_column = "octant_class_string"
            class_str = class_id[:-1]  # Remove 'o' suffix
        else:
            # Default to octant if no suffix
            class_column = "octant_class_string"
            class_str = class_id

        # Filter TorsionDF for the specified class
        class_df = torsion_df[torsion_df[class_column] == class_str]

        if len(class_df) == 0:
            _logger.warning("No disulfides found for class %s", class_id)
            return

        # Calculate means and standard deviations
        tor_means = class_df[
            ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
        ].mean()
        tor_stds = class_df[
            ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
        ].std()

        dist_means = class_df[
            ["ca_distance", "cb_distance", "sg_distance", "energy", "rho"]
        ].mean()
        dist_stds = class_df[
            ["ca_distance", "cb_distance", "sg_distance", "energy", "rho"]
        ].std()

        # Calculate pixel dimensions
        _width = figure_size[0] * dpi
        _height = figure_size[1] * dpi

        # Calculate scale factor based on DPI (300 DPI is the reference)
        scale_factor = dpi / 300

        # Scale font sizes based on DPI
        title_font_size = int(20 * scale_factor)
        axis_font_size = int(14 * scale_factor)
        tick_font_size = int(12 * scale_factor)
        text_font_size = int(10 * scale_factor)
        legend_font_size = int(10 * scale_factor)

        set_plotly_theme(theme)
        title = f"Class {class_id}: {len(class_df)} members"

        # Adjust vertical spacing based on scale factor
        vertical_spacing = 0.125 * (1 + 0.1 * (scale_factor - 1))

        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=vertical_spacing, column_widths=[1, 1]
        )

        fig.update_layout(
            title={
                "text": title,
                "xanchor": "center",
                "x": 0.5,
                "yanchor": "top",
                "font": {"size": title_font_size},
            },
            legend=dict(
                font=dict(size=legend_font_size),
                orientation="v",
                yanchor="top",
                xanchor="right",
                y=1.02,
                x=1,
            ),
            width=_width,
            height=_height,
            margin=dict(t=50 * scale_factor, b=50 * scale_factor),
        )

        fig.add_trace(
            go.Bar(
                x=["X1", "X2", "X3", "X2'", "X1'"],
                y=tor_means[:5],
                name="Torsion Angle (°) ",
                width=0.67,  # Control individual bar widths
                error_y=dict(
                    type="data",
                    array=tor_stds,
                    width=4 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[
                    f"{val:.2f} ± {std:.2f}"
                    for val, std in zip(tor_means[:5], tor_stds[:5])
                ],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="torsion",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=["rho"],
                y=[dist_means[4] * 100],
                name="ρ (°) * 100",
                error_y=dict(
                    type="data",
                    array=[dist_stds[4]],
                    width=4 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[f"{dist_means[4] * 100:.2f} ± {dist_stds[4]:.2f}"],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="rho",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text="Dihedral Angle (°)",
            range=[-200, 200],
            row=1,
            col=1,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )
        fig.update_yaxes(
            range=[0, 320],
            row=2,
            col=2,
            tickfont=dict(size=tick_font_size),
        )

        fig.add_trace(
            go.Bar(
                x=["Strain Energy (kcal/mol)"],
                y=[dist_means[3]],
                name="Energy (kcal/mol)",
                error_y=dict(
                    type="data",
                    array=[dist_stds[3].tolist()],
                    width=4 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[f"{dist_means[3]:.2f} ± {dist_stds[3]:.2f}"],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="energy",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.update_traces(width=0.5 * scale_factor, row=1, col=2)

        fig.update_yaxes(
            title_text="kcal/mol",
            range=[0, 8],
            row=1,
            col=2,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )

        fig.add_trace(
            go.Bar(
                x=["Cα Distance (Å)", "Cβ Distance (Å)", "Sγ Distance (Å)"],
                y=[dist_means[0], dist_means[1], dist_means[2]],
                name="Distances (Å)",
                width=0.3,  # Control individual bar widths
                error_y=dict(
                    type="data",
                    array=[
                        dist_stds[0].tolist(),
                        dist_stds[1].tolist(),
                        dist_stds[2].tolist(),
                    ],
                    width=4.0 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[
                    f"{dist_means[0]:.2f} ± {dist_stds[0]:.2f}",
                    f"{dist_means[1]:.2f} ± {dist_stds[1]:.2f}",
                    f"{dist_means[2]:.2f} ± {dist_stds[2]:.2f}",
                ],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="distances",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Distance (Å)",
            range=[0, 8],
            row=2,
            col=1,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )
        # Update layout for the distance subplot to control bar spacing
        fig.update_layout(
            bargap=0.3,  # Increase gap between bars
            bargroupgap=0.1,  # Gap between bar groups
        )

        fig.add_trace(
            go.Bar(
                x=["Torsion Length (Å)"],
                y=[tor_means[5]],
                name="Torsion Length (Å)",
                error_y=dict(
                    type="data",
                    array=[tor_stds[5]],
                    width=4 * scale_factor,
                    visible=True,
                    thickness=1.25 * scale_factor,
                ),
                text=[f"{tor_means[5]:.2f} ± {tor_stds[5]:.2f}"],
                textposition="outside",
                textfont=dict(size=text_font_size),
                legendgroup="torsion_length",
                showlegend=True,
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(
            title_text="Torsion Length",
            range=[0, 350],
            row=2,
            col=2,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=tick_font_size),
        )
        fig.update_traces(width=0.5 * scale_factor, row=2, col=2)

        # Update x-axis fonts
        fig.update_xaxes(tickfont=dict(size=tick_font_size))

        if display:
            fig.show()

        if save:
            # Convert DPI to scale factor (300 DPI is considered standard, so scale = dpi/300)
            scale = dpi / 300
            fig.write_image(fname, scale=scale)

    @staticmethod
    def plot_3d_hexbin_leftright(
        loader,
        width: int = 800,
        height: int = 600,
        gridsize: int = 60,
        tormin: float = -180.0,
        tormax: float = 180.0,
        scaling: str = "sqrt",
        column1: str = "chi2",
        column2: str = "chi4",
        title: str = None,
    ) -> None:
        """
        Create 3D hexbin plots for left and right-handed chi2-chi4 correlations with customizable z-scaling.

        :param loader: Loader object to retrieve torsion data
        :type loader: proteusPy.PDB_SS
        :param width: Window width in pixels
        :type width: int, optional
        :default width: 800
        :param height: Window height in pixels
        :type height: int, optional
        :default height: 600
        :param gridsize: Number of bins for hexbin
        :type gridsize: int, optional
        :default gridsize: 30
        :param tormin: Minimum torsion angle
        :type tormin: float, optional
        :default tormin: -180.0
        :param tormax: Maximum torsion angle
        :type tormax: float, optional
        :default tormax: 180.0
        :param scaling: Scaling method for z-values ('linear', 'sqrt', 'log', 'power')
        :type scaling: str, optional
        :default scaling: 'sqrt'
        :param column1: Name of the first column (x-axis)
        :type column1: str, optional
        :default column1: 'chi2'
        :param column2: Name of the second column (y-axis)
        :type column2: str, optional
        :default column2: 'chi4'
        """
        title = f"{column1} - {column2} Correlation" if title is None else title

        _SS_df = loader.getTorsions()

        _left = _SS_df["chi3"] <= 0.0
        _right = _SS_df["chi3"] > 0.0

        _SS_df_Left = _SS_df[_left]
        _SS_df_Right = _SS_df[_right]

        try:
            # Ensure width and height are integers
            width = int(width)
            height = int(height)

            # Extract data
            x_left = _SS_df_Left[column1]
            y_left = _SS_df_Left[column2]
            x_right = _SS_df_Right[column1]
            y_right = _SS_df_Right[column2]

            # Create 2D histogram bins for both datasets
            bins_left, xedges_left, yedges_left = np.histogram2d(
                x_left,
                y_left,
                bins=gridsize,
                range=[[tormin - 1, tormax + 1], [tormin - 1, tormax + 1]],
            )
            bins_right, xedges_right, yedges_right = np.histogram2d(
                x_right, y_right, bins=gridsize, range=[[tormin, tormax], [-180, 180]]
            )

            # Apply scaling to bin counts using match
            match scaling:
                case "linear":
                    scaled_bins_left = bins_left.T
                    scaled_bins_right = bins_right.T
                    scale_label = "linear scale"
                case "sqrt":
                    scaled_bins_left = np.sqrt(bins_left.T)
                    scaled_bins_right = np.sqrt(bins_right.T)
                    scale_label = "sqrt scale"
                case "log":
                    scaled_bins_left = np.log1p(bins_left.T)
                    scaled_bins_right = np.log1p(bins_right.T)
                    scale_label = "log scale"
                case "power":
                    power = 0.3
                    scaled_bins_left = np.power(bins_left.T, power)
                    scaled_bins_right = np.power(bins_right.T, power)
                    scale_label = f"power scale ({power})"
                case _:
                    raise ValueError(
                        f"Unsupported scaling method: {scaling}. Use 'linear', 'sqrt', 'log', or 'power'."
                    )

            # Debug: Print min and max of scaled values
            _logger.debug(
                "Left plot - Min: %s, Max: %s",
                scaled_bins_left.min(),
                scaled_bins_left.max(),
            )
            _logger.debug(
                "Right plot - Min: %s, Max: %s",
                scaled_bins_right.min(),
                scaled_bins_right.max(),
            )

            # Create mesh grid for plotting
            x_grid, y_grid = np.meshgrid(xedges_left[:-1], yedges_left[:-1])
            x_grid_r, y_grid_r = np.meshgrid(xedges_right[:-1], yedges_right[:-1])

            # Create PyVista plotter with two subplots
            plotter = pv.Plotter(shape=(1, 2), window_size=[width * 2, height])

            # Left-handed plot (subplot 0)
            plotter.subplot(0, 0)
            grid_left = pv.StructuredGrid(x_grid, y_grid, scaled_bins_left)
            grid_left.point_data["Height"] = scaled_bins_left.ravel(order="F")
            plotter.add_mesh(
                grid_left,
                scalars="Height",
                cmap="nipy_spectral",
                show_edges=False,
                clim=[scaled_bins_left.min(), scaled_bins_left.max()],
                show_scalar_bar=False,  # Add this line to hide the scalar bar
            )
            plotter.add_title(
                f"{title} (Left-handed, {scale_label})",
                font_size=8,
            )
            plotter.show_grid()
            # Add axes with custom labels
            plotter.add_axes(
                xlabel=column1,
                ylabel=column2,
                zlabel="cnt",
                line_width=2,
                color="black",
                interactive=True,
            )

            plotter.view_xy()
            plotter.enable_parallel_projection()

            # Right-handed plot (subplot 1)
            plotter.subplot(0, 1)
            grid_right = pv.StructuredGrid(x_grid_r, y_grid_r, scaled_bins_right)
            grid_right.point_data["Height"] = scaled_bins_right.ravel(order="F")
            plotter.add_mesh(
                grid_right,
                scalars="Height",
                cmap="nipy_spectral",
                show_edges=False,
                clim=[scaled_bins_right.min(), scaled_bins_right.max()],
                show_scalar_bar=False,  # Add this line to hide the scalar bar
            )
            plotter.add_title(
                f"{title} - (Right-handed, {scale_label})",
                font_size=8,
            )
            plotter.show_grid()
            plotter.view_xy()

            plotter.enable_parallel_projection()
            plotter.add_scalar_bar(
                title="Incidence",
                vertical=True,
                position_x=0.9,  # Right side
                position_y=0.25,
                width=0.05,
                height=0.5,
                title_font_size=14,
                label_font_size=12,
            )

            # Final adjustments
            plotter.reset_camera()
            plotter.link_views()
            plotter.show()

        except AttributeError as e:
            print(
                f"Error: DataFrame might be missing required columns (chi2, chi4): {e}"
            )
        except ValueError as e:
            print(f"Error: Invalid parameter value: {e}")

    @staticmethod
    def plot_3d_hexbin_df(
        df: "pandas.DataFrame",
        column1: str,
        column2: str,
        width: int = 1024,
        height: int = 1024,
        gridsize: int = 80,
        tormin: float = -180.0,
        tormax: float = 180.0,
        scaling: str = "sqrt",
        title: str = None,
    ) -> None:
        """
        Create a 3D hexbin plot for correlations between two columns from a
        single DataFrame with customizable z-scaling.

        :param df: Data containing the specified columns
        :type df: pandas.DataFrame
        :param column1: Name of the first column (x-axis)
        :type column1: str
        :param column2: Name of the second column (y-axis)
        :type column2: str
        :param width: Window width in pixels
        :type width: int, optional
        :default width: 1024
        :param height: Window height in pixels
        :type height: int, optional
        :default height: 1024
        :param gridsize: Number of bins for hexbin
        :type gridsize: int, optional
        :default gridsize: 80
        :param tormin: Minimum torsion angle
        :type tormin: float, optional
        :default tormin: -180
        :param tormax: Maximum torsion angle
        :type tormax: float, optional
        :default tormax: 180
        :param scaling: Scaling method for z-values ('linear', 'sqrt', 'log', 'power')
        :type scaling: str, optional
        :default scaling: 'sqrt'
        """
        title = f"{column1} - {column2} Correlation" if title is None else title

        try:
            # Validate column names exist in DataFrame
            if column1 not in df.columns or column2 not in df.columns:
                raise ValueError(
                    f"Columns '{column1}' or '{column2}' not found in DataFrame"
                )

            # Extract data from specified columns
            x = df[column1]
            y = df[column2]

            # Create 2D histogram bins
            bins, xedges, yedges = np.histogram2d(
                x, y, bins=gridsize, range=[[tormin, tormax], [tormin, tormax]]
            )

            # Apply scaling to bin counts using match
            match scaling:
                case "linear":
                    scaled_bins = bins.T
                    scale_label = "linear scale"
                case "sqrt":
                    scaled_bins = np.sqrt(bins.T)
                    scale_label = "sqrt scale"
                case "log":
                    scaled_bins = np.log1p(bins.T)
                    scale_label = "log scale"
                case "power":
                    power = 0.3
                    scaled_bins = np.power(bins.T, power)
                    scale_label = f"power scale ({power})"
                case _:
                    raise ValueError(
                        f"Unsupported scaling method: {scaling}. Use 'linear', 'sqrt', 'log', or 'power'."
                    )

            # Debug: Print min and max of scaled values
            print(f"Plot - Min: {scaled_bins.min()}, Max: {scaled_bins.max()}")

            # Create mesh grid for plotting
            x_grid, y_grid = np.meshgrid(xedges[:-1], yedges[:-1])

            # Create PyVista plotter (single view)
            plotter = pv.Plotter(window_size=[width, height])

            # Create and plot the grid
            grid = pv.StructuredGrid(x_grid, y_grid, scaled_bins)
            grid.point_data["Height"] = scaled_bins.ravel(order="F")
            plotter.add_mesh(
                grid,
                scalars="Height",
                cmap="nipy_spectral",
                show_edges=False,
                clim=[scaled_bins.min(), scaled_bins.max()],
                show_scalar_bar=False,  # Add this line to hide the scalar bar
            )
            plotter.add_title(f"{title} - ({scale_label})", font_size=8)

            # Add grid
            plotter.show_grid()
            # Add axes with custom labels
            plotter.add_axes(
                xlabel=column1,
                ylabel=column2,
                zlabel="cnt",
                line_width=2,
                color="black",
                interactive=True,
            )
            plotter.view_xy()
            plotter.enable_parallel_projection()
            # Scalar bar on the right side with smaller text
            plotter.add_scalar_bar(
                title="Incidence",
                vertical=True,
                position_x=0.9,  # Right side
                position_y=0.25,
                width=0.05,
                height=0.5,
                title_font_size=14,
                label_font_size=8,
            )

            # Final adjustments
            plotter.reset_camera()
            plotter.show()  # Native rendering

        except AttributeError as e:
            print(
                f"Error: DataFrame might be missing required columns ({column1}, {column2}): {e}"
            )
        except ValueError as e:
            print(f"Error: Invalid parameter value: {e}")

        return

    # End of file
