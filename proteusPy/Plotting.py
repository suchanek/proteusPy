"""
This module provides various utility functions for disulfide bond analysis, 
including functions for highlighting worst-case structures based on distance 
and angle deviations, as well as generating pie charts to visualize disulfide 
torsional classes.

Functions:
- highlight_worst_structures: Highlights the worst structures for distance and angle deviations and annotates their names.
- plot_class_chart: Creates a Matplotlib pie chart with segments of equal size to represent disulfide torsional classes.

Dependencies:
- matplotlib
- numpy
- pandas

Last Revision: 2025-02-19 23:16:31 -egs-
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from proteusPy.angle_annotation import AngleAnnotation

DPI = 300

# uncomment the following line to set the default renderer to 'png'
# pio.renderers.default = "png"  # or 'svg'


def highlight_worst_structures(df, top_n=10, sample_percent=10):
    """
    Highlight the worst structures for distance and angle deviations and annotate their names.
    Also, add a subplot showing the worst structures aggregated by PDB_ID.

    :param df: DataFrame containing the disulfide information.
    :type df: pd.DataFrame
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


def plot_class_chart(classes: int) -> None:
    """
    Create a Matplotlib pie chart with `classes` segments of equal size.

    This function returns a figure representing the angular layout of
    disulfide torsional classes for input `n` classes.

    Parameters:
        classes (int): The number of segments to create in the pie chart.

    Returns:
        None

    Example:
    >>> from proteusPy.Plotting import plot_class_chart
    >>> plot_class_chart(8)

    This will create a pie chart with 8 equal segments, and represents
    the layout of the disulfide octant class definition.
    """

    matplotlib.use("TkAgg")  # or 'Qt5Agg', 'MacOSX', etc.

    # Helper function to draw angle easily.
    def plot_angle(ax, pos, angle, length=0.95, acol="C0", **kwargs):
        vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        xy = np.c_[[length, 0], [0, 0], vec2 * length].T + np.array(pos)
        ax.plot(*xy.T, color=acol)
        return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)

    # fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig, ax1 = plt.subplots(sharex=True)

    # ax1, ax2 = fig.subplots(1, 2, sharey=True, sharex=True)

    fig.suptitle("SS Torsion Classes")
    fig.set_dpi(DPI)
    fig.set_size_inches(6.2, 6)

    fig.canvas.draw()  # Need to draw the figure to define renderer

    # Showcase different text positions.
    ax1.margins(y=0.4)
    ax1.set_title("textposition")
    _text = f"${360/classes}°$"
    kw = dict(size=75, unit="points", text=_text)

    plot_angle(ax1, (0, 0), 360 / classes, textposition="outside", **kw)

    # Create a list of segment values
    # !!!
    values = [1 for _ in range(classes)]

    # Create the pie chart
    # fig, ax = plt.subplots()
    wedges, _ = ax1.pie(
        values,
        startangle=0,
        counterclock=False,
        wedgeprops=dict(width=0.65),
    )

    # Set the chart title and size
    ax1.set_title(f"{classes}-Class Angular Layout")

    # Set the segment colors
    color_palette = plt.cm.get_cmap("tab20", classes)
    ax1.set_prop_cycle("color", [color_palette(i) for i in range(classes)])

    # Create the legend
    legend_labels = [f"Class {i+1}" for i in range(classes)]
    legend = ax1.legend(
        wedges,
        legend_labels,
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
    )

    # Set the legend fontsize
    plt.setp(legend.get_title(), fontsize="large")
    plt.setp(legend.get_texts(), fontsize="medium")

    # Show the chart
    fig.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
