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

Last Revision: 2025-02-25 23:51:14 -egs-
"""

# pylint: disable=C0301 # line too long

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from proteusPy.angle_annotation import AngleAnnotation

DPI = 300

# uncomment the following line to set the default renderer to 'png'
# pio.renderers.default = "png"  # or 'svg'


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
