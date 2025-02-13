"""
Various utility functions for disulfide bond analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio

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
    for i, row in worst_structures.iterrows():
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
