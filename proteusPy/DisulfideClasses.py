'''
Functions to create Disulfide Bond structural classes based on
dihedral angle rules.

Author: Eric G. Suchanek, PhD. \n

(c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
License: MIT
Last Modification: 2/18/23

'''

import copy
from io import StringIO
import time
import datetime
import pickle

import pandas as pd
import numpy as np
import proteusPy

from proteusPy.DisulfideLoader import DisulfideLoader

from proteusPy.data import DATA_DIR, SS_CLASS_DICT_FILE, CLASSOBJ_FNAME
from proteusPy.data import SS_CLASS_DEFINITIONS
from proteusPy.angle_annotation import AngleAnnotation
from proteusPy.ProteusGlobals import DPI

def create_classes(df):
    """
    Group the DataFrame by the sign of the chi columns and create a new class ID column for each unique grouping.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'cb_distance', 'torsion_length', and 'energy'.
    :return: A pandas DataFrame containing columns 'class_id', 'ss_id', and 'count', where 'class_id' is a unique identifier for each grouping of chi signs, 'ss_id' is a list of all 'ss_id' values in that grouping, and 'count' is the number of rows in that grouping.
    
    Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'ss_id': [1, 2, 3, 4, 5],
    ...    'chi1': [1.0, -1.0, 1.0, 1.0, -1.0],
    ...    'chi2': [-1.0, -1.0, -1.0, 1.0, 1.0],
    ...    'chi3': [-1.0, 1.0, -1.0, 1.0, -1.0],
    ...    'chi4': [1.0, -1.0, 1.0, -1.0, 1.0],
    ...    'chi5': [1.0, -1.0, -1.0, -1.0, -1.0],
    ...    'ca_distance': [3.1, 3.2, 3.3, 3.4, 3.5],
    ...    'cb_distance': [3.1, 3.2, 3.3, 3.4, 3.5],
    ...    'torsion_length': [120.1, 120.2, 120.3, 120.4, 121.0],
    ...    'energy': [-2.3, -2.2, -2.1, -2.0, -1.9]
    ... })
    >>> create_classes(df)
      class_id ss_id  count  incidence  percentage
    0    00200   [2]      1        0.2        20.0
    1    02020   [5]      1        0.2        20.0
    2    20020   [3]      1        0.2        20.0
    3    20022   [1]      1        0.2        20.0
    4    22200   [4]      1        0.2        20.0

    """
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)
    
    # Create a new column with the class ID for each row
    class_id_column = 'class_id'
    df[class_id_column] = (df[sign_columns] + 1).apply(lambda x: ''.join(x.astype(str)), axis=1)

    # Group the DataFrame by the class ID and return the grouped data
    grouped = df.groupby(class_id_column)['ss_id'].unique().reset_index()
    grouped['count'] = grouped['ss_id'].apply(lambda x: len(x))
    grouped['incidence'] = grouped['ss_id'].apply(lambda x: len(x)/len(df))
    grouped['percentage'] = grouped['incidence'].apply(lambda x: 100 * x)

    return grouped

def angle_within_range(angle, min_angle, max_angle):
    """
    Check whether the given angle is within the specified range.

    Parameters:
        angle (float): The angle to check, in degrees.
        min_angle (float): The minimum angle in the range, in degrees.
        max_angle (float): The maximum angle in the range, in degrees.

    Returns:
        bool: True if the angle is within the range, False otherwise.
    """
    import math
    # Convert angles to radians
    angle_rad = math.radians(angle)
    min_angle_rad = math.radians(min_angle)
    max_angle_rad = math.radians(max_angle)

    # Check whether the angle is within the range
    if min_angle_rad <= angle_rad <= max_angle_rad:
        return True
    else:
        return False

def get_quadrant(angle_deg):
    """
    Return the quadrant in which an angle in degrees lies.

    Parameters:
        angle_deg (float): The angle in degrees.

    Returns:
        int: The quadrant number (1, 2, 3, or 4) that the angle belongs to.
    """
    if angle_deg >= 0 and angle_deg < 90:
        return str(1)
    elif angle_deg >= 90 and angle_deg < 180:
        return str(2)
    elif angle_deg >= -180 and angle_deg < -90:
        return str(3)
    elif angle_deg >= -90 and angle_deg < 0:
        return str(4)
    else:
        raise ValueError("Invalid angle value: angle must be in the range [-180, 180).")

def torsion_to_sixclass(tors):
    '''
    Return the sextant class string for the input array of torsions.

    :param tors: Array of five torsions
    :return: Sextant string
    '''

    from proteusPy.DisulfideClasses import get_sixth_quadrant
    
    res = [get_sixth_quadrant(x) for x in tors]
    return ''.join([str(r) for r in res])

def get_sixth_quadrant(angle_deg):
    """
    Return the sextant in which an angle in degrees lies if the area is described by dividing a unit circle into 6 equal segments.

    Parameters:
        angle_deg (float): The angle in degrees.

    Returns:
        int: The sextant (1-6) that the angle belongs to.
    """
    # Normalize the angle to the range [0, 360)
    angle_deg = angle_deg % 360

    if angle_deg >= 0 and angle_deg < 60:
        return str(6)
    elif angle_deg >= 60 and angle_deg < 120:
        return str(5)
    elif angle_deg >= 120 and angle_deg < 180:
        return str(4)
    elif angle_deg >= 180 and angle_deg < 240:
        return str(3)
    elif angle_deg >= 240 and angle_deg < 300:
        return str(2)
    elif angle_deg >= 300 and angle_deg < 360:
        return str(1)
    else:
        raise ValueError("Invalid angle value: angle must be in the range [-360, 360).")

def get_half_quadrant(angle_deg):
    """
    Returns the half-quadrant in which an angle in degrees lies.

    Parameters:
        angle_deg (float): The angle in degrees.

    Returns:
        int: The half-quadrant number (1-8) that the angle belongs to.
    """

    if angle_deg >= 0 and angle_deg < 45:
        return str(1)
    elif angle_deg >= 45 and angle_deg < 90:
        return str(2)
    elif angle_deg >= 90 and angle_deg < 135:
        return str(3)
    elif angle_deg >= 135 and angle_deg < 180:
        return str(4)
    elif angle_deg >= -45 and angle_deg < 0:
        return str(5)
    elif angle_deg >= -90 and angle_deg < -45:
        return str(6)
    elif angle_deg >= -135 and angle_deg < -90:
        return str(7)
    elif angle_deg >= -180 and angle_deg < -135:
        return str(8)
    else:
        raise ValueError("Invalid angle value: angle must be in the range [-180, 180).")


def create_quat_classes(df):
    """
    Add new columns to the input DataFrame with a 4-class encoding for input 'chi' values.

    Takes a DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance',
    'cb_distance', 'torsion_length', 'energy', and 'rho' and adds new columns based on the following rules:
    1. The 'chi_t' column is set to the quadrant in which the dihedral angle is located.
    
    A new column named `class_id` is also added, which is the concatenation of the `_t` columns. The DataFrame is then
    grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
    the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and the
    percentage of incidence.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
               'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
    :return: The input DataFrame with the added columns
    """

    new_cols = []
    for col_name in ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']:
        col = df[col_name]
        new_col = []
        for val in col:
            new_col.append(get_quadrant(val))
        new_col_name = col_name + '_t'
        new_cols.append(new_col_name)
        df[new_col_name] = new_col
    
    class_id_column = 'class_id'

    df['class_id'] = df[new_cols].apply(lambda x: ''.join(x), axis=1)

    # Group the DataFrame by the class ID and return the grouped data
    grouped = df.groupby(class_id_column)['ss_id'].unique().reset_index()
    grouped['count'] = grouped['ss_id'].apply(lambda x: len(x))
    grouped['incidence'] = grouped['ss_id'].apply(lambda x: len(x)/len(df))
    grouped['percentage'] = grouped['incidence'].apply(lambda x: 100 * x)

    return grouped

def filter_by_percentage(df, cutoff):
       """
       Filter a pandas DataFrame by incidence
   
       :param df: A Pandas DataFrame with an 'incidence' column to filter by
       :param cutoff: A numeric value specifying the minimum incidence required for a row to be included in the output
       :type df: pandas.DataFrame
       :type cutoff: float
       :return: A new Pandas DataFrame containing only rows where the incidence is greater than or equal to the cutoff
       :rtype: pandas.DataFrame
       """
       return df[df['percentage'] >= cutoff]

def get_ss_id(df: pd.DataFrame, cls: str) -> str:
    '''
    Returns the 'ss_id' value in the given DataFrame that corresponds to the
    input 'cls' string.
    '''
    filtered_df = df[df['class_id'] == cls]
    if len(filtered_df) == 0:
        raise ValueError(f"No rows found for class_id '{cls}'")
    elif len(filtered_df) > 1:
        raise ValueError(f"Multiple rows found for class_id '{cls}'")
    return filtered_df.iloc[0]['ss_id']


def get_section(angle_deg, basis):
    """
    Returns the section in which an angle in degrees lies if the section is described by dividing a unit circle into `basis` equal segments.

    Parameters:
        angle_deg (float): The angle in degrees.
        basis (int): The number of equal angular divisions into which the unit circle is divided.

    Returns:
        int: The section number (1-basis) that the angle belongs to.
    """
    # Normalize the angle to the range [-180, 180)
    angle_deg = angle_deg % 360
    if angle_deg < -180:
        angle_deg += 360
    elif angle_deg >= 180:
        angle_deg -= 360

    # Calculate the size of each segment
    segment_size = 360 / basis

    # Calculate the section number
    section = int(angle_deg // segment_size) + 1
    if section <= 0:
        section += basis

    return str(section)

def is_between(x, a, b):
    """
    Returns True if x is between a and b (inclusive), False otherwise.

    :param x: The input number to be tested.
    :type x: int or float
    :param a: The lower limit of the range to check against.
    :type a: int or float
    :param b: The upper limit of the range to check against.
    :type b: int or float
    :return: True if x is between a and b (inclusive), False otherwise.
    :rtype: bool
    """
    return a <= x <= b

def plot_class_chart(classes: int) -> None:
    """
    Create a Matplotlib pie chart with `classes` segments of equal size.

    :param classes: The number of segments to create in the pie chart.
    :type classes: int
    
    :return: None

    :Example:
    
    Create a pie chart with 4 equal segments.
    
    >>> plot_class_chart(4)
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from proteusPy.angle_annotation import AngleAnnotation

    # Helper function to draw angle easily.
    def plot_angle(ax, pos, angle, length=0.95, acol="C4", **kwargs):
        """
        Helper function to draw an angle.

        :param ax: The Matplotlib axis to draw on.
        :type ax: matplotlib.axis.Axis

        :param pos: The starting position of the angle.
        :type pos: tuple[float, float]

        :param angle: The angle in degrees to draw.
        :type angle: float

        :param length: The length of the angle in the plot.
        :type length: float

        :param acol: The color of the angle.
        :type acol: str

        :return: An AngleAnnotation object representing the angle.
        """
        vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        xy = np.c_[[length, 0], [0, 0], vec2*length].T + np.array(pos)
        ax.plot(*xy.T, color=acol)
        return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)

    fig, ax1= plt.subplots(sharex=True)

    # Set up the figure
    fig.suptitle("SS Torsion Classes")
    fig.set_dpi(DPI)
    fig.set_size_inches(4, 4)
    fig.canvas.draw()  # Need to draw the figure to define renderer

    # Showcase different text positions.
    ax1.margins(y=0.4)
    ax1.set_title("textposition")
    _text = f"${360/classes}°$"

    kw = dict(size=144, unit="pixels", text=_text)
    #am1 = AngleAnnotation(center, p1[1], p2[1], ax=ax, size=75, text=r"$\alpha$")

    am7 = plot_angle(ax1, (0, 0), 360/classes, 
                    textposition="outside", **kw)

    # Create a list of segment values
    values = [1 for _ in range(classes)]

    # Create the pie chart
    wedges, _ = ax1.pie(
        values, startangle=0, counterclock=True, wedgeprops=dict(width=0.65))

    # Set the chart title and size
    ax1.set_title(f'{classes}-Class Angular Layout')

    # Set the segment colors
    color_palette = plt.cm.get_cmap('tab20', classes)
    ax1.set_prop_cycle('color', [color_palette(i) for i in range(classes)])

    # Create the legend
    legend_labels = [f'{i+1}' for i in range(classes)]
    legend = ax1.legend(wedges, legend_labels, title='Class', loc='center right', bbox_to_anchor=(1.2, 0.5))

    # Set the legend fontsize
    plt.setp(legend.get_title(), fontsize='medium')
    plt.setp(legend.get_texts(), fontsize='small')

    # Show the chart

def plot_count_vs_class_df(df, title='title', 
                        theme='plotly_dark',
                        save=False, savedir='.'):
    """
    Plots a line graph of count vs class ID using Plotly.

    :param df: A pandas DataFrame containing the data to be plotted.
    :param title: A string representing the title of the plot (default is 'title').
    :param theme: A string representing the name of the theme to use. Can be either 'notebook' or 'plotly_dark'. Default is 'plotly_dark'.
    :return: None
    """
    import plotly_express as px

    fig = px.line(df, x='class_id', y='count', 
                title=f'{title}', 
                labels={'class_id': 'Class ID', 'count': 'Count'})
    
    if theme == 'light':
        fig.update_layout(template='plotly_white')
    else:
        fig.update_layout(template='plotly_dark')

    fig.update_layout(showlegend=True, title_x=0.5, title_font=dict(size=20), 
                    xaxis_showgrid=False, yaxis_showgrid=False)
    if save:
        fname = f'{savedir}/{title}.png'
        fig.write_image(fname, 'png')
    else:          
        fig.show()


def plot_count_vs_classid(df, cls=None, title='title', theme='light'):
    """
    Plots a line graph of count vs class ID using Plotly.

    :param df: A pandas DataFrame containing the data to be plotted.
    :param title: A string representing the title of the plot (default is 'title').
    :param theme: A string representing the theme of the plot. Anything other than `light` is in `plotly_dark`.
    :return: None
    """

    import plotly_express as px
    if cls is None:
        fig = px.line(df, x='class_id', y='count', title=f'{title}')
    else:
        subset = df[df['class_id'] == cls]
        fig = px.line(subset, x='class_id', y='count', title=f'{title}')
        
    fig.update_layout(xaxis_title='Class ID', yaxis_title='Count', showlegend=True, 
                    title_x=0.5)
    
    if theme == 'light':
        fig.update_layout(template='plotly_white')
    else:
        fig.update_layout(template='plotly_dark')
        
    fig.show()


def plot_binary_to_sixclass_incidence(loader: DisulfideLoader, light=True):
    '''
    Plot the incidence of all sextant Disulfide classes for a given binary class.

    :param loader: `proteusPy.DisulfideLoader` object
    '''
    def enumerate_sixclass_fromlist(sslist):
        x = []
        y = []

        for sixcls in sslist:
            if sixcls is not None:
                _y = loader.tclass.sslist_from_classid(sixcls)
                # it's possible to have 0 SS in a class
                if _y is not None:
                    # only append if we have both.
                    x.append(sixcls)
                    y.append(len(_y))

        sslist_df = pd.DataFrame(columns=['class_id', 'count'])
        sslist_df['class_id'] = x
        sslist_df['count'] = y
        return(sslist_df)

    clslist = loader.tclass.classdf['class_id']
    for cls in clslist:
        sixcls = loader.tclass.binary_to_six_class(cls)
        df = enumerate_sixclass_fromlist(sixcls)
        plot_count_vs_class_df(df, cls, theme='light')
    return

def enumerate_sixclass_fromlist(loader, sslist):
    x = []
    y = []

    for sixcls in sslist:
        if sixcls is not None:
            _y = loader.tclass.sslist_from_classid(sixcls)
            # it's possible to have 0 SS in a class
            if _y is not None:
                # only append if we have both.
                x.append(sixcls)
                y.append(len(_y))

    sslist_df = pd.DataFrame(columns=['class_id', 'count'])
    sslist_df['class_id'] = x
    sslist_df['count'] = y
    return(sslist_df)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
