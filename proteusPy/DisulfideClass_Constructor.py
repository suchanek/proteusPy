'''
DisulfideBond Class Analysis Dictionary creation
Author: Eric G. Suchanek, PhD.
(c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
License: MIT
Last Modification: 3/8/23

# this workflow reads in the torsion database, groups it by torsions 
# to create the classes merges with the master class spreadsheet, and saves the 
# resulting dict to {DATA_DIR}PDB_SS_merged.csv

Disulfide Class definition using the +/- formalism of Hogg et al. (Biochem, 2006, 45, 7429-7433), across
all 32 possible classes ($$2^5$$). Classes are named per Hogg's convention.


+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| IDX|   chi1_s |   chi2_s |   chi3_s |   chi4_s |   chi5_s |   class_id | SS_Classname   | FXN        |
+====+==========+==========+==========+==========+==========+============+================+============+
|  0 |       -1 |       -1 |       -1 |       -1 |       -1 |      00000 | -LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  1 |       -1 |       -1 |       -1 |       -1 |        1 |      00002 | 00002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  2 |       -1 |       -1 |       -1 |        1 |       -1 |      00020 | -LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  3 |       -1 |       -1 |       -1 |        1 |        1 |      00022 | 00022          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  4 |       -1 |       -1 |        1 |       -1 |       -1 |      00200 | -RHStaple      | Allosteric |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  5 |       -1 |       -1 |        1 |       -1 |        1 |      00202 | 00202          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  6 |       -1 |       -1 |        1 |        1 |       -1 |      00220 | 00220          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  7 |       -1 |       -1 |        1 |        1 |        1 |      00222 | 00222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  8 |       -1 |        1 |       -1 |       -1 |       -1 |      02000 | 02000          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  9 |       -1 |        1 |       -1 |       -1 |        1 |      02002 | 02002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 10 |       -1 |        1 |       -1 |        1 |       -1 |      02020 | -LHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 11 |       -1 |        1 |       -1 |        1 |        1 |      02022 | 02022          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 12 |       -1 |        1 |        1 |       -1 |       -1 |      02200 | -RHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 13 |       -1 |        1 |        1 |       -1 |        1 |      02202 | 02202          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 14 |       -1 |        1 |        1 |        1 |       -1 |      02220 | -RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 15 |       -1 |        1 |        1 |        1 |        1 |      02222 | 02222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 16 |        1 |       -1 |       -1 |       -1 |       -1 |      20000 | +/-LHSpiral    | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 17 |        1 |       -1 |       -1 |       -1 |        1 |      20002 | +LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 18 |        1 |       -1 |       -1 |        1 |       -1 |      20020 | +/-LHHook      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 19 |        1 |       -1 |       -1 |        1 |        1 |      20022 | +LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 20 |        1 |       -1 |        1 |       -1 |       -1 |      20200 | +/-RHStaple    | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 21 |        1 |       -1 |        1 |       -1 |        1 |      20202 | +RHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 22 |        1 |       -1 |        1 |        1 |       -1 |      20220 | +/-RHHook      | Catalytic  |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 23 |        1 |       -1 |        1 |        1 |        1 |      20222 | 20222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 24 |        1 |        1 |       -1 |       -1 |       -1 |      22000 | -/+LHHook      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 25 |        1 |        1 |       -1 |       -1 |        1 |      22002 | 22002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 26 |        1 |        1 |       -1 |        1 |       -1 |      22020 | +/-LHStaple    | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 27 |        1 |        1 |       -1 |        1 |        1 |      22022 | +LHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 28 |        1 |        1 |        1 |       -1 |       -1 |      22200 | -/+RHHook      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 29 |        1 |        1 |        1 |       -1 |        1 |      22202 | +RHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 30 |        1 |        1 |        1 |        1 |       -1 |      22220 | +/-RHSpiral    | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 31 |        1 |        1 |        1 |        1 |        1 |      22222 | +RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
'''
# Cα Cβ Sγ

# this workflow reads in the torsion database, groups it by torsions 
# to create the classes merges with the master class spreadsheet, and saves the 
# resulting dict to {DATA_DIR}PDB_SS_merged.csv

__pdoc__ = {'__all__': True}

import pandas as pd
import numpy

from io import StringIO

import pyvista as pv
from pyvista import set_plot_theme

from Bio.PDB import *

# for using from the repo we 
from proteusPy.data import SS_CLASS_DICT_FILE, SS_CLASS_DEFINITIONS, DATA_DIR
from proteusPy.DisulfideList import DisulfideList
from proteusPy.Disulfide import *

merge_cols = ['chi1_s','chi2_s','chi3_s','chi4_s','chi5_s','class_id','SS_Classname','FXN','count','incidence','percentage','ca_distance_mean',
'ca_distance_std','torsion_length_mean','torsion_length_std','energy_mean','energy_std', 'ss_id']

class DisulfideClass_Constructor():
    '''
    Class manages structural classes for the disulfide bonds contained
    in the proteusPy disulfide database.
    '''

    def __init__(self, verbose=True, bootstrap=False) -> None:
        self.verbose = verbose
        self.classdict = {}
        self.classdf = None
        self.sixclass_df = None

        if bootstrap:
            if self.verbose:
                print(f'-> DisulfideClass_Constructor(): Building SS classes...')
            self.build_yourself()
        else:
            self.classdict = self.load_class_dict()

    def load_class_dict(self, fname=f'{DATA_DIR}{SS_CLASS_DICT_FILE}') -> dict:
        with open(fname,'rb') as f:
            #res = pickle.load(f)
            self.classdict = pickle.load(f)
    
    def build_class_df(self, class_df, group_df):
        ss_id_col = group_df['ss_id']
        result_df = pd.concat([class_df, ss_id_col], axis=1)
        return result_df

    def list_binary_classes(self):
        for k,v in enumerate(self.classdict):
            print(f'Class: |{k}|, |{v}|')

    #  class_cols = ['Idx','chi1_s','chi2_s','chi3_s','chi4_s','chi5_s','class_id','SS_Classname','FXN',
    # 'count','incidence','percentage','ca_distance_mean',
    # 'ca_distance_std','torsion_length_mean','torsion_length_std','energy_mean','energy_std']

    def concat_dataframes(self, df1, df2):
        """
        Concatenates columns from one data frame into the other 
        and returns the new result.

        Parameters
        ----------
        df1 : pandas.DataFrame
            The first data frame.
        df2 : pandas.DataFrame
            The second data frame.

        Returns
        -------
        pandas.DataFrame
            The concatenated data frame.

        """
        # Merge the data frames based on the 'SS_Classname' column
        result = pd.merge(df1, df2, on='class_id')

        return result

    def build_yourself(self):
        '''
        Builds the internal structures needed for the loader, including binary and six-fold classes.
        The classnames are defined by the sign of the dihedral angles, per XXX', the list of SS within
        the database classified, and the resulting dict created.
        '''

        from proteusPy.DisulfideClasses import create_classes, create_six_class_df
        from proteusPy.DisulfideLoader import Load_PDB_SS

        def ss_id_dict(df):
            ss_id_dict = dict(zip(df['SS_Classname'], df['ss_id']))
            return ss_id_dict

        PDB_SS = Load_PDB_SS(verbose=self.verbose, subset=False)
        self.version = proteusPy.__version__

        if self.verbose:
            PDB_SS.describe()

        tors_df = PDB_SS.getTorsions()

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): creating binary SS classes...')
        grouped = create_classes(tors_df)        
        
        # grouped.to_csv(f'{DATA_DIR}PDB_ss_classes.csv')
        
        # this file is hand made. Do not change it. -egs-
        #class_df = pd.read_csv(f'{DATA_DIR}PDB_ss_classes_master2.csv', dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})

        # !!! df = pd.read_csv(pd.compat.StringIO(csv_string))
        # class_df = pd.read_csv(f'{DATA_DIR}PDB_SS_class_definitions.csv', dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})
        
        class_df = pd.read_csv(StringIO(SS_CLASS_DEFINITIONS), dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})
        class_df['FXN'].str.strip()
        class_df['SS_Classname'].str.strip()
        class_df['class_id'].str.strip()

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): merging...')

        merged = self.concat_dataframes(class_df, grouped)
        merged.drop(columns=['Idx'], inplace=True)

        classdict = ss_id_dict(merged)
        self.classdict = classdict

        merged.to_csv(f'{DATA_DIR}PDB_SS_merged.csv')
        self.classdf = merged.copy()

        fname = f'{DATA_DIR}{SS_CLASS_DICT_FILE}'

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): writing {fname}...')

        with open(fname, "wb+") as f:
            pickle.dump(classdict, f)

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): creating sixfold SS classes...')
        
        grouped_sixclass = create_six_class_df(tors_df)
        grouped_sixclass.to_csv(f'{DATA_DIR}PDB_ss_six_classes.csv')
        self.sixclass_df = grouped_sixclass

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): initialization complete.')
        
        return
    
    def save(self, savepath=DATA_DIR):
        '''
        Save a copy of the fully instantiated Loader to the specified file.

        :param savepath: Path to save the file, defaults to DATA_DIR
        :param fname: Filename, defaults to LOADER_FNAME
        :param verbose: Verbosity, defaults to False
        :param cutoff: Distance cutoff used to build the database, -1 means no cutoff.
        '''
        self.version = proteusPy.__version__

        fname = CLASSOBJ_FNAME

        _fname = f'{savepath}{fname}'

        if self.verbose:
            print(f'-> DisulfideLoader.save(): Writing {_fname}... ')
        
        with open(_fname, 'wb+') as f:
            pickle.dump(self, f)
        
        if self.verbose:
            print(f'-> DisulfideLoader.save(): Done.')
        
# class definition ends
def plot_class_chart(classes: int) -> None:
    """
    Create a Matplotlib pie chart with `classes` segments of equal size.

    Parameters:
        classes (int): The number of segments to create in the pie chart.

    Returns:
        None

    Example:
    >>> plot_class_chart(4)

    This will create a pie chart with 4 equal segments.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    from proteusPy.angle_annotation import AngleAnnotation

    # Helper function to draw angle easily.
    def plot_angle(ax, pos, angle, length=0.95, acol="C0", **kwargs):
        vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        xy = np.c_[[length, 0], [0, 0], vec2*length].T + np.array(pos)
        ax.plot(*xy.T, color=acol)
        return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)

    #fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig, ax1= plt.subplots(sharex=True)

    #ax1, ax2 = fig.subplots(1, 2, sharey=True, sharex=True)

    fig.suptitle("SS Torsion Classes")
    fig.set_dpi(220)
    fig.set_size_inches(8, 6)

    fig.canvas.draw()  # Need to draw the figure to define renderer

    # Showcase different text positions.
    ax1.margins(y=0.4)
    ax1.set_title("textposition")
    _text = f"${360/classes}°$"
    kw = dict(size=75, unit="points", text=_text)

    #am6 = plot_angle(ax1, (2.0, 0), 60, textposition="inside", **kw)
    am7 = plot_angle(ax1, (0, 0), 360/classes, textposition="outside", **kw)

    # Create a list of segment values
    # !!!
    values = [1 for _ in range(classes)]

    # Create the pie chart
    #fig, ax = plt.subplots()
    wedges, _ = ax1.pie(
        values, startangle=0, counterclock=False, wedgeprops=dict(width=0.65))

    # Set the chart title and size
    ax1.set_title(f'{classes}-Class Angular Layout')

    # Set the segment colors
    color_palette = plt.cm.get_cmap('tab20', classes)
    ax1.set_prop_cycle('color', [color_palette(i) for i in range(classes)])

    # Create the legend
    legend_labels = [f'Class {i+1}' for i in range(classes)]
    legend = ax1.legend(wedges, legend_labels, title='Classes', loc='center left', bbox_to_anchor=(.8, 0.5))

    # Set the legend fontsize
    plt.setp(legend.get_title(), fontsize='large')
    plt.setp(legend.get_texts(), fontsize='medium')

    # Show the chart
    fig.show()

def add_sign_columns(df):
    """
    Create new columns with the sign of each dihedral angle (chi1-chi5)
    column and return a new DataFrame with the additional columns.
    This is used to build disulfide classes.

    :param df: pandas.DataFrame - The input DataFrame containing 
    the dihedral angle (chi1-chi5) columns.
        
    :return: A new DataFrame containing the columns 'ss_id', 'chi1_s', 
    'chi2_s', 'chi3_s', 'chi4_s', 'chi5_s' which represent the signs of 
    the dihedral angle columns in the input DataFrame.
        
    Example:
    >>> import pandas as pd
    >>> data = {'ss_id': [1, 2, 3], 'chi1': [-2, 1.0, 1.3], 'chi2': [0.8, -1.5, 0], 
    ...         'chi3': [-1, 2, 0.1], 'chi4': [0, 0.9, -1.1], 'chi5': [0.2, -0.6, -0.8]}
    >>> df = pd.DataFrame(data)
    >>> res_df = add_sign_columns(df)
    >>> print(res_df)
       ss_id  chi1_s  chi2_s  chi3_s  chi4_s  chi5_s
    0      1      -1       1      -1       1       1
    1      2       1      -1       1       1      -1
    2      3       1       1       1      -1      -1
    """
    # Create columns for the resulting DF
    tors_vector_cols = ['ss_id', 'chi1_s', 'chi2_s', 'chi3_s', 'chi4_s', 'chi5_s']
    res_df = pd.DataFrame(columns=tors_vector_cols)
    
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)
    res_df = df[tors_vector_cols].copy()
    return res_df

def group_by_sign(df):
    '''
    Group a DataFrame by the sign of each dihedral angle (chi1-chi5) column.

    This function creates new columns in the input DataFrame with the sign of each chi column, 
    and groups the DataFrame by these new columns. The function returns the aggregated data, including 
    the mean and standard deviation of the 'ca_distance', 'torsion_length', and 'energy' columns.

    :param df: The input DataFrame to group by sign.
    :type df: pandas.DataFrame
    :return: The DataFrame grouped by sign, including means and standard deviations.
    :rtype: pandas.DataFrame

    Example:
    >>> df = pd.DataFrame({'pdbid': ['1ABC', '1DEF', '1GHI', '1HIK'],
    ...                    'chi1': [120.0, -45.0, 70.0, 90],
    ...                    'chi2': [90.0, 180.0, -120.0, -90],
    ...                    'chi3': [-45.0, -80.0, 20.0, 0],
    ...                    'chi4': [0.0, 100.0, -150.0, -120.0],
    ...                    'chi5': [-120.0, -10.0, 160.0, -120.0],
    ...                    'ca_distance': [3.5, 3.8, 2.5, 3.3],
    ...                    'torsion_length': [3.2, 2.8, 3.0, 4.4],
    ...                    'energy': [-12.0, -10.0, -15.0, -20.0]})
    >>> grouped = group_by_sign(df)
    >>> grouped
       chi1_s  chi2_s  chi3_s  chi4_s  chi5_s  ca_distance_mean  ca_distance_std  torsion_length_mean  torsion_length_std  energy_mean  energy_std
    0      -1       1      -1       1      -1               3.8              NaN                  2.8                 NaN        -10.0         NaN
    1       1      -1       1      -1      -1               3.3              NaN                  4.4                 NaN        -20.0         NaN
    2       1      -1       1      -1       1               2.5              NaN                  3.0                 NaN        -15.0         NaN
    3       1       1      -1       1      -1               3.5              NaN                  3.2                 NaN        -12.0         NaN
    
    '''
    
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)

    # Group the DataFrame by the sign columns and return the aggregated data
    group_columns = sign_columns
    agg_columns = ['ca_distance', 'torsion_length', 'energy']
    grouped = df.groupby(group_columns)[agg_columns].agg(['mean', 'std'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    return grouped.reset_index()

def Create_classes(df):
    """
    Group the DataFrame by the sign of the chi columns and create a new class ID column for each unique grouping.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'torsion_length', and 'energy'.
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
    ...    'torsion_length': [120.1, 120.2, 120.3, 120.4, 121.0],
    ...    'energy': [-2.3, -2.2, -2.1, -2.0, -1.9]
    ... })
    >>> Create_classes(df)
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

# end of file

