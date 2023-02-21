# DisulfideBond Class Analysis Dictionary creation
# Author: Eric G. Suchanek, PhD.
# (c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
# License: MIT
# Last Modification: 2/18/23
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
    in the proteusPy disulfide database
    '''

    def __init__(self, verbose=False, bootstrap=False) -> None:
        self.verbose = verbose
        self.classdict = {}
        self.classdf = None

        if bootstrap:
            if self.verbose:
                print(f'--> DisulfideClass_Constructor(): Building SS classes...')
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

    def list_classes(self):
        for k,v in enumerate(self.classdict):
            print(f'Class: |{k}|, |{v}|')

    #  class_cols = ['Idx','chi1_s','chi2_s','chi3_s','chi4_s','chi5_s','class_id','SS_Classname','FXN',
    # 'count','incidence','percentage','ca_distance_mean',
    # 'ca_distance_std','torsion_length_mean','torsion_length_std','energy_mean','energy_std']

    def build_yourself(self):
        '''
        Builds the internal dictionary mapping the disulfide class names to their respective members.
        The classnames are defined by the sign of the dihedral angles, per XXX', the list of SS within
        the database classified, and the resulting dict created.
        '''

        def ss_id_dict(df):
            ss_id_dict = dict(zip(df['SS_Classname'], df['ss_id']))
            return ss_id_dict

        PDB_SS = proteusPy.DisulfideLoader.Load_PDB_SS(verbose=self.verbose, subset=False)
        if self.verbose:
            PDB_SS.describe()

        tors_df = PDB_SS.getTorsions()

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): creating SS classes...')

        grouped = Create_classes(tors_df)
        self.class_df = grouped

        # grouped.to_csv(f'{DATA_DIR}PDB_ss_classes.csv')
        if self.verbose:
            print(f'{grouped.head(32)}')

        #grouped_summary = grouped.drop(columns=['ss_id'], axis=1)
        #grouped_summary.to_csv(f'{DATA_DIR}PDB_ss_classes_summary.csv')
        
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

        merged = self.build_class_df(class_df, grouped)
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
            print(f'--> DisulfideClass_Constructor(): initialization complete.')
        
        return

        
# class definition ends

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

