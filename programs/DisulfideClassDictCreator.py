# DisulfideBond Class Analysis Dictionary creation
# Author: Eric G. Suchanek, PhD.
# (c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
# License: MIT
# Last Modification: 2/18/23
# Cα Cβ Sγ

# this workflow reads in the torsion database, groups it by torsions 
# to create the classes merges with the master class spreadsheet, and saves the 
# resulting dict to {DATA_DIR}PDB_SS_merged.csv

import pandas as pd
import numpy

import pyvista as pv
from pyvista import set_plot_theme

from Bio.PDB import *

# for using from the repo we 
import proteusPy
from proteusPy import *
from proteusPy.data import *
from proteusPy.Disulfide import *
from proteusPy.utility import Create_classes

merge_cols = ['chi1_s','chi2_s','chi3_s','chi4_s','chi5_s','class_id','SS_Classname','FXN','count','incidence','percentage','ca_distance_mean',
'ca_distance_std','torsion_length_mean','torsion_length_std','energy_mean','energy_std', 'ss_id']

def load_class_dict(fname=f'{DATA_DIR}PDB_ss_classes_dict.pkl') -> dict:
    with open(fname,'rb') as f:
        res = pickle.load(f)
        return res

def build_class_df(class_df, group_df):
    ss_id_col = group_df['ss_id']
    result_df = pd.concat([class_df, ss_id_col], axis=1)
    return result_df

def ss_id_dict(df):
    ss_id_dict = dict(zip(df['SS_Classname'], df['ss_id']))
    return ss_id_dict

def sslist_from_classid(classid: str, loader: DisulfideLoader, classdict: dict) -> DisulfideList:
    res = DisulfideList([], 'tmp')
    
    try:
        sslist = classdict[classid]
        res = DisulfideList([loader[ssid] for ssid in sslist], 'classid')
        return res
    except KeyError:
        print(f'No class: {classid}')

def list_binary_classes(ssdict):
    for k,v in enumerate(ssdict):
        print(f'Class: |{k}|, |{v}|')

PDB_SS = Load_PDB_SS(verbose=True, subset=False)
PDB_SS.describe()

tors_df = PDB_SS.getTorsions()
#print(f'{tors_df.describe()}')


grouped = Create_classes(tors_df)
grouped.to_csv(f'{DATA_DIR}PDB_ss_classes.csv')

grouped_summary = grouped.drop(columns=['ss_id'], axis=1)
grouped_summary.to_csv(f'{DATA_DIR}PDB_ss_classes_summary.csv')
print(f'{grouped.head(32)}')

class_cols = ['Idx','chi1_s','chi2_s','chi3_s','chi4_s','chi5_s','class_id','SS_Classname','FXN','count','incidence','percentage','ca_distance_mean',
'ca_distance_std','torsion_length_mean','torsion_length_std','energy_mean','energy_std']

class_df = pd.read_csv(f'{DATA_DIR}PDB_ss_classes_master2.csv', dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})
class_df['FXN'].str.strip()
class_df['SS_Classname'].str.strip()
class_df['class_id'].str.strip()

print(f'{class_df.head(32)}')

merged = build_class_df(class_df, grouped)
merged.drop(columns=['Idx'], inplace=True)
merged.to_csv(f'{DATA_DIR}PDB_SS_merged.csv')

print(f'Testing read of: {SS_CLASS_DICT_FILE}')
ssdict2 = load_class_dict()
print(f'{ssdict2}')

exit()

# end of file

