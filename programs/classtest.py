#!/usr/bin/env python
# coding: utf-8
'''
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 11/2/2023
'''

import panel as pn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hvplot.pandas
pn.extension('tabulator')

#pn.extension('plotly', 'tabulator')
import proteusPy

from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.DisulfideClasses import enumerate_sixclass_fromlist
from proteusPy.DisulfideClasses import plot_count_vs_classid, plot_count_vs_class_df

_vers = 0.1

_default_binary = "00000"
_binarylist = [
    "00000", "00002", "00022", "00200", "00202", "00220", "00222",
    "02000", "02002", "02020", "02022", "02200", "02202", "02220",
    "20000", "20002", "20020", "20022", "20200", "20202", "20220",
    "20222", "22000", "22002", "22020", "22022", "22200", "22202",
    "22220", "22222"
    ]

PDB_SS = Load_PDB_SS(verbose=True, subset=False)

vers = PDB_SS.version
tot = PDB_SS.TotalDisulfides
pdbs = len(PDB_SS.SSDict)
orientation_widget = True
enable_keybindings = True

RCSB_list = sorted(PDB_SS.IDList)

# pn.state.template.param.update(title=f"RCSB Disulfide Class Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{vers}")

def get_theme() -> str:
    """Return the current theme: 'default' or 'dark'

    Returns:
        str: The current theme
    """
    args = pn.state.session_args
    if "theme" in args and args["theme"][0] == b"dark":
        return "dark"
    return "light"

# Widgets

binary_class_widget = pn.widgets.Select(name="Binary Class", value=_default_binary, options=_binarylist)


# markdown panels for various text outputs
title_md = pn.pane.Markdown("Title")
output_md = pn.pane.Markdown("Output goes here")
db_md = pn.pane.Markdown("Database Info goes here")

info_md = pn.pane.Markdown("SS Info")
ss_info = pn.WidgetBox('# Disulfide Info', info_md)
db_info = pn.Column('### RCSB Database Info', db_md)

# controls on sidebar
widgets = pn.WidgetBox('# Binary Class Selection',
                        binary_class_widget)

# Callbacks

def update_title(ss):
    src = ss.pdb_id
    name = ss.name

    title = f'## {name}'
    title_md.object = title

@pn.depends(binary_class_widget.param.value)
def update_info(ss):
    '''Update the info for the input disulfide.
    '''
    
    src = ss.pdb_id
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution

    info_string = f'### {name}  \n**Resolution:** {resolution:.2f} Å  \n**Energy:** {enrg:.2f} kcal/mol  \n**Cα distance:** {ss.ca_distance:.2f} Å  \n**Cβ distance:** {ss.cb_distance:.2f} Å  \n**Torsion Length:** {ss.torsion_length:.2f}°'
    info_md.object = info_string
    print(f'Info: {info_string}')

@pn.depends(binary_class_widget.param.value)
def update_sixclass_info(binary_class, loader=PDB_SS):
    '''Update the list of sixfold classes based on the binary class selected.
    '''
    ss_list = loader.tclass.binary_to_six_class(class_id)

    info_string = f'### {ss_list}'
    print(f'Info: {info_string}')
    info_md.object = info_string

@pn.depends(binary_class_widget.param.value)
def Oplot_classes(class_id, loader=PDB_SS, theme='light'):
    ss_list = loader.tclass.binary_to_six_class(class_id)
    df = enumerate_sixclass_fromlist(PDB_SS, ss_list)
    fig = plot_count_vs_class_df(df, class_id, theme=theme)
    return pn.pane.Plotly(fig)

@pn.depends(binary_class_widget.param.value)
def plot_classes(class_id, loader=PDB_SS, theme='light'):
    ss_list = loader.tclass.binary_to_six_class(class_id)
    df = enumerate_sixclass_fromlist(PDB_SS, ss_list)
    print(f'{df.head(5)}')
    return(df[class_id].hvplot())

LHSpiral_neg = PDB_SS.tclass.binary_to_six_class("00000")
RHSpiral_neg = PDB_SS.tclass.binary_to_six_class("02220")

binary_class_widget.param.watch(update_sixclass_info, 'value')

dash2 = pn.Column(binary_class_widget, info_md)
# dash2.servable()


# Create a Panel layout
#layout = pn.Column(binary_class_widget, pn.Column(plot_classes))
layout = pn.Column(binary_class_widget, plot_classes, info_md, )

# Show the Panel
layout.servable()

