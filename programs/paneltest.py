'''
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 11/2/2023
'''

import sys
import time
import pyvista as pv
from pyvista import examples
import panel as pn

import proteusPy

from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.DisulfideList import DisulfideList

pn.extension('vtk', sizing_mode='stretch_width', template='fast')


pn.state.template.param.update(title=f"RCSB Disulfide Browser")
def click_plot(event):
    """Force a re-render of the currently selected disulfide. Removes the pane
    and re-adds it to the panel.

    Returns:
        None
    """
    global render_win
    plotter = pv.Plotter(notebook=False)

    vtkpan = pn.pane.VTK(plotter.ren_win, margin=0, sizing_mode='stretch_both', 
                         orientation_widget=orientation_widget,
                         enable_keybindings=enable_keybindings, min_height=500
                         )
    # this position is dependent on the vtk panel position in the render_win pane!
    render_win[1] = vtkpan

def get_theme() -> str:
    """Return the current theme: 'default' or 'dark'

    Returns:
        str: The current theme
    """
    args = pn.state.session_args
    if "theme" in args and args["theme"][0] == b"dark":
        return "dark"
    return "default"

# Widgets

rcsb_ss_widget = pn.widgets.Select(name="Disulfide", value="ss", options=[])

button = pn.widgets.Button(name='Refresh', button_type='primary')
button.on_click(click_plot)

styles_group = pn.widgets.RadioBoxGroup(name='Rending Style', 
                                        options=['Split Bonds', 'CPK', 'Ball and Stick'], 
                                        inline=False)

single_checkbox = pn.widgets.Checkbox(name='Single View', value=True)

def update_single(click):
    """Toggle the rendering style radio box depending on the state of the Single View checkbox.
    
    Returns:
        None
    """
    single_checked = single_checkbox.value
    if single_checked is not True:
        styles_group.disabled = True
    else:
        styles_group.disabled = False
    click_plot(click)

# not used atm    
shadows_checkbox = pn.widgets.Checkbox(name='Shadows', value=False)

rcsb_selector_widget = pn.widgets.AutocompleteInput(name="RCSB ID", value="ss", restrict=True,
                                                    placeholder="Search Here", options=[])

# markdown panels for various text outputs
title_md = pn.pane.Markdown("Title")
output_md = pn.pane.Markdown("Output goes here")
info_md = pn.pane.Markdown("SS Info")
db_md = pn.pane.Markdown("Database Info goes here")

# controls on sidebar
ss_props = pn.WidgetBox('# Disulfide Selection',
                        rcsb_selector_widget, rcsb_ss_widget
                        )

ss_styles = pn.WidgetBox('# Rendering Styles',
                         styles_group, single_checkbox
                        )

ss_info = pn.WidgetBox('# Disulfide Info', info_md)

m = examples.download_st_helens().warp_by_scalar()

# default camera position
cpos = [
    (567000.9232163235, 5119147.423216323, 6460.423216322832),
    (562835.0, 5114981.5, 2294.5),
    (-0.4082482904638299, -0.40824829046381844, 0.8164965809277649)
]

# pyvista plotter
pl = pv.Plotter(notebook=True)
actor = pl.add_mesh(m, smooth_shading=True, lighting=True)
pl.camera_position = cpos #set camera position

# save initial camera properties
renderer = list(pl.ren_win.GetRenderers())[0]
initial_camera = renderer.GetActiveCamera()
initial_camera_pos = {
    "focalPoint": initial_camera.GetFocalPoint(),
    "position": initial_camera.GetPosition(),
    "viewUp": initial_camera.GetViewUp()
}

# Panel creation using the VTK Scene created by the plotter pyvista
orientation_widget = True
enable_keybindings = True
vtkpan = pn.pane.VTK(
    pl.ren_win, margin=0, sizing_mode='stretch_both', orientation_widget=orientation_widget,
    enable_keybindings=enable_keybindings, min_height=600
)

widgetbox = pn.Column(
          ss_props, ss_styles, ss_info  
        ).servable(target='sidebar'),        
        
pn.Column(
    "This example demonstrates the use of **VTK and pyvista** to display a *scene*",
    pn.Row(
        widgetbox,
        vtkpan.servable(title='VTK - Mt. St Helens')
    ), min_height=600
)
