"""
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 10/22/2024
"""

# pylint: disable=C0301 # line too long
# pylint: disable=C0413 # wrong import order
# pylint: disable=C0103 # wrong variable name
# pylint: disable=W0212 # access to a protected member _render of a client class
# pylint: disable=W0612 # unused variable
# pylint: disable=W0613 # unused argument
# pylint: disable=W0603 # Using global for variable

import logging
import os

import numpy as np
import panel as pn
import pyvista as pv

from proteusPy import (
    FONTSIZE,
    Disulfide,
    DisulfideList,
    Load_PDB_SS,
    configure_master_logger,
    create_logger,
    get_jet_colormap,
    grid_dimensions,
)
from proteusPy.atoms import BOND_RADIUS
from proteusPy.ProteusGlobals import WINSIZE

# Set PyVista to use offscreen rendering if the environment variable is set
if os.getenv("PYVISTA_OFF_SCREEN", "false").lower() == "true":
    pv.OFF_SCREEN = True

pn.extension("vtk", sizing_mode="stretch_width", template="fast")

_vers = 0.90

_logger = create_logger("dbviewer", log_level=logging.INFO)

configure_master_logger("dbviewer.py")
_logger.info("Starting Panel Disulfide Viewer v%s.", _vers)

current_theme = pn.config.theme
_logger.info("Current Theme: %s", current_theme)

# defaults for the UI

_rcsid_default = "2q7q"
_default_ss = "2q7q_75D_140D"
_ssidlist = [
    "2q7q_75D_140D",
    "2q7q_81D_113D",
    "2q7q_88D_171D",
    "2q7q_90D_138D",
    "2q7q_91D_135D",
    "2q7q_98D_129D",
    "2q7q_130D_161D",
]

_rcsid = "2q7q"
_style = "Split Bonds"
_single = True

# Set up logging
# create a local logger
_logger = create_logger("DBViewer", log_level=logging.INFO)

# globals
ss_state = {}
RCSB_list = []
PDB_SS = None

# Widgets
styles_group = pn.widgets.RadioBoxGroup(
    name="Rending Style",
    options=["Split Bonds", "CPK", "Ball and Stick"],
    inline=False,
)

single_checkbox = pn.widgets.Checkbox(name="Single View", value=True)
rcsb_ss_widget = pn.widgets.Select(
    name="Disulfide", value=_default_ss, options=_ssidlist
)

rcsb_selector_widget = pn.widgets.AutocompleteInput(
    name="RCSB ID (start typing)",
    value=_rcsid_default,
    restrict=True,
    placeholder="Search Here",
    options=RCSB_list,
)

# controls on sidebar
ss_props = pn.WidgetBox(
    "# Disulfide Selection", rcsb_selector_widget, rcsb_ss_widget
).servable(target="sidebar")


# Replace single checkbox with a selector widget
view_selector = pn.widgets.Select(
    name="View Mode", options=["Single View", "Overlay", "List"], value="Single View"
)

# Modify the layout to use the new selector instead of the checkbox
ss_styles = pn.WidgetBox("# Display Style", styles_group, view_selector).servable(
    target="sidebar"
)

# Create the "Reset Camera" button
# reset_camera_button = pn.widgets.Button(name="Reset Camera", button_type="primary")


def reset_camera(event):
    """
    Sets the camera to a specific view where the x-axis is pointed down and the
    y-axis into the screen.
    """
    global vtkpan, plotter

    camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]  # Example values
    plotter.camera_position = camera_position
    mess = f"Reset Camera: {plotter.camera_position}"
    _logger.info(mess)

    plotter.render()
    plotter.reset_camera()
    vtkpan = pn.pane.VTK(
        plotter.ren_win,
        margin=0,
        sizing_mode="stretch_both",
        orientation_widget=True,
        enable_keybindings=True,
        min_height=500,
    )

    vtkpan.object = plotter.ren_win


# Bind the reset_camera function to the button click event
# reset_camera_button.on_click(reset_camera)


# Adjust the update_single function to handle the different view options
def update_view(click):
    """
    Adjust the rendering style radio box depending on the state of the view mode selector.
    Returns:
        None
    """
    global styles_group

    _logger.info("Update View")

    selected_view = view_selector.value
    if selected_view == "Single View":
        styles_group.disabled = False
    elif selected_view == "Overlay":
        styles_group.disabled = True
    elif selected_view == "List":
        styles_group.disabled = False

    click_plot(click)


# Update references to `single_checkbox` to `view_selector` in callbacks
view_selector.param.watch(update_view, "value")

# markdown panels for various text outputs
title_md = pn.pane.Markdown("Title")
output_md = pn.pane.Markdown("Output goes here")
db_md = pn.pane.Markdown("Database Info goes here")

info_md = pn.pane.Markdown("SS Info")
ss_info = pn.WidgetBox("# Disulfide Info", info_md).servable(target="sidebar")

# default selections
ss_state_default = {
    "single": "True",
    "style": "sb",
    "rcsb_list": "['2q7q']",
    "rcsid": "2q7q",
    "defaultss": "2q7q_75D_140D",
    "ssid_list": "['2q7q_75D_140D', '2q7q_81D_113D', '2q7q_88D_171D', '2q7q_90D_138D', \
        '2q7q_91D_135D','2q7q_98D_129D']",
    "theme": "default",
}


def set_window_title():
    """
    Sets the window title using values from the global loader.
    """

    tot = PDB_SS.TotalDisulfides
    pdbs = len(PDB_SS.SSDict)
    vers = PDB_SS.version

    win_title = f"RCSB Disulfide Browser v{_vers}: {tot:,} Disulfides {pdbs:,} Structures {vers}"
    pn.state.template.param.update(title=win_title)

    mess = f"Set Window Title: {win_title}"
    _logger.debug(mess)

    return


def set_widgets_defaults():
    """
    Set the default values for the widgets and ensure the RCSB list is populated.

    This function sets the default values for the rendering style, single view checkbox,
    and RCSB selector widget. It also ensures that the RCSB list is correctly populated
    from the loaded data if it is not already populated.

    Globals:
        RCSB_list (list): A global list of RCSB IDs.
        PDB_SS (object): A global object containing the loaded PDB data.

    Returns:
        dict: The default state dictionary for the widgets.
    """
    global RCSB_list

    _logger.info("Setting widget defaults.")
    styles_group.value = "Split Bonds"
    single_checkbox.value = True

    # Ensure the RCSB list is correctly populated from loaded data
    if not RCSB_list:
        RCSB_list = sorted(
            PDB_SS.IDList
        )  # Load PDB IDs into RCSB_list if not populated

    rcsb_selector_widget.options = RCSB_list
    rcsb_selector_widget.value = ss_state_default["rcsid"]
    rcsb_ss_widget.value = ss_state_default["defaultss"]
    return ss_state_default


def set_state(event):
    """
    Set the ss_state dict to the state variables and UI interaface. Push to cache.
    """
    global ss_state

    _logger.info("Set state.")

    ss_state["rcsb_list"] = RCSB_list.copy()
    ss_state["rcsid"] = _rcsid_default
    ss_state["ssid_list"] = _ssidlist.copy()
    ss_state["single"] = single_checkbox.value
    ss_state["style"] = styles_group.value
    ss_state["defaultss"] = rcsb_ss_widget.value
    ss_state["theme"] = get_theme()

    pn.state.cache["ss_state"] = ss_state
    click_plot(None)


def plot(pl, ss, style="sb", light=True, panelsize=512) -> pv.Plotter:
    """
    Return the pyVista Plotter object for the Disulfide bond in the specific rendering style.

    :param single: Display the bond in a single panel in the specific style.
    :param style:  Rendering style: One of:
        * 'sb' - split bonds
        * 'bs' - ball and stick
        * 'cpk' - CPK style
        * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
        * 'plain' - boring single color
    :param light: If True, light background, if False, dark
    """
    global plotter, vtkpan

    _logger.debug("Entering plot: %s", ss.name)

    mode = view_selector.value

    if light:
        pv.set_plot_theme("document")
    else:
        pv.set_plot_theme("dark")

    if mode == "Single View":
        _logger.info("Single View")
        plotter = pv.Plotter(window_size=WINSIZE)
        plotter.clear()
        ss._render(plotter, style=style)
        # vtkpan.object = plotter.ren_win

    elif mode == "Overlay":
        _logger.info("Overlay")
        plotter = pv.Plotter(shape=(2, 2), window_size=WINSIZE)
        plotter.clear()

        plotter.subplot(0, 0)
        ss._render(plotter, style="cpk")

        plotter.subplot(0, 1)
        ss._render(plotter, style="bs")

        plotter.subplot(1, 0)
        ss._render(plotter, style="sb")

        plotter.subplot(1, 1)
        ss._render(plotter, style="pd")

        plotter.link_views()
        plotter.reset_camera()
        # vtkpan.object = plotter.ren_win
    else:
        _logger.info("List")
        pdbid = ss.pdb_id
        sslist = PDB_SS[pdbid]
        tot_ss = len(sslist)  # number off ssbonds
        rows, cols = grid_dimensions(tot_ss)
        winsize = (panelsize * cols, panelsize * rows)

        avg_enrg = sslist.average_energy
        avg_dist = sslist.average_distance
        resolution = sslist.average_resolution

        title = f"<{pdbid}> {resolution:.2f} Å: ({tot_ss} SS), Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"

        plotter = pv.Plotter(window_size=winsize, shape=(rows, cols))
        plotter.clear()
        plotter = sslist._render(plotter, style)
        plotter.enable_anti_aliasing("msaa")
        plotter.add_title(title=title, font_size=FONTSIZE)
        plotter.link_views()
        plotter.reset_camera()
        # vtkpan.object = plotter.ren_win

    plotter.reset_camera()
    return plotter


@pn.cache()
def load_data():
    """Load the RCSB Disulfide Database and return the object."""
    global RCSB_list, PDB_SS

    _logger.info("Loading RCSB Disulfide Database")

    PDB_SS = Load_PDB_SS(verbose=True, subset=False)

    RCSB_list = sorted(PDB_SS.IDList)

    message = f"Loaded RCSB Disulfide Database: {len(RCSB_list)} entries"
    _logger.info(message)

    set_window_title()
    pn.state.cache["data"] = PDB_SS
    return PDB_SS


if "data" in pn.state.cache:
    PDB_SS = pn.state.cache["data"]
else:
    PDB_SS = load_data()

set_window_title()
set_widgets_defaults()


def get_theme() -> str:
    """Return the current theme: 'default' or 'dark'

    Returns:
        str: The current theme
    """
    return pn.config.theme


def click_plot(event):
    """Force a re-render of the currently selected disulfide. Reuses the existing plotter."""

    # Reuse the existing plotter and re-render the scene
    global plotter

    plotter = render_ss()  # Reuse the existing plotter
    plotter.reset_camera()
    # Update the vtkpan and trigger a refresh
    vtkpan.object = plotter.ren_win
    # plotter.render()


def update_single(click):
    """
    Toggle the rendering style radio box depending on the state of the
    Single View checkbox.

    Returns:
        None
    """
    global styles_group

    single_checked = single_checkbox.value
    if single_checked is not True:
        styles_group.disabled = True
    else:
        styles_group.disabled = False

    click_plot(click)


# Callbacks
def get_ss_idlist(event) -> list:
    """Determine the list of disulfides for the given RCSB entry and
    update the RCSB_ss_widget appropriately.

    Returns:
        List of SS Ids
    """

    rcs_id = event.new
    sslist = DisulfideList([], "tmp")
    sslist = PDB_SS[rcs_id]
    idlist = []

    if sslist:
        idlist = [ss.name for ss in sslist]
        rcsb_ss_widget.options = idlist
        mess = f"get_ss_idlist |{rcs_id}| |{idlist}|"
        _logger.debug(mess)

    return idlist


def update_title(ss):
    """Update the title of the disulfide bond in the markdown pane."""

    name = ss.name

    title = f"## {name}"
    title_md.object = title


def update_info(ss):
    """Update the information of the disulfide bond in the markdown pane."""

    info_string = f"""
    ### {ss.name}
    **Resolution:** {ss.resolution:.2f} Å  
    **Energy:** {ss.energy:.2f} kcal/mol  
    **Cα distance:** {ss.ca_distance:.2f} Å  
    **Cβ distance:** {ss.cb_distance:.2f} Å  
    **Torsion Length:** {ss.torsion_length:.2f}°  
    **Rho:** {ss.rho:.2f}°  
    **Proximal Secondary:** {ss.proximal_secondary}  
    **Distal Secondary:** {ss.distal_secondary}
    """
    info_md.object = info_string


def update_output(ss):
    """Update the output of the disulfide bond in the markdown pane."""
    info_string = f"""
    **Cα-Cα:** {ss.ca_distance:.2f} Å  
    **Cβ-Cβ:** {ss.cb_distance:.2f} Å  
    **Torsion Length:** {ss.torsion_length:.2f}°  
    **Resolution:** {ss.resolution:.2f} Å  
    **Energy:** {ss.energy:.2f} kcal/mol
    """
    output_md.object = info_string


def get_ss(event) -> Disulfide:
    """Get the currently selected Disulfide"""
    ss_id = event.new
    ss = Disulfide(PDB_SS[ss_id])
    return ss


def get_ss_id(event):
    """Return the name of the currently selected Disulfide"""
    rcsb_ss_widget.value = event.new


def render_ss():
    """
    Render the currently selected disulfide with the current plotter.
    """
    global plotter

    light = True
    styles = {"Split Bonds": "sb", "CPK": "cpk", "Ball and Stick": "bs"}

    # Determine the theme
    theme = get_theme()
    if theme == "dark":
        light = False

    # Retrieve the selected disulfide
    ss_id = rcsb_ss_widget.value
    ss = PDB_SS[ss_id]

    if ss is None:
        update_output(f"Cannot find ss_id {ss_id}! Returning!")
        return

    # Update the UI
    update_title(ss)
    update_info(ss)
    update_output(ss)

    # Reuse and clear the existing plotter before rendering
    style = styles[styles_group.value]

    # Render the structure in the plotter
    return plot(plotter, ss, style=style, light=light)


def on_theme_change(event):
    """
    Callback function to handle theme changes.
    """
    global ss_state
    ss_state = pn.state.cache["ss_state"]
    new_theme = event.new
    ss_state["theme"] = new_theme
    pn.state.cache["ss_state"] = ss_state

    # Add your logic to handle the theme change here
    print(f"Theme changed to: {new_theme}")
    # Example: Update the plotter or other components based on the new theme
    if new_theme == "dark":
        plotter.set_background("black")
    else:
        plotter.set_background("white")
    plotter.render()
    vtkpan.object = plotter.ren_win  # Update the VTK pane object


def display_overlay(
    sslist,
    pl,
    verbose=False,
    light="Auto",
):
    """
    Display all disulfides in the list overlaid in stick mode against
    a common coordinate frames. This allows us to see all of the disulfides
    at one time in a single view. Colors vary smoothy between bonds.

    :param screenshot: Save a screenshot, defaults to False
    :param movie: Save a movie, defaults to False
    :param verbose: Verbosity, defaults to True
    :param fname: Filename to save for the movie or screenshot, defaults to 'ss_overlay.png'
    :param light: Background color, defaults to True for White. False for Dark.
    """

    ssbonds = sslist.data
    tot_ss = len(ssbonds)  # number off ssbonds

    res = 100

    if tot_ss > 100:
        res = 60
    if tot_ss > 200:
        res = 30
    if tot_ss > 300:
        res = 8

    pl.clear()
    pl.enable_anti_aliasing("msaa")
    pl.add_axes()

    mycol = np.zeros(shape=(tot_ss, 3))
    mycol = get_jet_colormap(tot_ss)

    # scale the overlay bond radii down so that we can see the individual elements better
    # maximum 90% reduction

    brad = BOND_RADIUS if tot_ss < 10 else BOND_RADIUS * 0.75
    brad = brad if tot_ss < 25 else brad * 0.8
    brad = brad if tot_ss < 50 else brad * 0.7
    brad = brad if tot_ss < 100 else brad * 0.5

    # print(f'Brad: {brad}')
    center_of_mass = sslist.center_of_mass
    # sslist.translate(center_of_mass)

    for i, ss in zip(range(tot_ss), ssbonds):
        color = [int(mycol[i][0]), int(mycol[i][1]), int(mycol[i][2])]
        ss._render(
            pl,
            style="plain",
            bondcolor=color,
            translate=False,
            bond_radius=brad,
            res=res,
        )

    pl.reset_camera()
    return pl


rcsb_selector_widget.param.watch(get_ss_idlist, "value")
rcsb_ss_widget.param.watch(set_state, "value")
styles_group.param.watch(set_state, "value")
single_checkbox.param.watch(update_single, "value")

plotter = pv.Plotter()
plotter = render_ss()

vtkpan = pn.pane.VTK(
    plotter.ren_win,
    margin=0,
    sizing_mode="stretch_both",
    orientation_widget=True,
    enable_keybindings=True,
    min_height=500,
)

pn.bind(get_ss_idlist, rcs_id=rcsb_selector_widget)
pn.bind(update_single, click=styles_group)

set_window_title()

render_win = pn.Column(vtkpan)
render_win.servable()

# end of file
