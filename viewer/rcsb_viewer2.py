"""
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 10/22/2024
"""

# pylint: disable=C0301 # line too long
# pylint: disable=C0413 # wrong import order
# pylint: disable=C0103 # wrong variable name
# pylint: disable=W0212 # access to a protected member _render of a client class
# pylint: disable=W0613 # unused argument
# pylint: disable=W0602 # Using the global statement
# pylint: disable=W0603 # Using the global statement


import logging
import os

import numpy as np
import panel as pn
import pyvista as pv

from proteusPy import (
    BOND_RADIUS,
    BS_SCALE,
    SPEC_POWER,
    SPECULARITY,
    Disulfide,
    DisulfideList,
    Load_PDB_SS,
    create_logger,
    get_jet_colormap,
)
from proteusPy.ProteusGlobals import WINSIZE

# Set PyVista to use offscreen rendering if the environment variable is set
if os.getenv("PYVISTA_OFF_SCREEN", "false").lower() == "true":
    pv.OFF_SCREEN = True

pn.extension("vtk", sizing_mode="stretch_width", template="fast")


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
# configure a master logger. This will create ~/logs/DBViewer.log
# configure_master_logger("DBViewer.log", log_level=logging.WARNING)

root_logger = logging.getLogger()
root_logger.handlers.clear()

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
multiview_checkbox = pn.widgets.Checkbox(name="Multi View", value=False)

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

ss_styles = pn.WidgetBox(
    "# Rendering Styles", styles_group, single_checkbox, multiview_checkbox
).servable(target="sidebar")

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
}


def set_window_title():
    """
    Sets the window title using values from the global loader.
    """

    tot = PDB_SS.TotalDisulfides
    pdbs = len(PDB_SS.SSDict)
    vers = PDB_SS.version

    win_title = f"RCSB Disulfide Browser: {tot:,} Disulfides {pdbs:,} Structures {vers}"
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
    # _rcsid_default, _ssidlist, _default_ss, single_checkbox, styles_group

    ss_state["rcsb_list"] = RCSB_list.copy()
    ss_state["rcsid"] = _rcsid_default
    ss_state["ssid_list"] = _ssidlist.copy()
    ss_state["single"] = single_checkbox.value
    ss_state["multiview"] = multiview_checkbox.value
    ss_state["style"] = styles_group.value
    ss_state["defaultss"] = rcsb_ss_widget.value
    _logger.debug("--> Set state.")
    pn.state.cache["ss_state"] = ss_state
    click_plot(None)


def load_state():
    """
    Load the state variables from the cache, update the interface.
    """
    # global _ssidlist, _rcsid, _style, _single, _default_ss
    _ss_state = {}

    if "ss_state" in pn.state.cache:
        _ss_state = pn.state.cache["ss_state"]
        _ssidlist = _ss_state["ssid_list"]
        _rcsid = _ss_state["rcsid"]
        _style = _ss_state["style"]
        _single = _ss_state["single"]
        _multiview = _ss_state["multiview"]
        _default_ss = _ss_state["defaultss"]

        styles_group.value = _style
        single_checkbox.value = _single
        multiview_checkbox.value = _multiview
        rcsb_selector_widget.value = _rcsid
        rcsb_ss_widget.value = _default_ss

    else:
        _logger.debug("Setting widgets.")
        set_widgets_defaults()

    return _ss_state


def set_camera_view(the_plotter):
    """
    Sets the camera to a specific view where the x-axis is pointed down and the
    y-axis into the screen.
    """
    # camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]  # Example values
    # plotter.camera_position = camera_position
    the_plotter.reset_camera()


def plot(pl, ss, single=True, style="sb", light=True) -> pv.Plotter:
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
    # src = ss.pdb_id
    # enrg = ss.energy
    # title = f"{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol. Cα: {ss.ca_distance:.2f} Å Cβ: {ss.cb_distance:.2f} Å Tors: {ss.torsion_length:.2f}°"

    if light:
        pv.set_plot_theme("document")
    else:
        pv.set_plot_theme("dark")

    # Create a new plotter with the desired shape
    _logger.info("Single view, creating new plotter")

    pl.enable_anti_aliasing("msaa")
    pl.clear()
    set_camera_view(pl)

    ss._render(
        pl,
        style=style,
        bs_scale=BS_SCALE,
        spec=SPECULARITY,
        specpow=SPEC_POWER,
    )

    pl.reset_camera()
    return pl


@pn.cache()
def load_data():
    """Load the disulfide database"""
    global RCSB_list, PDB_SS

    message = "Loading RCSB Disulfide Database"
    _logger.info(message)

    PDB_SS = Load_PDB_SS(verbose=True, subset=False, loadpath="/app/data")

    RCSB_list = sorted(PDB_SS.IDList)

    message = f"Loaded: {len(RCSB_list)}"
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
    """Returns the current theme: 'default' or 'dark'

    Returns:
        str: The current theme
    """
    args = pn.state.session_args
    if "theme" in args and args["theme"][0] == b"dark":
        return "dark"
    return "default"


def click_plot(event):
    """Force a re-render of the currently selected disulfide. Reuses the existing plotter."""

    # Reuse the existing plotter and re-render the scene
    global plotter

    plotter = render_ss()  # Reuse the existing plotter

    # Update the vtkpan and trigger a refresh
    vtkpan.object = plotter.ren_win
    vtkpan.param.trigger("object")
    plotter.render()


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


def update_multiview(click):
    """
    Toggle the rendering style radio box depending on the state of the
    Single View checkbox.

    Returns:
        None
    """
    global styles_group

    multiview_checked = multiview_checkbox.value
    if multiview_checked is not True:
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
        mess = f"--> get_ss_idlist |{rcs_id}| |{idlist}|"
        _logger.debug(mess)
    return idlist


def update_title(ss):
    """Update the title of the disulfide bond in the markdown pane."""

    name = ss.name

    title = f"## {name}"
    title_md.object = title


def update_info(ss):
    """Update the information of the disulfide bond in the markdown pane."""
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution
    prox_sec = ss.proximal_secondary  # Assuming these attributes exist
    dist_sec = ss.distal_secondary  # Assuming these attributes exist

    info_string = f"""
    ### {name}
    **Resolution:** {resolution:.2f} Å  
    **Energy:** {enrg:.2f} kcal/mol  
    **Cα distance:** {ss.ca_distance:.2f} Å  
    **Cβ distance:** {ss.cb_distance:.2f} Å  
    **Torsion Length:** {ss.torsion_length:.2f}°  
    **Proximal Secondary:** {prox_sec}  
    **Distal Secondary:** {dist_sec}
    """
    info_md.object = info_string


def update_output(ss):
    """Update the disulfide bond info in the markdown pane."""
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
    global PDB_SS
    ss_id = event.new
    ss = Disulfide(PDB_SS[ss_id])
    return ss


def get_ss_id(event):
    """Return the name of the currently selected Disulfide"""
    rcsb_ss_widget.value = event.new


def display_overlay(
    sslist,
    pl,
    screenshot=False,
    movie=False,
    verbose=False,
    fname="ss_overlay.png",
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

    # from proteusPy.utility import get_theme

    pid = sslist.pdb_id

    ssbonds = sslist.data
    tot_ss = len(ssbonds)  # number off ssbonds

    res = 100

    if tot_ss > 100:
        res = 60
    if tot_ss > 200:
        res = 30
    if tot_ss > 300:
        res = 8

    # title = f"<{pid}> {resolution:.2f} Å: ({tot_ss} SS), Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"

    if light == "light":
        pv.set_plot_theme("document")
    elif light == "dark":
        pv.set_plot_theme("dark")
    else:
        _theme = get_theme()
        if _theme == "light":
            pv.set_plot_theme("document")
        elif _theme == "dark":
            pv.set_plot_theme("dark")
            _logger.info("Dark mode detected.")
        else:
            pv.set_plot_theme("document")

    if movie:
        pl = pv.Plotter(window_size=WINSIZE, off_screen=True)
    else:
        pl = pv.Plotter(window_size=WINSIZE, off_screen=False)

    pl.clear()
    # pl.add_title(title=title, font_size=FONTSIZE)
    pl.enable_anti_aliasing("msaa")
    pl.add_axes()

    mycol = np.zeros(shape=(tot_ss, 3))
    mycol = get_jet_colormap(tot_ss)

    # scale the overlay bond radii down so that we can see the individual elements better
    # maximum 90% reduction

    brad = BOND_RADIUS if tot_ss < 10 else BOND_RADIUS * 0.75
    brad = brad if tot_ss < 25 else brad * 0.8
    brad = brad if tot_ss < 50 else brad * 0.8
    brad = brad if tot_ss < 100 else brad * 0.6

    # print(f'Brad: {brad}')

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

    if screenshot:
        pl.show(auto_close=False)  # allows for manipulation
        pl.screenshot(fname)

    elif movie:
        if verbose:
            print(f" -> display_overlay(): Saving mp4 animation to: {fname}")

        pl.open_movie(fname)
        path = pl.generate_orbital_path(n_points=360)
        pl.orbit_on_path(path, write_frames=True)
        pl.close()

        if verbose:
            print(f" -> display_overlay(): Saved mp4 animation to: {fname}")

    pl.show()
    return pl


def plot_multiview(pl, ss_id, style="sb", light=True) -> pv.Plotter:
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
    # src = ss.pdb_id
    # enrg = ss.energy
    # title = f"{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol. Cα: {ss.ca_distance:.2f} Å Cβ: {ss.cb_distance:.2f} Å Tors: {ss.torsion_length:.2f}°"

    if light:
        pv.set_plot_theme("document")
    else:
        pv.set_plot_theme("dark")

    # Create a new plotter with the desired shape
    _logger.info("Multi view, creating new plotter")

    pl.enable_anti_aliasing("msaa")
    pl.clear()
    set_camera_view(pl)

    sslist = PDB_SS[ss_id]
    _logger.info(f"Rendering {ss_id} disulfides with {len(sslist)} bonds.")

    pl = display_overlay(
        sslist, pl, screenshot=False, movie=False, verbose=False, light="Auto"
    )
    pl.reset_camera()
    return pl


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
    pdb_id = rcsb_selector_widget.value
    ss = PDB_SS[ss_id]

    if ss is None:
        update_output(f"Cannot find ss_id {ss_id}! Returning!")
        return

    # Update the UI
    update_title(ss)
    update_info(ss)
    update_output(ss)

    # Reuse and clear the existing plotter before rendering
    # plotter.clear()
    style = styles[styles_group.value]
    single = single_checkbox.value

    if single:
        # Render the structure in the plotter
        return plot(plotter, ss, single=single, style=style, light=light)

    return plot_multiview(plotter, pdb_id, style=style, light=light)


def on_theme_change(event):
    """Handle a theme change event."""
    selected_theme = event.obj.theme
    message = f"--> Theme Change: {selected_theme}"
    _logger.debug(message)


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