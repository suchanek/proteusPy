"""
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 1/14/2024
"""

import sys
import time

import panel as pn
import pyvista as pv

from proteusPy.atoms import BOND_COLOR, BS_SCALE, SPEC_POWER, SPECULARITY
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideList import DisulfideList
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.ProteusGlobals import *

pn.extension("vtk", sizing_mode="stretch_width", template="fast")

_vers = 0.6

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
orientation_widget = True
enable_keybindings = True

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

ss_styles = pn.WidgetBox("# Rendering Styles", styles_group, single_checkbox).servable(
    target="sidebar"
)

# markdown panels for various text outputs
title_md = pn.pane.Markdown("Title")
output_md = pn.pane.Markdown("Output goes here")
db_md = pn.pane.Markdown("Database Info goes here")

info_md = pn.pane.Markdown("SS Info")
ss_info = pn.WidgetBox("# Disulfide Info", info_md).servable(target="sidebar")
db_info = pn.Column("### RCSB Database Info", db_md)

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
    Sets the window title using values from the ss_state dictionary.
    """

    global PDB_SS

    tot = PDB_SS.TotalDisulfides
    pdbs = len(PDB_SS.SSDict)
    vers = PDB_SS.version

    pn.state.template.param.update(
        title=f"RCSB Disulfide Browser: {tot} Disulfides, {pdbs} Structures, {vers}"
    )
    print(f"--> Set Window Title: {tot:,} Disulfides, {pdbs:,} Structures, {vers}")
    return


def set_widgets_defaults():
    global RCSB_list

    styles_group.value = "Split Bonds"
    single_checkbox.value = True

    # Ensure the RCSB list is correctly populated from loaded data
    if not RCSB_list:
        RCSB_list = sorted(
            PDB_SS.IDList
        )  # Load PDB IDs into RCSB_list if not populated

    rcsb_selector_widget.options = RCSB_list
    rcsb_selector_widget.value = _rcsid_default
    rcsb_ss_widget.value = _default_ss
    return ss_state_default


def set_state(event):
    """
    Set the ss_state dict to the state variables and UI interaface. Push to cache.
    """
    global ss_state, _rcsid_default, _ssidlist, _default_ss, single_checkbox, styles_group

    ss_state["rcsb_list"] = RCSB_list.copy()
    ss_state["rcsid"] = _rcsid_default
    ss_state["ssid_list"] = _ssidlist.copy()
    ss_state["single"] = single_checkbox.value
    ss_state["style"] = styles_group.value
    ss_state["defaultss"] = rcsb_ss_widget.value
    print("--> Set state.")
    # print_state(ss_state)
    pn.state.cache["ss_state"] = ss_state
    set_window_title()
    click_plot(None)


def print_state(state):
    print(f"--> State: {state}")
    return


def load_state():
    """
    Load the state variables from the cache, update the interface.
    """
    global _ssidlist, _rcsid, _style, _single, _ssbond, _default_ss, _boot, single_checkbox, styles_group
    _ss_state = {}

    if "ss_state" in pn.state.cache:
        _ss_state = pn.state.cache["ss_state"]
        _ssidlist = _ss_state["ssid_list"]
        _rcsid = _ss_state["rcsid"]
        _style = _ss_state["style"]
        _single = _ss_state["single"]
        _default_ss = _ss_state["defaultss"]

        styles_group.value = _style
        single_checkbox.value = _single
        rcsb_selector_widget.value = _rcsid
        rcsb_ss_widget.value = _default_ss

    else:
        print("--> setting widgets.")
        set_widgets_defaults()
        print_state(_ss_state)

    return _ss_state


def set_camera_view(plotter):
    """
    Sets the camera to a specific view where the x-axis is pointed down and the
    y-axis into the screen.
    """
    camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]  # Example values
    # plotter.camera_position = camera_position
    plotter.reset_camera()


def plot(pl, ss, single=True, style="sb", light=True, shadows=False) -> pv.Plotter:
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
    src = ss.pdb_id
    enrg = ss.energy

    title = f"{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol. Cα: {ss.ca_distance:.2f} Å Cβ: {ss.cb_distance:.2f} Å Tors: {ss.torsion_length:.2f}°"

    if light:
        pv.set_plot_theme("document")
    else:
        pv.set_plot_theme("dark")

    # Create a new plotter with the desired shape
    if single:
        new_pl = pv.Plotter(window_size=WINSIZE)
        # new_pl = pl

    else:
        new_pl = pv.Plotter(shape=(2, 2), window_size=WINSIZE)

    new_pl.enable_anti_aliasing("msaa")
    new_pl.clear()
    set_camera_view(new_pl)

    if single is True:
        ss._render(
            new_pl,
            style=style,
            bs_scale=BS_SCALE,
            spec=SPECULARITY,
            specpow=SPEC_POWER,
        )
        # new_pl.reset_camera()
    else:
        new_pl.subplot(0, 0)

        ss._render(
            new_pl,
            style="cpk",
            bondcolor=BOND_COLOR,
            bs_scale=BS_SCALE,
            spec=SPECULARITY,
            specpow=SPEC_POWER,
        )

        new_pl.subplot(0, 1)

        ss._render(
            new_pl,
            style="bs",
            bondcolor=BOND_COLOR,
            bs_scale=BS_SCALE,
            spec=SPECULARITY,
            specpow=SPEC_POWER,
        )

        new_pl.subplot(1, 0)

        ss._render(
            new_pl,
            style="sb",
            bondcolor=BOND_COLOR,
            bs_scale=BS_SCALE,
            spec=SPECULARITY,
            specpow=SPEC_POWER,
        )

        new_pl.subplot(1, 1)
        ss._render(
            pl,
            style="pd",
            bondcolor=BOND_COLOR,
            bs_scale=BS_SCALE,
            spec=SPECULARITY,
            specpow=SPEC_POWER,
        )

        new_pl.link_views()
    new_pl.reset_camera()
    return new_pl


@pn.cache()
def load_data():
    global vers, tot, pdbs, RCSB_list

    loader = Load_PDB_SS(verbose=True, subset=False)  # Load some data
    
    RCSB_list = sorted(loader.IDList)
    print(f"--> Load Data: {len(RCSB_list)}")
    # set_state(event=None)
    set_window_title()
    
    pn.state.cache["data"] = loader
    return loader


if "data" in pn.state.cache:
    PDB_SS = pn.state.cache["data"]
else:
    PDB_SS = load_data()

set_window_title()
set_widgets_defaults()

# PDB_SS = pn.state.as_cached("data", load_data)
# ss_state = load_state()
# print(f'--> found state: {ss_state}')


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
    global plotter

    # Reuse the existing plotter and re-render the scene
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
    global plotter, single_checkbox, styles_group
    single_checked = single_checkbox.value
    if single_checked is not True:
        styles_group.disabled = True
    else:
        styles_group.disabled = False
    ## plotter = pv.Plotter()
    click_plot(click)


# Callbacks
def get_ss_idlist(event) -> list:
    """Determine the list of disulfides for the given RCSB entry and
    update the RCSB_ss_widget appropriately.

    Returns:
        List of SS Ids
    """
    global PDB_SS

    rcs_id = event.new
    print(rcs_id)
    sslist = DisulfideList([], "tmp")
    sslist = PDB_SS[rcs_id]
    idlist = []

    if sslist:
        idlist = [ss.name for ss in sslist]
        rcsb_ss_widget.options = idlist
    print(f"--> get_ss_idlist |{rcs_id}| |{idlist}|")
    return idlist


def update_title(ss):
    src = ss.pdb_id
    name = ss.name

    title = f"## {name}"
    title_md.object = title


def update_info(ss):
    src = ss.pdb_id
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
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution

    info_string = f"""
    **Cα-Cα:** {ss.ca_distance:.2f} Å  
    **Cβ-Cβ:** {ss.cb_distance:.2f} Å  
    **Torsion Length:** {ss.torsion_length:.2f}°  
    **Resolution:** {resolution:.2f} Å  
    **Energy:** {enrg:.2f} kcal/mol
    """
    output_md.object = info_string


def get_ss(event) -> Disulfide:
    global PDB_SS
    ss_id = event.new
    ss = Disulfide(PDB_SS[ss_id])
    return ss


def get_ss_id(event):
    rcsb_ss_widget.value = event.new


def render_ss():
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
    # plotter.clear()
    style = styles[styles_group.value]
    single = single_checkbox.value

    # Render the structure in the plotter
    return plot(plotter, ss, single=single, style=style, shadows=False, light=light)


def on_theme_change(event):
    selected_theme = event.obj.theme
    print(f"--> Theme Change: {selected_theme}")


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
    orientation_widget=orientation_widget,
    enable_keybindings=enable_keybindings,
    min_height=500,
)

pn.bind(get_ss_idlist, rcs_id=rcsb_selector_widget)
pn.bind(update_single, click=styles_group)

pn.state.template.param.update(
    title=f"RCSB Disulfide Browser: {PDB_SS.TotalDisulfides} Disulfides, {len(PDB_SS.SSDict)} Structures, {PDB_SS.version}"
)

render_win = pn.Column(vtkpan)
render_win.servable()

# end of file
