"""
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 1/19/2024
"""

import pyvista as pv
import panel as pn

pn.extension("vtk", sizing_mode="stretch_width", template="fast")

from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS

_vers = 0.8
_default_rcsid = "2q7q"
_default_ss = "2q7q_75D_140D"
_default_style = "Ball and Stick"
_default_single = True
_ssidlist = [
    "2q7q_75D_140D",
    "2q7q_81D_113D",
    "2q7q_88D_171D",
    "2q7q_90D_138D",
    "2q7q_91D_135D",
    "2q7q_98D_129D",
    "2q7q_130D_161D",
]

# globals
ss_state = {}
tot = pdbs = 0
dbvers = 0
render_style = _default_style
single_style = _default_single
rcsid = _default_rcsid
ssidlist = _ssidlist
default_ss = _default_ss
RCSB_list = []

orientation_widget = True
enable_keybindings = True


def get_theme() -> str:
    """Return the current theme: 'default' or 'dark'

    Returns:
        str: The current theme
    """
    args = pn.state.session_args
    if "theme" in args and args["theme"][0] == b"dark":
        return "dark"
    return "default"


def click_plot(event):
    """Force a re-render of the currently selected disulfide. Removes the pane
    and re-adds it to the panel.

    Returns:
        None
    """
    global render_win
    # vtkpan
    cnt = event.new

    plotter = render_ss()
    vtkpan = pn.pane.VTK(
        plotter.ren_win,
        margin=0,
        sizing_mode="stretch_both",
        orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings,
        min_height=500,
    )

    vtkpan.param.trigger("object")
    render_win[0] = vtkpan
    update_output(f"{render_win[0]}, {cnt}")
    ss = get_ss()


def update_info_cb(event):
    ss_id = event.new
    update_output(f"Got id: {ss_id}")
    ss = get_ss()
    update_info(ss)


def update_info(ss: Disulfide):
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution
    # info_string = ss.pprint()
    info_string = f"### {name}  \n**Resolution:** {resolution:.2f} Å  \n**Energy:** {enrg:.2f} kcal/mol  \n**Cα distance:** {ss.ca_distance:.2f} Å  \n**Cβ distance:** {ss.cb_distance:.2f} Å  \n**Torsion Length:** {ss.torsion_length:.2f}°"
    info_md.object = info_string


# Widgets
styles_group = pn.widgets.RadioBoxGroup(
    name="Rending Style",
    options=["Split Bonds", "CPK", "Ball and Stick"],
    inline=False,
)

single_checkbox = pn.widgets.Checkbox(name="Single View", value=True)


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


button = pn.widgets.Button(name="Refresh", button_type="primary")
button.on_click(click_plot)

# not used atm
shadows_checkbox = pn.widgets.Checkbox(name="Shadows", value=False)

rcsb_ss_widget = pn.widgets.Select(
    name="Disulfide", value=default_ss, options=ssidlist
)
rcsb_selector_widget = pn.widgets.AutocompleteInput(
    name="RCSB ID",
    value=rcsid,
    restrict=True,
    placeholder="Search Here",
    options=RCSB_list,
)

# controls on sidebar
ss_props = pn.WidgetBox(
    "# Disulfide Selection", rcsb_selector_widget, rcsb_ss_widget
).servable(target="sidebar")

ss_styles = pn.WidgetBox(
    "# Rendering Styles", styles_group, single_checkbox, button
).servable(target="sidebar")

# markdown panels for various text outputs
title_md = pn.pane.Markdown("Title")
output_md = pn.pane.Markdown("Output goes here")
db_md = pn.pane.Markdown("Database Info goes here")
info_md = pn.pane.Markdown("SS Info")

ss_info = pn.WidgetBox("# Disulfide Info \n ", info_md).servable(
    target="sidebar"
)
output_info = pn.WidgetBox("## Program Output\n ", output_md).servable(
    target="sidebar"
)
db_info = pn.Column("### RCSB Database Info", db_md)


def update_output(info_string: str):
    output_md.object = info_string


def load_data():
    global dbvers, tot, pdbs, RCSB_list, _boot

    print(f"Uncachced, loading first time")
    PDB_SS = Load_PDB_SS(verbose=True, subset=False)  # Load some data
    dbvers = PDB_SS.version
    tot = PDB_SS.TotalDisulfides
    pdbs = len(PDB_SS.SSDict)
    RCSB_list = sorted(PDB_SS.IDList)
    pn.state.cache["data"] = PDB_SS
    pn.state.template.param.update(
        title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{_vers}"
    )
    rcsb_selector_widget.options = RCSB_list
    return PDB_SS


# Callbacks
def get_ss_idlist(id) -> list:
    """Determine the list of disulfides for the given RCSB entry and update the RCSB_ss_widget
    appropriately.

    Returns:
        List of SS Ids
    """
    global PDB_SS
    idlist = []

    rcs_id = id.new
    update_output(f"--> rcs_id: {rcs_id}")
    if rcs_id is not None:
        # sslist = DisulfideList([],'tmp')
        sslist = PDB_SS[rcs_id]
        update_output(f"--> get_ss_idlist {rcs_id}, {sslist}")
    else:
        update_output(f"Cannot get ss_idlist for id {rcs_id}?")

    idlist = [ss.name for ss in sslist]
    # update_output(f'--> get_ss_idlist |{idlist}|')
    rcsb_ss_widget.options = idlist
    return idlist


def update_title(ss: Disulfide):
    src = ss.pdb_id
    name = ss.name

    title = f"## {name}"
    title_md.object = title


def get_ss() -> Disulfide:
    global PDB_SS
    ss_id = rcsb_ss_widget.value
    ss = PDB_SS[ss_id]
    if ss is None:
        update_output(f"Cannot find ss_id {ss_id}! Returning!")
        return None
    else:
        return ss


def render_ss(clk=True):
    global PDB_SS
    global plotter
    global vtkpan
    global render_win

    light = True

    styles = {"Split Bonds": "sb", "CPK": "cpk", "Ball and Stick": "bs"}

    theme = get_theme()
    if theme == "dark":
        light = False

    ss_id = rcsb_ss_widget.value
    ss = PDB_SS[ss_id]
    if ss is not None:
        update_title(ss)
        update_info(ss)
    else:
        update_output(
            "# Error: \nClick_plot can't find a disulfide for the current UI?"
        )
        return

    style = styles[styles_group.value]
    single = single_checkbox.value
    shadows = shadows_checkbox.value

    plotter.clear()
    plotter = ss.plot(
        plotter, single=single, style=style, shadows=shadows, light=light
    )

    vtkpan = pn.pane.VTK(
        plotter.ren_win,
        margin=0,
        sizing_mode="stretch_both",
        orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings,
        min_height=500,
    )

    ### vtkpan.param.trigger('object')

    return plotter


"""
from bokeh.plotting import figure

def app():
    p = figure()
    p.line([1, 2, 3], [1, 2, 3])
    return p

pn.serve(app, threaded=True)

pn.state.execute(lambda: p.y_range.update(start=0, end=4))
"""

# Main program
if "data" in pn.state.cache:
    PDB_SS = pn.state.cache["data"]
    RCSB_list = sorted(PDB_SS.IDList)
    dbvers = PDB_SS.version
    tot = PDB_SS.TotalDisulfides
    pdbs = len(PDB_SS.SSDict)
    rcsb_selector_widget.options = RCSB_list
    pn.state.template.param.update(
        title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{_vers}"
    )
    print(f"--> Data cached: {len(RCSB_list)}")
else:
    PDB_SS = load_data()
    RCSB_list = sorted(PDB_SS.IDList)
    pn.state.cache["data"] = PDB_SS
    dbvers = PDB_SS.version
    tot = PDB_SS.TotalDisulfides
    pdbs = len(PDB_SS.SSDict)
    pn.state.template.param.update(
        title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{_vers}"
    )
    print(f"--> Data not cached: {len(RCSB_list)}")

# PDB_SS = Load_PDB_SS(verbose=True, subset=False)
print(f"Loaded version: {dbvers}")

current_ss = get_ss()
update_info(current_ss)

plotter = pv.Plotter()
# plotter = render_ss()

vtkpan = pn.pane.VTK(
    plotter.ren_win,
    margin=0,
    sizing_mode="stretch_both",
    orientation_widget=orientation_widget,
    enable_keybindings=enable_keybindings,
    min_height=500,
)

vtkpan.param.trigger("object")
render_win = pn.Column(vtkpan)

plotter = render_ss()

rcsb_selector_widget.param.watch(get_ss_idlist, "value")
single_checkbox.param.watch(update_single, "value")
rcsb_ss_widget.param.watch(update_info_cb, "value")

# styles_group.param.watch(click_plot, 'value')

pn.bind(get_ss_idlist, id=rcsb_selector_widget)
pn.bind(update_single, click=styles_group)
pn.bind(
    update_info,
    rcsb_ss_widget,
)

update_output(f"{render_win}")

render_win.servable()


"""
pn.Column(
    "This example demonstrates the use of **VTK and pyvista** to display a *scene*",
    pn.Row(
        render_win
    ), min_height=500
)
"""
