import darkdetect  # Optional: Install via `pip install darkdetect`
import numpy as np
import panel as pn
import pyvista as pv

from proteusPy import (
    BOND_RADIUS,
    DisulfideList,
    Load_PDB_SS,
    get_jet_colormap,
    get_theme,
    grid_dimensions,
)
from proteusPy.ProteusGlobals import DATA_DIR

pn.extension("vtk")


class DisulfideViewerPanel:
    def __init__(self, ss_list):
        # Define color dictionary
        self.colors = {
            "dark": {
                "background": "#2E2E2E",  # Dark Gray
                "midnight_blue": "#191970",
                "dark_slate_gray": "#2F4F4F",
                "charcoal": "#36454F",
                "deep_space_sparkle": "#4A646C",
                "black": "black",
            },
            "light": {
                "background": "#FFFFFF",  # White
                "light_gray": "#D3D3D3",
                "ivory": "#FFFFF0",
                "honeydew": "#F0FFF0",
                "azure": "#F0FFFF",
            },
            "auto": {  # Define "auto" theme
                "background": "#FFFFFF",  # Default to light; will be dynamically set
                # Add other colors if necessary
            },
        }

        self.current_theme = "auto"  # Set default theme to "auto"
        self.ss_list = ss_list
        if not self.ss_list:
            raise ValueError("ss_list is empty.")
        self.current_ss = self.ss_list[0]
        self.current_pdb_id = self.current_ss.pdb_id
        self.current_style = "sb"  # Initialize with default style
        self.single = True

        # Create a dictionary where keys are pdb_id and values are lists of disulfides
        self.ss_dict = {}
        for ss in self.ss_list:
            if ss.pdb_id not in self.ss_dict:
                self.ss_dict[ss.pdb_id] = []
            self.ss_dict[ss.pdb_id].append(ss)

        # Initialize Panel widgets
        self.init_widgets()

        # Initialize PyVista plotter
        self.init_plotter()

        # Layout the application
        self.layout = self.create_layout()

        # Initial display
        self.display()

    def init_widgets(self):
        # Rendering style buttons
        self.button_cpk = pn.widgets.Button(name="CPK", button_type="primary")
        self.button_sb = pn.widgets.Button(name="SB", button_type="primary")
        self.button_bs = pn.widgets.Button(name="BS", button_type="primary")
        self.button_pd = pn.widgets.Button(name="PD", button_type="primary")

        # Connect buttons to callbacks
        self.button_cpk.on_click(lambda event: self.update_style("cpk"))
        self.button_sb.on_click(lambda event: self.update_style("sb"))
        self.button_bs.on_click(lambda event: self.update_style("bs"))
        self.button_pd.on_click(lambda event: self.update_style("pd"))

        # PDB ID text input
        self.pdb_textbox = pn.widgets.TextInput(
            name="PDB ID", placeholder="Enter PDB ID"
        )
        self.pdb_textbox.param.watch(self.on_pdb_textbox_enter, "value")

        # PDB ID dropdown
        self.pdb_dropdown = pn.widgets.Select(
            name="Select PDB ID", options=list(self.ss_dict.keys())
        )
        self.pdb_dropdown.param.watch(self.on_pdb_dropdown_change, "value")

        # Disulfide dropdown
        self.dropdown = pn.widgets.Select(
            name="Select Disulfide", options=[ss.name for ss in self.ss_list]
        )
        self.dropdown.param.watch(self.on_dropdown_change, "value")

        # Theme toggle button
        self.button_theme = pn.widgets.Toggle(
            name="Toggle Theme", button_type="default"
        )
        self.button_theme.param.watch(self.toggle_theme_state, "value")

        # Single/multiple display checkbox
        self.checkbox_single = pn.widgets.Checkbox(name="Single Display", value=True)
        self.checkbox_single.param.watch(self.on_checkbox_single_change, "value")

        # Reset camera button
        self.button_reset = pn.widgets.Button(
            name="Reset Camera", button_type="warning"
        )
        self.button_reset.on_click(lambda event: self.set_camera_view())

        # Spin camera button
        self.button_spin = pn.widgets.Button(name="Spin Camera", button_type="success")
        self.button_spin.on_click(lambda event: self.spin_camera())

        # Camera control sliders
        self.slider_x = pn.widgets.IntSlider(
            name="Camera X", start=-100, end=100, step=1, value=0
        )
        self.slider_y = pn.widgets.IntSlider(
            name="Camera Y", start=-100, end=100, step=1, value=0
        )

        self.slider_x.param.watch(self.update_camera_position, "value")
        self.slider_y.param.watch(self.update_camera_position, "value")

        # Status bar
        self.status_bar = pn.widgets.StaticText(
            value=f"Displaying: {self.current_ss.name}"
        )

    def init_plotter(self):
        # Initialize PyVista Plotter
        self.plotter = pv.Plotter(window_size=(800, 600))

        # Add a test mesh (sphere) to verify rendering
        self.plotter.add_mesh(pv.Sphere(), color="red")
        print("Added test sphere to plotter.")

        self.plotter_widget = pn.pane.VTK(self.plotter, sizing_mode="stretch_both")

    def create_layout(self):
        # Group PDB controls
        pdb_controls = pn.Column(
            self.pdb_textbox,
            self.pdb_dropdown,
            self.dropdown,
            width=300,
            margin=(10, 10),
        )

        # Group rendering style buttons
        rendering_styles = pn.Column(
            self.button_cpk,
            self.button_sb,
            self.button_bs,
            self.button_pd,
            self.checkbox_single,
            self.button_spin,
            width=150,
            margin=(10, 10),
        )

        # Controls column
        controls = pn.Column(
            pn.pane.Markdown("### Rendering Styles"),
            rendering_styles,
            pn.pane.Markdown("### PDB Controls"),
            pdb_controls,
            pn.Row(self.button_theme, self.button_reset),
            width=320,
            # background=self.colors[self.current_theme]["background"],
            sizing_mode="fixed",
            margin=(10, 10),
        )

        # Camera sliders
        camera_controls = pn.Row(
            self.slider_x, self.slider_y, sizing_mode="stretch_width"
        )

        # Main layout
        main_layout = pn.Row(
            controls,
            self.plotter_widget,
            pn.Column(camera_controls, sizing_mode="stretch_height"),
            sizing_mode="stretch_both",
        )

        # Menu bar
        menu = self.create_menu()

        # Status bar at the bottom
        status_bar = pn.Row(self.status_bar, sizing_mode="stretch_width")

        # Combine menu, main layout, and status bar
        full_layout = pn.Column(
            menu, main_layout, status_bar, sizing_mode="stretch_both"
        )

        return full_layout

    def create_menu(self):
        # File menu buttons
        save_screenshot = pn.widgets.Button(
            name="Save Screenshot", button_type="primary"
        )
        export_scene = pn.widgets.Button(name="Export Scene", button_type="primary")
        reset = pn.widgets.Button(name="Reset", button_type="warning")

        # Connect to callbacks
        save_screenshot.on_click(self.save_screenshot)
        export_scene.on_click(self.export_scene)
        reset.on_click(lambda event: self.reset())

        # Arrange in a row
        menu = pn.Row(
            save_screenshot,
            export_scene,
            reset,
            # background="lightgray",
            sizing_mode="stretch_width",
        )

        return menu

    def update_style(self, style):
        print(f"Updating style to: {style}")
        self.current_style = style
        self.checkbox_single.value = True  # Ensure "Single" is checked
        self.display()

    def on_pdb_textbox_enter(self, event):
        pdb_id = event.new.strip()
        print(f"PDB ID entered: {pdb_id}")
        if pdb_id in self.ss_dict:
            self.update_pdb_selection(pdb_id)
            self.status_bar.value = f"Displaying: {pdb_id}"
            print(f"PDB ID '{pdb_id}' loaded successfully.")
        else:
            pn.state.notifications.error(f"The entered PDB ID '{pdb_id}' is not valid.")
            print(f"Invalid PDB ID entered: {pdb_id}")
        disulfides = self.ss_dict.get(pdb_id, [])

        self.dropdown.options = [ss.name for ss in disulfides]
        self.current_sslist = DisulfideList(list(disulfides), "sublist")
        self.pdb_dropdown.value = pdb_id

    def on_pdb_dropdown_change(self, event):
        pdb_id = event.new
        print(f"PDB dropdown changed to: {pdb_id}")
        disulfides = self.ss_dict.get(pdb_id, [])

        self.dropdown.options = [ss.name for ss in disulfides]
        if disulfides:
            self.current_ss = disulfides[0]
            self.current_sslist = DisulfideList(list(disulfides), "sublist")
            self.status_bar.value = f"Displaying: {self.current_ss.name}"
            self.display()
            print(f"Displaying disulfide: {self.current_ss.name}")
        self.pdb_textbox.value = pdb_id

    def update_pdb_selection(self, pdb_id):
        print(f"Updating PDB selection to: {pdb_id}")
        self.current_pdb_id = pdb_id
        disulfides = self.ss_dict.get(pdb_id, [])

        self.current_sslist = DisulfideList(list(disulfides), "sublist2")
        self.dropdown.options = [ss.name for ss in self.ss_list]

    def on_dropdown_change(self, event):
        ss_name = event.new
        print(f"Disulfide dropdown changed to: {ss_name}")
        # Find the disulfide with the given name
        selected_ss = next((ss for ss in self.ss_list if ss.name == ss_name), None)
        if selected_ss:
            self.current_ss = selected_ss
            self.status_bar.value = f"Displaying: {self.current_ss.name}"
            self.display()
            print(f"Displaying disulfide: {self.current_ss.name}")

    def on_checkbox_single_change(self, event):
        self.single = event.new
        print(f"Single Display set to: {self.single}")
        self.display()

    def update_camera_position(self, event):
        x = self.slider_x.value
        y = self.slider_y.value
        print(f"Updating camera position to: x={x}, y={y}")

        self.plotter.camera_position = [(x, y, 10), (0, 0, 0), (0, 1, 0)]
        self.plotter.render()

    def toggle_theme_callback(self):
        # Toggle between "light" and "dark" if not "auto"
        if self.current_theme == "light":
            self.current_theme = "dark"
        elif self.current_theme == "dark":
            self.current_theme = "light"
        else:  # self.current_theme == "auto"
            # Optionally, toggle between system theme and manual selection
            detected_theme = "dark" if darkdetect.isDark() else "light"
            self.current_theme = detected_theme
        print(f"Toggling theme to: {self.current_theme}")
        self.apply_theme()

    def toggle_theme_state(self, event):
        self.toggle_theme_callback()

    def apply_theme(self):
        if self.current_theme == "auto":
            detected_theme = "dark" if darkdetect.isDark() else "light"
            background_color = self.colors[detected_theme]["background"]
            print(f"Auto-detected theme: {detected_theme}")
        else:
            background_color = self.colors[self.current_theme]["background"]
            print(f"Applying theme: {self.current_theme}")

        self.plotter.set_background(background_color)
        self.layout[1].background = background_color  # Update controls background
        self.plotter.render()

    def set_camera_view(self):
        print("Resetting camera view.")
        camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]
        self.plotter.camera_position = camera_position
        self.plotter.reset_camera()
        self.plotter.render()

    def spin_camera(self, duration=5, n_frames=150):
        print("Spinning camera.")
        theta = np.linspace(0, 2 * np.pi, n_frames)
        path = np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)]
        self.current_frame = 0

        def update():
            if self.current_frame < n_frames:
                x, y, z = path[self.current_frame]
                self.plotter.camera_position = [(x, y, 10), (0, 0, 0), (0, 1, 0)]
                self.plotter.render()
                self.current_frame += 1
            else:
                pn.state.curdoc().remove_periodic_callback(callback)
                print("Completed camera spin.")

        callback = pn.state.curdoc().add_periodic_callback(
            update, interval=duration * 1000 // n_frames
        )

    def add_floor(self, plotter, size=15, position=(0, 0, -5)):
        floor = pv.Plane(center=position, direction=(0, 0, 1), i_size=size, j_size=size)
        plotter.add_mesh(floor, color="lightgrey", opacity=1.0)

    def add_custom_lights(self, plotter):
        print("Adding custom lights.")
        plotter.remove_all_lights()

        light1 = pv.Light(
            position=(5, 5, 5), focal_point=(0, 0, 0), color="white", intensity=1.0
        )
        light2 = pv.Light(
            position=(-5, -5, 5), focal_point=(0, 0, 0), color="white", intensity=0.5
        )
        light3 = pv.Light(
            position=(5, -5, -5), focal_point=(0, 0, 0), color="white", intensity=0.5
        )

        plotter.add_light(light1)
        plotter.add_light(light2)
        plotter.add_light(light3)

    def new_plotter(self, rows=1, cols=1):
        # Reset the plotter with new dimensions
        print(f"Creating new plotter with rows={rows}, cols={cols}")
        self.plotter = pv.Plotter(window_size=(800, 600), shape=(rows, cols))
        self.plotter_widget.object = self.plotter

    def display_list(self, style="sb", light="Auto", panelsize=512):
        sslist = self.current_sslist
        ssbonds = sslist.data
        tot_ss = len(ssbonds)  # number of ssbonds
        rows, cols = grid_dimensions(tot_ss)
        winsize = (panelsize * cols, panelsize * rows)

        self.new_plotter(rows, cols)

        pl = self.plotter

        avg_enrg = sslist.average_energy
        avg_dist = sslist.average_distance
        resolution = sslist.resolution

        if light.lower() == "light":
            pv.set_plot_theme("document")
        elif light.lower() == "dark":
            pv.set_plot_theme("dark")
        else:
            pv.set_plot_theme(get_theme())

        title = f"{resolution:.2f} Å: ({tot_ss} SS), Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"

        pl = sslist._render(pl, style)
        pl.enable_anti_aliasing("msaa")
        pl.reset_camera()

    def display(self, light="Auto", shadows=False):
        print("Updating display...")
        style = self.current_style
        camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]

        plotter = self.plotter
        plotter.camera_position = camera_position

        plotter.camera_set = True
        plotter.camera.view_up = (0, 1, 0)

        plotter.clear()
        plotter.add_axes()
        plotter.reset_camera()

        if shadows:
            plotter.enable_shadows()

        title = (
            f"{self.current_ss.energy:.2f} kcal/mol. "
            f"Cα: {self.current_ss.ca_distance:.2f} Å "
            f"Cβ: {self.current_ss.cb_distance:.2f} Å "
            f"Tors: {self.current_ss.torsion_length:.2f}°"
        )
        _theme = light.lower() if light.lower() in ["light", "dark"] else get_theme()
        if _theme == "dark":
            pv.set_plot_theme("dark")
            plotter.set_background("black")
        else:
            pv.set_plot_theme("document")
            plotter.set_background("white")

        self.current_theme = _theme
        self.apply_theme()

        plotter.enable_anti_aliasing("msaa")

        self.add_custom_lights(plotter)

        if self.single:
            print(f"Rendering single disulfide: {self.current_ss.name}")
            self.current_ss._render(plotter, style=style)
        else:
            print("Rendering multiple disulfides.")
            self.display_overlay(plotter)

        plotter.camera.SetParallelProjection(False)  # Perspective
        plotter.reset_camera()
        plotter.render()

        # Update status bar
        self.status_bar.value = f"Displaying: {self.current_ss.name}"

    def display_overlay(self, plotter):
        print("Displaying overlay of multiple disulfides.")
        sslist = self.current_sslist
        ssbonds = sslist.data
        tot_ss = len(ssbonds)
        avg_enrg = sslist.average_energy
        avg_dist = sslist.average_distance
        resolution = sslist.resolution

        res = 100
        if tot_ss > 100:
            res = 60
        if tot_ss > 200:
            res = 30
        if tot_ss > 300:
            res = 8

        title = (
            f"{self.current_ss.name}: {resolution:.2f} Å: ({tot_ss} SS), "
            f"Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"
        )

        plotter.add_axes()

        mycol = get_jet_colormap(tot_ss)

        # Scale bond radii
        if tot_ss < 10:
            brad = BOND_RADIUS
        elif tot_ss < 25:
            brad = BOND_RADIUS * 0.75
        elif tot_ss < 50:
            brad = BOND_RADIUS * 0.75 * 0.8
        elif tot_ss < 100:
            brad = BOND_RADIUS * 0.75 * 0.8 * 0.7
        else:
            brad = BOND_RADIUS * 0.75 * 0.8 * 0.7 * 0.6

        for i, ss in zip(range(tot_ss), ssbonds):
            color = [int(c) for c in mycol[i]]
            ss._render(
                plotter,
                style="plain",
                bondcolor=color,
                translate=False,
                bond_radius=brad,
                res=res,
            )

        self.set_camera_view()
        plotter.render()

    def save_screenshot(self, event):
        print("Saving screenshot.")
        # Panel does not support native file dialogs. Use a FileDownload widget.
        screenshot = self.plotter.screenshot()  # Returns a NumPy array
        # Convert to PNG bytes
        import io

        from PIL import Image

        image = Image.fromarray(screenshot)
        byte_io = io.BytesIO()
        image.save(byte_io, format="PNG")
        byte_io.seek(0)

        download_button = pn.widgets.FileDownload(
            filename="screenshot.png",
            content=byte_io.read(),
            button_type="success",
            label="Download Screenshot",
        )

        # Display the download button in the status bar
        pn.state.notifications.success(
            "Screenshot captured! Click the button below to download."
        )
        self.layout[-1] = pn.Column(
            self.layout[-1],
            download_button,
            self.status_bar,
            sizing_mode="stretch_width",
        )

    def export_scene(self, event):
        print("Exporting scene.")
        # Export the scene as an HTML file using PyVista's export_html
        file_path = f"{self.current_ss.name}_scene.html"
        self.plotter.export_html(file_path)

        # Read the HTML file
        with open(file_path, "rb") as f:
            content = f.read()

        download_button = pn.widgets.FileDownload(
            filename=file_path,
            content=content,
            button_type="success",
            label="Download Scene HTML",
        )

        # Display the download button in the status bar
        pn.state.notifications.success(
            "Scene exported! Click the button below to download."
        )
        self.layout[-1] = pn.Column(
            self.layout[-1],
            download_button,
            self.status_bar,
            sizing_mode="stretch_width",
        )

    def reset(self):
        """
        Resets all widgets and data structures to their default state.
        """
        print("Resetting application to default state.")
        self.status_bar.value = "Resetting..."
        self.pdb_textbox.value = ""
        if self.ss_dict:
            first_pdb_id = list(self.ss_dict.keys())[0]
            self.pdb_dropdown.value = first_pdb_id
            self.dropdown.options = [ss.name for ss in self.ss_dict[first_pdb_id]]
            self.current_ss = self.ss_dict[first_pdb_id][0]
            self.current_sslist = DisulfideList(
                list(self.ss_dict[first_pdb_id]), "sublist"
            )
        else:
            self.pdb_dropdown.value = None
            self.dropdown.options = []
        self.checkbox_single.value = True
        self.current_style = "sb"
        self.current_pdb_id = self.current_ss.pdb_id
        self.set_camera_view()
        self.display()
        self.status_bar.value = f"Displaying: {self.current_ss.name}"

    def add_custom_lights(self, plotter):
        print("Adding custom lights.")
        plotter.remove_all_lights()

        light1 = pv.Light(
            position=(5, 5, 5), focal_point=(0, 0, 0), color="white", intensity=1.0
        )
        light2 = pv.Light(
            position=(-5, -5, 5), focal_point=(0, 0, 0), color="white", intensity=0.5
        )
        light3 = pv.Light(
            position=(5, -5, -5), focal_point=(0, 0, 0), color="white", intensity=0.5
        )

        plotter.add_light(light1)
        plotter.add_light(light2)
        plotter.add_light(light3)


def main():
    print("Loading PDB data...")
    pdb = Load_PDB_SS(subset=True, verbose=True)
    if pdb is not None:
        print("Loaded database...")
        ss_list = sorted(pdb.SSList, key=lambda ss: ss.pdb_id)
    else:
        print("Reloading...")
        pdb = Load_PDB_SS(subset=False, verbose=True)
        if pdb is None:
            print("Unable to load database!")
            return pn.pane.Markdown("**Error:** Unable to load database.")
        ss_list = sorted(pdb.SSList, key=lambda ss: ss.pdb_id)

    if not ss_list:
        print("SS List is empty!")
        return pn.pane.Markdown("**Error:** No disulfide bonds to display.")

    viewer = DisulfideViewerPanel(ss_list)
    return viewer.layout


# Expose the Panel app
app = main()

# To serve the app, save this script as `disulfide_viewer_panel.py` and run:
# panel serve disulfide_viewer_panel.py --show

# Alternatively, if using Jupyter Notebook or JupyterLab, you can display it directly:
# app
