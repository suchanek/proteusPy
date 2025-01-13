"""
Disulfide Structure Viewer

This program is part of the proteusPy package, and is a PyQt5-based application for viewing 
molecular structures with support for multiple disulfide bonds and various rendering styles. 
It uses PyVista for 3D visualization and PyQt5 for the graphical user interface. The application 
allows users to toggle between different rendering styles, switch between light and dark themes, 
and save or export the visualized scenes.

Modules:
    os: Provides a way of using operating system dependent functionality.
    sys: Provides access to some variables used or maintained by the interpreter.
    pyvista: A 3D plotting and mesh analysis through a streamlined interface for the Visualization 
    Toolkit (VTK).
    PyQt5.QtWidgets: Provides a set of UI elements to create classic desktop-style user interfaces.
    pyvistaqt: Provides a PyQt5 widget for embedding PyVista plots.
    proteusPy: Custom module for loading PDB files and getting macOS theme.

Classes:
    DisulfideViewer: Main window class for the molecular structure viewer application.

Functions:
    main: The main entry point of the application.

Author:
    Eric G. Suchanek, PhD.
"""

# pylint: disable=W0212
# pylint: disable=E0611
# pylint: disable=E0401
# pylint: disable=C0413

import logging
import os
import sys

import numpy as np
import pyvista as pv


def can_import(module_name):
    """Check we can import the requested module."""
    try:
        return __import__(module_name)
    except ImportError:
        return None


# Check if PyQt5 is installed
if can_import("PyQt5") is None:
    sys.exit(
        "This program requires PyQt5 to be installed. "
        "Please install it using: pip install proteusPy[pyqt5]"
    )

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# pylint: disable=E0611
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from proteusPy import (
    BOND_RADIUS,
    DisulfideList,
    Load_PDB_SS,
    configure_master_logger,
    create_logger,
    get_jet_colormap,
    get_theme,
)

os.environ["QT_IM_MODULE"] = "qtvirtualkeyboard"

_version = 1.0

_logger = create_logger(__name__, log_level=logging.INFO)

configure_master_logger("viewer.py")
_logger.info("Starting Disulfide Viewer...")


class DisulfideViewer(QMainWindow):
    """
    A PyQt5-based application for viewing molecular structures with support for
    multiple disulfide bonds and various rendering styles.

    Attributes:
        ss_list (list): List of disulfide bond objects.
        current_ss (object): Currently selected disulfide bond.
        current_style (str): Currently selected rendering style.
        plotter_widget (QtInteractor): PyVista QtInteractor widget for 3D visualization.
        ss_dict (dict): Dictionary where keys are pdb_id and values are lists of disulfides
        with that pdb_id.
    """

    def __init__(self, ss_list, winsize=(1200, 800)):
        """
        Initializes the MolecularViewer application.

        Parameters:
            ss_list (list): List of disulfide bond objects.
            winsize (tuple): Minimum size for the PyVista plotter widget.
        """
        super().__init__()

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
        }

        # Set up the main window
        self._version = _version
        self.setWindowTitle("proteusPy Disulfide Viewer")
        self.setGeometry(100, 100, winsize[0], winsize[1])
        self.timer = None
        self.current_frame = 0

        # Validate and store the list of disulfides
        self.ss_list = ss_list
        self.current_ss = self.ss_list[0]
        self.current_pdb_id = self.current_ss.pdb_id
        self.current_style = "sb"  # Initialize with default style
        self.single = True

        # Create a dictionary where keys are pdb_id and values are lists of disulfides
        # with that pdb_id
        self.current_sslist = DisulfideList([], "sublist")

        # Populate the dictionary with key pdb_id and values list of disulfides
        # for that pdb_id
        self.ss_dict = {}
        for ss in self.ss_list:
            if ss.pdb_id not in self.ss_dict:
                self.ss_dict[ss.pdb_id] = []
            self.ss_dict[ss.pdb_id].append(ss)

        # Create buttons for different rendering styles
        self.button_cpk = QPushButton("CPK")
        self.button_sb = QPushButton("SB")
        self.button_bs = QPushButton("BS")
        self.button_pd = QPushButton("PD")

        # Connect buttons to methods with style tracking
        self.button_cpk.clicked.connect(lambda: self.update_style("cpk"))
        self.button_sb.clicked.connect(lambda: self.update_style("sb"))
        self.button_bs.clicked.connect(lambda: self.update_style("bs"))
        self.button_pd.clicked.connect(lambda: self.update_style("pd"))

        # Create a text box for entering PDB IDs
        self.pdb_textbox = QLineEdit()
        self.pdb_textbox.setPlaceholderText("Enter PDB ID")
        self.pdb_textbox.returnPressed.connect(self.on_pdb_textbox_enter)

        # Create a dropdown selector for PDB IDs
        self.pdb_dropdown = QComboBox()
        self.pdb_dropdown.addItems(self.ss_dict.keys())
        self.pdb_dropdown.currentIndexChanged.connect(self.on_pdb_dropdown_change)

        # Create a dropdown selector for Disulfides
        self.dropdown = QComboBox()
        self.dropdown.addItems([ss.name for ss in self.ss_list])
        self.dropdown.currentIndexChanged.connect(self.on_dropdown_change)

        # Create a checkbox for single/multiple display
        self.checkbox_single = QCheckBox("Single")
        self.checkbox_single.setChecked(True)
        self.checkbox_single.stateChanged.connect(self.on_checkbox_single_change)

        self.button_reset = QPushButton("Reset Camera")
        self.button_reset.clicked.connect(self.set_camera_view)

        # self.timer2 = QTimer()
        # self.timer2.timeout.connect(self.update_sliders_from_camera)
        # self.timer2.start(100)  # Check every 100 milliseconds

        # Create a font object with increased size
        font = QFont()
        font.setPointSize(14)  # Set the desired font size

        # Group the PDB controls
        pdb_groupbox = QGroupBox("PDB Controls")
        pdb_groupbox.setFont(font)  # Set the font for the groupbox label
        pdb_layout = QVBoxLayout()
        pdb_layout.addWidget(self.pdb_textbox)
        pdb_layout.addWidget(self.pdb_dropdown)
        pdb_layout.addWidget(self.dropdown)
        pdb_groupbox.setLayout(pdb_layout)

        # Group the buttons with a label
        button_group = QGroupBox("Rendering Styles")
        button_group.setFont(font)  # Set the font for the groupbox label
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.button_cpk)
        button_layout.addWidget(self.button_sb)
        button_layout.addWidget(self.button_bs)
        button_layout.addWidget(self.button_pd)
        button_layout.addWidget(self.checkbox_single)  # Add single/multiple checkbox

        self.button_spin = QPushButton("Spin Molecule")
        self.button_spin.clicked.connect(self.spin_camera)

        button_layout.addWidget(self.button_spin)  # Add spin camera button

        # button_layout.addWidget(self.button_cov)
        button_group.setLayout(button_layout)

        # Layout for the main controls
        control_layout = QVBoxLayout()
        control_layout.addWidget(button_group)  # Add button group
        control_layout.addSpacerItem(
            QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )  # Add spacer
        control_layout.addWidget(pdb_groupbox)  # Add PDB groupbox
        control_layout.addSpacerItem(
            QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )  # Add spacer

        control_layout.addWidget(self.button_reset)  # Add Reset button
        control_layout.addStretch(1)

        # Create a QWidget for the controls
        control_widget = QWidget()
        control_widget.setLayout(control_layout)

        # Create the PyVista QtInteractor widget for 3D visualization
        self.plotter_widget = QtInteractor(self)
        # self.plotter_widget.setMinimumSize(*winsize)

        # Create sliders for camera control
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(0, 360)
        self.slider_x.setValue(180)
        self.slider_x.valueChanged.connect(self.update_camera_position)

        self.slider_y = QSlider(Qt.Vertical)
        self.slider_y.setRange(0, 360)
        self.slider_y.setValue(180)
        self.slider_y.valueChanged.connect(self.update_camera_position)

        self.button_spin = QPushButton("Spin Camera")
        self.button_spin.clicked.connect(self.spin_camera)

        # Main layout combining controls and visualization
        main_layout = QHBoxLayout()
        main_layout.addWidget(control_widget)
        main_layout.addWidget(self.plotter_widget, 1)
        main_layout.addWidget(self.slider_y)  # Add vertical slider

        # Set the central widget
        container = QWidget()
        container.setLayout(main_layout)

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(container)
        bottom_layout.addWidget(self.slider_x)  # Add horizontal slider

        main_container = QWidget()
        main_container.setLayout(bottom_layout)
        self.setCentralWidget(main_container)

        # Status Bar for user feedback
        self.statusBar().showMessage(f"Displaying: {self.current_ss.name}")

        # Initialize the theme
        self.current_theme = "auto"  # Default theme
        self.apply_theme()

        # Create the menubar
        self.create_menubar()

        # Force a refresh on the disulfide list dropdown
        self.set_camera_view()
        self.on_pdb_dropdown_change(0)

        # Initial display
        # self.display()  # Set a default view with current style

    def update_sliders_from_camera(self):
        """
        Updates the slider values to match the camera's position.

        Parameters:
            caller: The object that triggered the event.
            event: The event type (ModifiedEvent).
        """
        # Get the current camera position
        current_position = self.plotter_widget.camera_position[0]
        current_x = current_position[0]
        current_y = current_position[1]
        current_z = current_position[2]

        # Calculate the spherical coordinates
        distance = np.sqrt(current_x**2 + current_y**2 + current_z**2)

        # Avoid division by zero for atan2
        x_angle = np.degrees(np.arctan2(current_z, current_x)) if distance > 0 else 0
        y_angle = (
            np.degrees(np.arctan2(current_y, np.sqrt(current_x**2 + current_z**2)))
            if distance > 0
            else 0
        )

        # Ensure the angles are within 0-360 degrees
        x_angle = x_angle % 360
        y_angle = y_angle % 360

        # Update the sliders only if values differ to avoid unnecessary signals
        if self.slider_x.value() != int(x_angle):
            self.slider_x.blockSignals(True)  # Temporarily block signals to avoid loops
            self.slider_x.setValue(int(x_angle))
            self.slider_x.blockSignals(False)

        if self.slider_y.value() != int(y_angle):
            self.slider_y.blockSignals(True)  # Temporarily block signals to avoid loops
            self.slider_y.setValue(int(y_angle))
            self.slider_y.blockSignals(False)

    def set_camera_view(self):
        """
        Sets the camera to a specific view where the x-axis is pointed down and the
        y-axis into the screen, and resets the x and y sliders to their default values.
        """
        # Set the camera position
        camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]  # Example values
        self.plotter_widget.camera_position = camera_position
        self.plotter_widget.reset_camera()

        # Reset the x and y sliders to their default values
        self.slider_x.setValue(180)  # Assuming the default value is 180
        self.slider_y.setValue(180)  # Assuming the default value is 180

    def spin_camera(self, duration=5, n_frames=150):
        """
        Spins the camera around the z-axis in an orbital path.

        :param duration: Duration of the spin in seconds.
        :param n_frames: Number of frames for the animation.
        """
        # Define the circular path around the z-axis
        theta = np.linspace(0, 2 * np.pi, n_frames)
        path = np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)]
        duration = 5

        # Function to update the camera position
        def update_camera():
            self.plotter_widget.camera_position = path[self.current_frame]
            self.current_frame = (self.current_frame + 1) % n_frames

        # Create a QTimer to update the camera position
        self.timer = QTimer(self)
        self.timer.timeout.connect(update_camera)
        self.current_frame = 0

        # Function to stop the timer after the duration
        def stop_timer():
            self.timer.stop()

        # Start the timer
        interval = int(duration * 1000 / n_frames)
        self.timer.start(interval)
        QTimer.singleShot(duration * 1000, stop_timer)

    def reset(self):
        """
        Resets all widgets and data structures to their default state.
        """
        self.statusBar().showMessage("Resetting...")
        self.pdb_textbox.clear()
        self.pdb_dropdown.setCurrentIndex(0)
        self.dropdown.clear()
        self.checkbox_single.setChecked(True)
        self.current_style = "sb"
        self.current_ss = self.ss_list[0]
        self.current_pdb_id = self.current_ss.pdb_id
        self.set_camera_view()
        self.on_pdb_dropdown_change(0)

    def add_floor(self, plotter, size=15, position=(0, 0, -5)):
        """
        Adds a simple plane beneath the disulfide bond to act as a 'floor'.

        :param plotter: The PyVista plotter to add the floor to.
        :type plotter: pv.Plotter
        :param size: The size of the floor plane.
        :type size: float
        :param position: The position of the floor plane.
        :type position: tuple
        """
        floor = pv.Plane(center=position, direction=(0, 0, 1), i_size=size, j_size=size)
        plotter.add_mesh(floor, color="lightgrey", opacity=1.0)

    def update_style(self, style):
        """
        Updates the current rendering style and refreshes the visualization.

        :param style: The rendering style to apply (e.g., 'cpk', 'sb').
        :type style: str
        """
        self.current_style = style
        self.checkbox_single.setChecked(True)  # Ensure the "Single" checkbox is checked
        self.display()

    def on_pdb_textbox_enter(self):
        """
        Handles the event when the user presses Enter in the PDB ID text box.
        """
        pdb_id = self.pdb_textbox.text()
        if pdb_id in self.ss_dict:
            self.update_pdb_selection(pdb_id)
        else:
            QMessageBox.warning(
                self, "Invalid PDB ID", "The entered PDB ID is not valid."
            )
        disulfides = self.ss_dict.get(pdb_id, [])

        self.dropdown.clear()
        self.dropdown.addItems([ss.name for ss in disulfides])
        self.current_sslist = DisulfideList(list(disulfides), "sub")
        self.pdb_dropdown.setCurrentText(pdb_id)

    def on_pdb_dropdown_change(self, index):
        """
        Handles the event when the PDB ID dropdown selection changes.

        Parameters:
            index (int): The index of the selected item in the PDB ID dropdown.
        """
        pdb_id = self.pdb_dropdown.currentText()

        self.dropdown.clear()
        self.pdb_textbox.setText(pdb_id)

        disulfides = self.ss_dict.get(pdb_id, [])

        if disulfides:
            self.dropdown.addItems([ss.name for ss in disulfides])
            self.current_ss = disulfides[0]
            self.current_sslist = DisulfideList(list(disulfides), "sublist")
            self.statusBar().showMessage(f"Displaying: {self.current_ss.name}")
            self.display()
        else:
            self.statusBar().showMessage(f"No Disulfides: {pdb_id}")

    def update_pdb_selection(self, pdb_id):
        """
        Updates the current PDB selection and refreshes the disulfide dropdown.

        :param pdb_id: The PDB ID to select.
        :type pdb_id: str
        """
        self.current_pdb_id = pdb_id
        disulfides = self.ss_dict.get(pdb_id, [])

        self.current_sslist = DisulfideList(list(disulfides), "sublist2")
        self.dropdown.clear()
        self.dropdown.addItems([ss.name for ss in self.ss_list])

    def on_dropdown_change(self, index):
        """
        Handles the event when the dropdown selection changes.

        Parameters:
            index (int): The index of the selected item in the dropdown.
        """
        selected_name = self.dropdown.currentText()
        self.current_ss = next(
            (ss for ss in self.ss_list if ss.name == selected_name), None
        )
        if self.current_ss is not None:
            self.statusBar().showMessage(f"Displaying: {self.current_ss.name}")
            self.display()

    def on_checkbox_single_change(self, state):
        """
        Handles the event when the single/multiple checkbox state changes.

        Parameters:
            state (int): The state of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.single = state == Qt.Checked
        self.display(self)  # Use the current style

    def update_camera_position(self):
        """
        Update the camera position based on the slider values.
        """

        x_angle = self.slider_x.value()
        y_angle = self.slider_y.value()

        # Convert angles to radians
        x_rad = np.radians(x_angle)
        y_rad = np.radians(y_angle)

        # Read the current camera position
        current_position = self.plotter_widget.camera_position[0]
        current_x = current_position[0]
        current_y = current_position[1]
        current_z = current_position[2]

        # Calculate the current distance from the origin
        distance = np.sqrt(current_x**2 + current_y**2 + current_z**2)

        # Calculate new camera position based on the angles and distance
        xpos = distance * np.cos(y_rad) * np.cos(x_rad)
        ypos = distance * np.sin(y_rad)
        zpos = distance * np.cos(y_rad) * np.sin(x_rad)

        # Set the camera position and up vector
        self.plotter_widget.camera_position = [(xpos, ypos, zpos), (0, 0, 0), (0, 1, 0)]
        self.plotter_widget.render()

    def apply_theme(self):
        """
        Apply the current theme to the plotter widget.
        """
        if self.current_theme == "dark":
            self.plotter_widget.set_background(self.colors["dark"]["background"])
        else:
            self.plotter_widget.set_background(self.colors["light"]["background"])

        # Render the plotter to apply the changes
        self.plotter_widget.render()

    def create_menubar(self):
        """
        Creates the menubar
        """
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        reset_menu = menubar.addMenu("Reset")

        # Save screenshot action
        save_screenshot_action = QAction("Save Screenshot", self)
        save_screenshot_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(save_screenshot_action)

        # Export scene action
        export_scene_action = QAction("Export Scene", self)
        export_scene_action.triggered.connect(self.export_scene)
        file_menu.addAction(export_scene_action)

        # Reset action
        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset)
        reset_menu.addAction(reset_action)

    def save_screenshot(self):
        """
        Save the current scene as a screenshot
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "",
            "PNG Files (*.png);;All Files (*)",
            options=options,
        )
        if file_path:
            self.plotter_widget.screenshot(file_path)

    def export_scene(self):
        """
        Export the current scene as an HTML file
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Scene",
            "",
            "HTML Files (*.html);;All Files (*)",
            options=options,
        )
        if file_path:
            self.plotter_widget.export_html(file_path)

    def add_custom_lights(self, plotter):
        """
        Adds custom lighting to the plotter for enhanced visualization.

        Parameters:
            plotter (QtInteractor): The PyVista plotter widget.
        """
        # Remove existing lights to prevent duplication
        plotter.remove_all_lights()

        # Define multiple light sources for better illumination
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
        """
        Create a new QtInteractor with the specified number of rows and columns.
        """

        # Create a new QtInteractor with the desired shape
        new_plotter_widget = QtInteractor(self, shape=(rows, cols))

        # Update the layout to replace the old plotter widget with the new one
        layout = self.centralWidget().layout()
        layout.removeWidget(self.plotter_widget)  # Remove the old plotter widget
        self.plotter_widget.setParent(
            None
        )  # Detach the old plotter widget from its parent

        layout.addWidget(new_plotter_widget)  # Add the new plotter widget to the layout
        self.plotter_widget = (
            new_plotter_widget  # Update the reference to the new plotter widget
        )

    def display(self, light="Auto", shadows=False):
        """
        Renders the current disulfide bond using the specified style.

        Parameters:
            style (str): The rendering style (e.g., 'cpk', 'sb').
            light (str): The lighting theme ('Light', 'Dark', or 'Auto').
            single (bool): Whether to display a single bond or multiple.
            winsize (tuple): The window size for rendering.
        """
        style = self.current_style
        camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]  # Example values

        plotter = self.plotter_widget  # Use QtInteractor as the plotter
        self.plotter_widget.camera_position = camera_position

        # Set the camera up vector
        plotter.camera_set = True
        plotter.camera.view_up = (0, 1, 0)

        plotter.clear()
        plotter.add_axes()
        plotter.reset_camera()

        if shadows is True:
            plotter.enable_shadows()

        title = (
            f"{self.current_ss.energy:.2f} kcal/mol. "
            f"Cα: {self.current_ss.ca_distance:.2f} Å "
            f"Cβ: {self.current_ss.cb_distance:.2f} Å "
            f"Tors: {self.current_ss.torsion_length:.2f}°"
        )
        # Determine the theme and set the background color
        _theme = light.lower() if light in ["Light", "Dark"] else get_theme()
        if _theme == "dark":
            pv.set_plot_theme("dark")
            plotter.set_background("black")
        else:
            pv.set_plot_theme("document")
            plotter.set_background("white")

        # Apply the current theme
        self.current_theme = _theme
        self.apply_theme()

        self.setWindowTitle(f"{self.current_ss.name}: {title}")
        plotter.enable_anti_aliasing("msaa")

        # Add custom lights
        self.add_custom_lights(plotter)

        # Perform the plotting with the specified style
        if self.single is True:
            # self.new_plotter(rows=1, cols=1)
            self.current_ss._render(plotter, style=style)
        else:
            self.display_overlay(plotter)

        # Set perspective projection
        plotter.camera.SetParallelProjection(False)  # False for perspective
        plotter.reset_camera()
        plotter.render()

    def display_overlay(self, pl):
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
        sslist = self.current_sslist
        ssbonds = sslist.data
        tot_ss = len(ssbonds)  # number off ssbonds
        avg_enrg = sslist.average_energy
        avg_dist = sslist.average_distance
        resolution = sslist.resolution

        res = 100

        if tot_ss > 20:
            res = 60
        if tot_ss > 30:
            res = 30
        if tot_ss > 50:
            res = 8

        title = (
            f"{self.current_ss.name}: {resolution:.2f} Å: ({tot_ss} SS), "
            f"Avg E: {avg_enrg:.2f} kcal/mol, Avg Dist: {avg_dist:.2f} Å"
        )

        self.setWindowTitle(f"{title}")

        pl.add_axes()

        mycol = np.zeros(shape=(tot_ss, 3))
        mycol = get_jet_colormap(tot_ss)

        # scale the overlay bond radii down so that we can see the individual elements better
        # maximum 90% reduction

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
            color = [int(mycol[i][0]), int(mycol[i][1]), int(mycol[i][2])]
            ss._render(
                pl,
                style="plain",
                bondcolor=color,
                translate=False,
                bond_radius=brad,
                res=res,
            )

        self.set_camera_view()

        return pl


def main():
    """
    The main entry point of the application.
    """

    pdb = Load_PDB_SS(subset=False, verbose=True)
    if pdb is not None:
        ss_list = sorted(pdb.SSList, key=lambda ss: ss.pdb_id)
        app = QApplication(sys.argv)
        viewer = DisulfideViewer(ss_list)
        viewer.show()
        sys.exit(app.exec_())
    else:
        print("Unable to load database!")
        sys.exit(1)


if __name__ == "__main__":
    main()
