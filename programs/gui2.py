import os
import sys

os.environ["QT_IM_MODULE"] = "qtvirtualkeyboard"
import pyvista as pv

# pylint: disable=E0611
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from proteusPy import Load_PDB_SS, get_macos_theme


class MolecularViewer(QMainWindow):
    """
    A PyQt5-based application for viewing molecular structures with support for
    multiple disulfide bonds and various rendering styles.

    Attributes:
        ss_list (list): List of disulfide bond objects.
        current_ss (object): Currently selected disulfide bond.
        current_style (str): Currently selected rendering style.
        plotter_widget (QtInteractor): PyVista QtInteractor widget for 3D visualization.
        ss_dict (dict): Dictionary where keys are pdb_id and values are lists of disulfides with that pdb_id.
    """

    def __init__(self, ss_list, winsize=(1200, 800)):
        """
        Initializes the MolecularViewer application.

        Parameters:
            ss_list (list): List of disulfide bond objects.
            winsize (tuple): Minimum size for the PyVista plotter widget.
        """
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Molecular Structure Viewer")
        self.setGeometry(100, 100, winsize[0], winsize[1])

        # Validate and store the list of disulfides
        self.ss_list = ss_list
        if not self.ss_list:
            QMessageBox.critical(
                self, "Error", "No disulfide bonds found in the provided PDB."
            )
            sys.exit(1)
        self.current_ss = self.ss_list[0]
        self.current_style = "sb"  # Initialize with default style

        # Create a dictionary where keys are pdb_id and values are lists of disulfides with that pdb_id
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
        # self.button_cov = QPushButton("COV")

        # Connect buttons to methods with style tracking
        self.button_cpk.clicked.connect(lambda: self.update_style("cpk"))
        self.button_sb.clicked.connect(lambda: self.update_style("sb"))
        self.button_bs.clicked.connect(lambda: self.update_style("bs"))
        self.button_pd.clicked.connect(lambda: self.update_style("pd"))
        # self.button_cov.clicked.connect(lambda: self.update_style("cov"))

        # Create a dropdown selector for PDB IDs
        self.pdb_dropdown = QComboBox()
        self.pdb_dropdown.addItems(self.ss_dict.keys())
        self.pdb_dropdown.currentIndexChanged.connect(self.on_pdb_dropdown_change)

        # Create a dropdown selector for Disulfides
        self.dropdown = QComboBox()
        self.dropdown.addItems([ss.name for ss in self.ss_list])
        self.dropdown.currentIndexChanged.connect(self.on_dropdown_change)

        # Create a button to toggle themes
        self.button_theme = QPushButton("Toggle Theme")
        self.button_theme.clicked.connect(self.toggle_theme)

        # Layout for the buttons and dropdown
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.button_cpk)
        button_layout.addWidget(self.button_sb)
        button_layout.addWidget(self.button_bs)
        button_layout.addWidget(self.button_pd)
        # button_layout.addWidget(self.button_cov)
        button_layout.addWidget(self.pdb_dropdown)  # Add PDB ID dropdown
        button_layout.addWidget(self.dropdown)
        button_layout.addWidget(self.button_theme)  # Add theme toggle button
        button_layout.addStretch(1)

        # Create a QWidget for the buttons and dropdown
        button_widget = QWidget()
        button_widget.setLayout(button_layout)

        # Create the PyVista QtInteractor widget for 3D visualization
        self.plotter_widget = QtInteractor(self)
        # self.plotter_widget.setMinimumSize(*winsize)

        # Main layout combining controls and visualization
        main_layout = QHBoxLayout()
        main_layout.addWidget(button_widget)
        main_layout.addWidget(self.plotter_widget, 1)

        # Central widget setup
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Status Bar for user feedback
        self.statusBar().showMessage(f"Displaying: {self.current_ss.name}")

        # Initialize the theme
        self.current_theme = "Light"  # Default theme
        self.apply_theme()

        # Create the menubar
        self.create_menubar()

        # Initial display
        self.display(self.current_style)  # Set a default view with current style

    def update_style(self, style):
        """
        Updates the current rendering style and refreshes the visualization.

        Parameters:
            style (str): The rendering style to apply (e.g., 'cpk', 'sb').
        """
        self.current_style = style
        self.display(style)

    def on_pdb_dropdown_change(self, index):
        """
        Handles the event when the PDB ID dropdown selection changes.

        Parameters:
            index (int): The index of the selected item in the PDB ID dropdown.
        """
        pdb_id = self.pdb_dropdown.currentText()
        disulfides = self.ss_dict.get(pdb_id, [])
        self.dropdown.clear()
        self.dropdown.addItems([ss.name for ss in disulfides])
        if disulfides:
            self.current_ss = disulfides[0]
            self.statusBar().showMessage(f"Displaying: {self.current_ss.name}")
            self.display(self.current_style)

    def on_dropdown_change(self, index):
        """
        Handles the event when the dropdown selection changes.

        Parameters:
            index (int): The index of the selected item in the dropdown.
        """
        self.current_ss = self.ss_list[index]
        self.statusBar().showMessage(f"Displaying: {self.current_ss.name}")
        self.display(self.current_style)  # Use the current style

    def toggle_theme(self):
        """
        Toggles between light and dark themes for the visualization.
        """
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        print(f"Toggling theme to: {self.current_theme}")

        self.statusBar().showMessage(
            f"Theme switched to: {self.current_theme.capitalize()}"
        )
        self.apply_theme()

    def apply_theme(self):
        """
        Applies the specified theme to the visualization.
        """
        _theme = self.current_theme.lower()
        if _theme == "dark":
            pv.set_plot_theme("dark")
            self.plotter_widget.set_background("black")
        else:
            pv.set_plot_theme("document")
            self.plotter_widget.set_background("white")
        print(f"Applying theme: {_theme}")

        # Render the plotter to apply the changes
        self.plotter_widget.render()

    def create_menubar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Save screenshot action
        save_screenshot_action = QAction("Save Screenshot", self)
        save_screenshot_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(save_screenshot_action)

        # Export scene action
        export_scene_action = QAction("Export Scene", self)
        export_scene_action.triggered.connect(self.export_scene)
        file_menu.addAction(export_scene_action)

    def save_screenshot(self):
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
            position=(1, 1, 1), focal_point=(0, 0, 0), color="white", intensity=1.0
        )
        light2 = pv.Light(
            position=(-1, -1, 1), focal_point=(0, 0, 0), color="white", intensity=0.5
        )
        light3 = pv.Light(
            position=(1, -1, -1), focal_point=(0, 0, 0), color="white", intensity=0.5
        )

        plotter.add_light(light1)
        plotter.add_light(light2)
        plotter.add_light(light3)

    def display(self, style, light="Auto", single=True, winsize=(800, 800)):
        """
        Renders the current disulfide bond using the specified style.

        Parameters:
            style (str): The rendering style (e.g., 'cpk', 'sb').
            light (str): The lighting theme ('Light', 'Dark', or 'Auto').
            single (bool): Whether to display a single bond or multiple.
            winsize (tuple): The window size for rendering.
        """

        plotter = self.plotter_widget  # Use QtInteractor as the plotter

        plotter.clear()

        title = f"{self.current_ss.energy:.2f} kcal/mol. Cα: {self.current_ss.ca_distance:.2f} Å Cβ: {self.current_ss.cb_distance:.2f} Å Tors: {self.current_ss.torsion_length:.2f}°"

        # Determine the theme and set the background color
        _theme = (
            light.lower() if light in ["Light", "Dark"] else get_macos_theme().lower()
        )
        if _theme == "dark":
            pv.set_plot_theme("dark")
            plotter.set_background("black")
        else:
            pv.set_plot_theme("document")
            plotter.set_background("white")

        # Apply the current theme
        self.current_theme = _theme

        self.apply_theme()

        self.setWindowTitle(f"{self.current_ss.name} - {title}")
        plotter.enable_anti_aliasing("msaa")

        # Add custom lights
        self.add_custom_lights(plotter)

        # Perform the plotting with the specified style
        try:
            self.current_ss._render(plotter, style=style)
        except AttributeError as e:
            QMessageBox.critical(self, "Error", f"Rendering failed: {e}")
            return

        # Set perspective projection
        plotter.camera.SetParallelProjection(False)  # False for perspective

        plotter.reset_camera()
        plotter.render()


def main():
    """
    The main entry point of the application.
    """
    try:
        pdb = Load_PDB_SS(subset=True, verbose=True)
        ss_list = (
            pdb.SSList
        )  # Ensure Load_PDB_SS returns an object with SSList attribute
    except Exception as e:
        # Since we are outside the QApplication context, use print for error
        print(f"Error loading PDB: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = MolecularViewer(ss_list)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
