import sys

import pyvista as pv
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from proteusPy import Load_PDB_SS
from proteusPy.atoms import (
    ATOM_COLORS,
    ATOM_RADII_COVALENT,
    ATOM_RADII_CPK,
    BOND_COLOR,
    BOND_RADIUS,
    BS_SCALE,
    FONTSIZE,
    SPEC_POWER,
    SPECULARITY,
)


class MolecularViewer(QMainWindow):
    def __init__(self, ss, winsize=(800, 800)):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Molecular Structure Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Create buttons for different styles
        self.button_cpk = QPushButton("CPK")
        self.button_sb = QPushButton("SB")
        self.button_bs = QPushButton("BS")
        self.button_pd = QPushButton("PD")
        self.button_cov = QPushButton("COV")

        # Connect buttons to methods
        self.button_cpk.clicked.connect(lambda: self.display("cpk"))
        self.button_sb.clicked.connect(lambda: self.display("sb"))
        self.button_bs.clicked.connect(lambda: self.display("bs"))
        self.button_pd.clicked.connect(lambda: self.display("pd"))
        self.button_cov.clicked.connect(lambda: self.display("cov"))

        # Layout for the buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.button_cpk)
        button_layout.addWidget(self.button_sb)
        button_layout.addWidget(self.button_bs)
        button_layout.addWidget(self.button_pd)
        button_layout.addWidget(self.button_cov)
        button_layout.addStretch(1)

        # Create a QWidget for the buttons
        button_widget = QWidget()
        button_widget.setLayout(button_layout)

        # Create the PyVista QtInteractor widget
        self.plotter_widget = QtInteractor(self)
        self.plotter_widget.setMinimumSize(*winsize)

        # Set background color to black
        self.plotter_widget.set_background("black")

        # Add bounding box
        # self.plotter_widget.add_bounding_box(
        #    color="red", line_width=2, name="bounding_box"
        # )

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(button_widget)
        main_layout.addWidget(self.plotter_widget, 1)

        # Central widget setup
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Initialize the disulfide bond object
        self.ss = ss

        # Create the menubar
        self.create_menubar()

        # Initial display
        self.display("sb")  # Set a default view

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
            "OBJ Files (*.obj);;All Files (*)",
            options=options,
        )
        if file_path:
            self.plotter_widget.export_obj(file_path)

    def add_custom_lights(self, plotter):
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
        # plotter.add_light(light3)

    def display(self, style, light="Auto", single=True, winsize=(800, 800)):
        from proteusPy import get_macos_theme

        plotter = self.plotter_widget  # Use QtInteractor as the plotter

        plotter.clear()

        title = "Disulfide Bond"

        # Set plot theme based on light preference
        _theme = (
            light.lower() if light in ["Light", "Dark"] else get_macos_theme().lower()
        )
        if _theme == "dark":
            pv.set_plot_theme("dark")
            plotter.set_background("black")
        else:
            pv.set_plot_theme("document")
            plotter.set_background("white")

        self.setWindowTitle(f"{self.ss.name} - {title}")
        plotter.enable_anti_aliasing("msaa")

        # Add custom lights
        self.add_custom_lights(plotter)

        # Perform the plotting with the specified style
        self.ss._render(plotter, style=style)
        # self.plot(plotter=plotter, style=style)

        # Set perspective projection
        plotter.camera.SetParallelProjection(False)  # False for perspective

        plotter.reset_camera()
        plotter.render()


if __name__ == "__main__":
    try:
        pdb = Load_PDB_SS(subset=True, verbose=True)
        ss1 = pdb[0]
    except Exception as e:
        print(f"Error loading PDB: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = MolecularViewer(ss1)
    viewer.show()
    sys.exit(app.exec_())
