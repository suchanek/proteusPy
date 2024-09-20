import logging
import sys

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MolecularVisualizer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MolecularVisualizer, self).__init__(parent)
        self.setWindowTitle("Molecular Visualizer with PyVista")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        self.frame = QtWidgets.QFrame()
        self.layout = QtWidgets.QHBoxLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        # PyVista Plotter
        self.plotter = QtInteractor(self.frame)
        self.layout.addWidget(self.plotter.interactor)

        # Control Panel
        self.control_panel = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.control_panel)

        # Add buttons for different styles
        self.btn_cpk = QtWidgets.QPushButton("CPK")
        self.btn_split_bonds = QtWidgets.QPushButton("Split Bonds")
        self.btn_ball_sticks = QtWidgets.QPushButton("Ball & Sticks")
        self.btn_exit = QtWidgets.QPushButton("Exit")

        self.control_panel.addWidget(self.btn_cpk)
        self.control_panel.addWidget(self.btn_split_bonds)
        self.control_panel.addWidget(self.btn_ball_sticks)
        self.control_panel.addStretch()
        self.control_panel.addWidget(self.btn_exit)

        # Connect buttons to functions
        self.btn_cpk.clicked.connect(self.show_cpk)
        self.btn_split_bonds.clicked.connect(self.show_split_bonds)
        self.btn_ball_sticks.clicked.connect(self.show_ball_sticks)
        self.btn_exit.clicked.connect(self.close)

        # Define a sample molecule (Methane: CH4)
        self.define_molecule()

        # Initialize actors
        self.atom_actors = []
        self.bond_actors = []

        # Show initial style
        self.show_cpk()

    def define_molecule(self):
        """Define the atoms and bonds of the molecule."""
        # Example: Methane (CH4)
        self.atoms = [
            {"element": "C", "position": (0, 0, 0)},
            {"element": "H", "position": (1, 1, 1)},
            {"element": "H", "position": (-1, -1, 1)},
            {"element": "H", "position": (-1, 1, -1)},
            {"element": "H", "position": (1, -1, -1)},
        ]

        # Bonds defined by pairs of atom indices
        self.bonds = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
        ]

        # Element colors (CPK coloring)
        self.element_colors = {
            "C": "black",
            "H": "white",
            "O": "red",
            "N": "blue",
            # Add more elements as needed
        }

    def clear_actors(self):
        """Remove all current actors from the plotter."""
        try:
            for actor in self.atom_actors + self.bond_actors:
                self.plotter.remove_actor(actor)
            self.atom_actors = []
            self.bond_actors = []
        except Exception as e:
            logger.error(f"Error clearing actors: {e}")

    def show_cpk(self):
        """Display the molecule in CPK style (atoms as colored spheres)."""
        try:
            self.clear_actors()
            for atom in self.atoms:
                color = self.element_colors.get(atom["element"], "grey")
                sphere = pv.Sphere(radius=0.3, center=atom["position"])
                actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True)
                self.atom_actors.append(actor)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as e:
            logger.error(f"Error in show_cpk: {e}")

    def show_ball_sticks(self):
        """Display the molecule in Ball & Sticks style (atoms and bonds)."""
        try:
            self.clear_actors()
            # Add atoms
            for atom in self.atoms:
                color = self.element_colors.get(atom["element"], "grey")
                sphere = pv.Sphere(radius=0.3, center=atom["position"])
                actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True)
                self.atom_actors.append(actor)
            # Add bonds
            for bond in self.bonds:
                start = self.atoms[bond[0]]["position"]
                end = self.atoms[bond[1]]["position"]
                bond_cylinder = pv.Cylinder(
                    center=(
                        (start[0] + end[0]) / 2,
                        (start[1] + end[1]) / 2,
                        (start[2] + end[2]) / 2,
                    ),
                    direction=(end[0] - start[0], end[1] - start[1], end[2] - start[2]),
                    radius=0.1,
                    height=np.linalg.norm(np.array(end) - np.array(start)),
                )
                actor = self.plotter.add_mesh(
                    bond_cylinder, color="grey", smooth_shading=True
                )
                self.bond_actors.append(actor)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as e:
            logger.error(f"Error in show_ball_sticks: {e}")

    def show_split_bonds(self):
        """Display the molecule with split bonds (atoms and split bonds)."""
        try:
            self.clear_actors()
            # Add atoms
            for atom in self.atoms:
                color = self.element_colors.get(atom["element"], "grey")
                sphere = pv.Sphere(radius=0.3, center=atom["position"])
                actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True)
                self.atom_actors.append(actor)
            # Add split bonds (two parallel cylinders per bond)
            for bond in self.bonds:
                start = self.atoms[bond[0]]["position"]
                end = self.atoms[bond[1]]["position"]
                direction = [end[i] - start[i] for i in range(3)]
                # Normalize direction
                norm = sum([d**2 for d in direction]) ** 0.5
                direction = [d / norm for d in direction]
                # Calculate a perpendicular vector for splitting
                if direction[0] != 0 or direction[1] != 0:
                    perp = [-direction[1], direction[0], 0]
                else:
                    perp = [0, -direction[2], direction[1]]
                # Scale perpendicular vector
                perp = [p * 0.1 for p in perp]
                # First split bond
                center1 = [(start[i] + end[i]) / 2 + perp[i] for i in range(3)]
                bond_cyl1 = pv.Cylinder(
                    center=center1,
                    direction=direction,
                    radius=0.05,
                    height=np.linalg.norm(np.array(end) - np.array(start)),
                )
                actor1 = self.plotter.add_mesh(
                    bond_cyl1, color="grey", smooth_shading=True
                )
                self.bond_actors.append(actor1)
                # Second split bond
                center2 = [(start[i] + end[i]) / 2 - perp[i] for i in range(3)]
                bond_cyl2 = pv.Cylinder(
                    center=center2,
                    direction=direction,
                    radius=0.05,
                    height=np.linalg.norm(np.array(end) - np.array(start)),
                )
                actor2 = self.plotter.add_mesh(
                    bond_cyl2, color="grey", smooth_shading=True
                )
                self.bond_actors.append(actor2)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as e:
            logger.error(f"Error in show_split_bonds: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MolecularVisualizer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
