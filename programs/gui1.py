import sys

import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from proteusPy import Disulfide, Load_PDB_SS
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
        self.button1 = QPushButton("cpk")
        self.button2 = QPushButton("sb")
        self.button3 = QPushButton("bs")

        # Connect buttons to methods
        self.button1.clicked.connect(lambda: self.display("cpk"))
        self.button2.clicked.connect(lambda: self.display("sb"))
        self.button3.clicked.connect(lambda: self.display("bs"))

        # Layout for the buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)
        button_layout.addStretch(1)

        # Create a QWidget for the buttons
        button_widget = QWidget()
        button_widget.setLayout(button_layout)

        # Create the PyVista plotter widget with QtInteractor
        self.plotter_widget = QtInteractor(self)
        self.plotter_widget.setMinimumSize(*winsize)

        # Associate the QtInteractor's interactor with a PyVista Plotter
        self.plotter = pv.Plotter(off_screen=False)
        self.plotter.ren_win.SetInteractor(self.plotter_widget.interactor)

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

    def plot(
        self,
        style="bs",
        bondcolor=BOND_COLOR,
        bs_scale=BS_SCALE,
        spec=SPECULARITY,
        specpow=SPEC_POWER,
        translate=True,
        bond_radius=BOND_RADIUS,
        res=100,
    ):
        pvp = self.plotter
        ss = self.ss
        model = ss.modelled
        missing_atoms = ss.missing_atoms
        coords = ss._internal_coords()
        clen = coords.shape[0]

        if model:
            all_atoms = False
        else:
            all_atoms = True

        if translate:
            coords -= ss.cofmass

        atoms = (
            "N",
            "C",
            "C",
            "O",
            "C",
            "SG",
            "N",
            "C",
            "C",
            "O",
            "C",
            "SG",
            "C",
            "N",
            "C",
            "N",
        )

        if style == "cpk":
            for i, atom in enumerate(atoms):
                rad = ATOM_RADII_CPK[atom]
                pvp.add_mesh(
                    pv.Sphere(center=coords[i], radius=rad),
                    color=ATOM_COLORS[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )

        elif style == "cov":
            for i, atom in enumerate(atoms):
                rad = ATOM_RADII_COVALENT[atom]
                pvp.add_mesh(
                    pv.Sphere(center=coords[i], radius=rad),
                    color=ATOM_COLORS[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )

        elif style == "bs":  # ball and stick
            for i, atom in enumerate(atoms):
                rad = ATOM_RADII_CPK[atom] * bs_scale
                if i > 11:
                    rad = rad * 0.75

                pvp.add_mesh(
                    pv.Sphere(center=coords[i], radius=rad),
                    color=ATOM_COLORS[atom],
                    smooth_shading=True,
                    specular=spec,
                    specular_power=specpow,
                )

            pvp = ss._draw_bonds(
                pvp,
                coords,
                style="bs",
                all_atoms=all_atoms,
                bond_radius=bond_radius,
            )

        elif style == "sb":  # splitbonds
            pvp = ss._draw_bonds(
                pvp,
                coords,
                style="sb",
                all_atoms=all_atoms,
                bond_radius=bond_radius,
            )

        elif style == "pd":  # proximal-distal
            pvp = ss._draw_bonds(
                pvp,
                coords,
                style="pd",
                all_atoms=all_atoms,
                bond_radius=bond_radius,
            )

        else:  # plain
            pvp = ss._draw_bonds(
                pvp,
                coords,
                style="plain",
                bcolor=bondcolor,
                all_atoms=all_atoms,
                bond_radius=bond_radius,
            )

        return pvp

    def display(self, style, light="Auto", single=True, winsize=(800, 800)):
        from proteusPy import get_macos_theme

        self.plotter.clear()

        # title = f"{src}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol. Cα: {self.ca_distance:.2f} Å Cβ: {self.cb_distance:.2f} Å Tors: {self.torsion_length:.2f}°"
        title = "Disulfide Bond"

        if light == "Light":
            pv.set_plot_theme("document")
        elif light == "Dark":
            pv.set_plot_theme("dark")
        else:
            _theme = get_macos_theme()
            if _theme == "light":
                pv.set_plot_theme("document")
            elif _theme == "dark":
                pv.set_plot_theme("dark")
            else:
                pv.set_plot_theme("document")

        if single is True:
            self.plotter.add_title(title=title, font_size=FONTSIZE)
            self.plotter.enable_anti_aliasing("msaa")

        if style == "cpk":
            self.plot(style="cpk")
        elif style == "sb":
            self.plot(style="sb")
        elif style == "bs":
            self.plot(style="bs")

        self.plotter_widget.update()


if __name__ == "__main__":
    pdb = Load_PDB_SS(subset=True, verbose=True)
    ss1 = pdb[0]
    app = QApplication(sys.argv)
    viewer = MolecularViewer(ss1)
    viewer.show()
    sys.exit(app.exec_())
