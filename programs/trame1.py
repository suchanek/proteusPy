from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout

import pyvista as pv
from pyvista.trame.ui import plotter_ui

# Always set PyVista to plot off screen with Trame
pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

mesh = pv.Wavelet()

pl = pv.Plotter()
pl.add_mesh(mesh)

with SinglePageLayout(server) as layout:
    with layout.content:
        # Use PyVista's Trame UI helper method
        #  this will add UI controls
        view = plotter_ui(pl)

server.start()
