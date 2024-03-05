import panel as pn
import pyvista as pv

# Create a PyVista plotter object
plotter = pv.Plotter()

# Add a mesh to the plotter
mesh = pv.Sphere()
plotter.add_mesh(mesh)


def update_plotter(event):
    # Update the plotter
    mesh.points[:, 0] += 0.3
    plotter.update()


# Create a VTK pane
vtk_pane = pn.pane.VTK(plotter.ren_win)

# Create a button to update the plotter
button = pn.widgets.Button(name="Update")
button.on_click(update_plotter)

# Create a Panel app
app = pn.Column(vtk_pane, button)

# Serve the app
app.servable()
