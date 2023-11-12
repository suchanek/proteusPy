
import panel as pn
import pyvista as pv
from pyvista import examples

# Create a PyVista plotter object
plotter = pv.Plotter(notebook=True)

# Load an example mesh
mesh = examples.download_st_helens().warp_by_scalar()

# Add the mesh to the plotter
plotter.add_mesh(mesh)

# Create a Panel pane to display the plotter object
pane = pn.panel(plotter.ren_win, sizing_mode='stretch_width', aspect_ratio=1, orientation_widget=True)

# Define a function to update the plotter object
def update_plot(event):
    plotter.clear()
    plotter.add_mesh(mesh, scalars='Elevation')
    pane.param.trigger('object')

# Create a button that will update the plotter when clicked
button = pn.widgets.Button(name='Update Plot', button_type='primary')
button.on_click(update_plot)

# Create a Panel layout with the button and the plotter
layout = pn.Column(button, pane)

# Show the layout
layout.servable()
