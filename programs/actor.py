import pyvista as pv

metal = pv.Property(
    show_edges=False,
    
    interpolation='Physically based rendering',
    metallic=1,
    roughness=0.0,
    specular=1,
    specular_power=100,
    color='r'
)

fuzzy = pv.Property(
    show_edges=False,
    interpolation='Physically based rendering',
    metallic=0.0,
    roughness=1,
)
actor = pv.Actor()

pl = pv.Plotter()
actor = pl.add_mesh(pv.Sphere(center=(-.5,0,0), radius=1), smooth_shading=True, show_edges=False)
actor.prop = pv.Property(
    color='r',
    show_edges=False,
    interpolation='Physically based rendering',
    metallic=.5,
    roughness=0.4
)

actor2 = pl.add_mesh(pv.Sphere(center=(.5,0,0), radius=1), color='blue', smooth_shading=True)
actor2.prop = pv.Property(
    color='r',
    show_edges=False,
    interpolation='Physically based rendering',
    metallic=0,
    roughness=0.4
)

pl.show()