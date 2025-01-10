import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np 

def plot_3d_array(array: np.ndarray, show = True, **kwargs):
    
    """
    array: array to be plotted, shaoe (x,y,z)
    show: whether to render a live plot
    fname: if provided the file will be saved as a pdf
    opacity (list): the rendering isosurfaces 
    
    """
    Nx,Ny,Nz = array.shape

    grid = pv.wrap(array)

    # Add metadata if needed
    grid.dimensions = (Nx, Ny, Nz)
    grid.spacing = (1, 1, 1)

    # Plot
    plotter = pv.Plotter(off_screen=False)
    if "opacity" in kwargs:
        opacity = kwargs["opacity"]
    else:
        opacity = 'linear'  # Fully transparent below threshold, fully opaque above

    plotter.add_volume(grid, cmap='jet', opacity=opacity)
    # Plot the volume
    plotter.show_axes()
    
    if "fname" in kwargs:
        fname = kwargs["fname"]
        
        plotter.save_graphic(f"{fname}.pdf")

    if show:
        plotter.show()

    plotter.close()
    
    
def plot_vecs(kins, kouts, qvecs):
    
    #Function to add vectors to the plot

        # Create a PyVista plotter
        plotter = pv.Plotter(off_screen=False)

        def add_vectors(plotter, start_points, vectors, color, shaft_scale = 0.25, tip_scale=.2):
            
            magnitude = np.linalg.norm(vectors[0])
            
            shaft_radius = shaft_scale * 1.0 / magnitude 
            tip_radius = shaft_radius+tip_scale
            
            for start, vec in zip(start_points, vectors):
                arrow = pv.Arrow(start=start, direction=vec, scale='auto', shaft_radius=shaft_radius, tip_radius=tip_radius)
                plotter.add_mesh(arrow, color=color)

        kins = kins *1e-10#/ np.linalg.norm(kins, axis = 1)[:, np.newaxis]
        kouts = kouts *1e-10#/ np.linalg.norm(kouts, axis = 1)[:, np.newaxis]
        qvecs = qvecs *1e-10#/ np.linalg.norm(qvecs, axis = 1)[:, np.newaxis]
        
        

        # Origin for kins and kouts
        origin = np.array([0, 0, 0])


        X = 500
        
        # Add kins vectors
        add_vectors(plotter, [origin] * len(kins[:X]), kins[:X], "blue", shaft_scale= 0.01, tip_scale=.02)

        # Add kouts vectors
        add_vectors(plotter, [origin] * len(kouts[:X]), kouts[:X], "green", shaft_scale= 0.01,tip_scale=.02)

        # Add qvec vectors (start at kins, end at kouts)
        add_vectors(plotter, [origin] * len(qvecs[:X]), qvecs[:X], "red", shaft_scale= 0.01,tip_scale=.01)
        
        plotter.view_xz()
        plotter.background_color = 'white'
        # Add legend and show
        plotter.add_legend([
            ("kins (blue)", "blue"),
            ("kouts (green)", "green"),
            ("qvec (red)", "red"),
        ])

        # Add axes
        plotter.show_axes()  # Displays x, y, z axes in the 3D plot
        plotter.show()
        plotter.close()

fldr = '/Users/mohahmed/Documents/ConvergentBeam/Codes/Analysis_P11_2024/'
kins = np.load(fldr+'k_ins.npy')
kouts = np.load(fldr+'k_outs.npy')
qvecs = np.load(fldr+'q_vecs.npy')

plot_vecs(kins, kouts, qvecs)