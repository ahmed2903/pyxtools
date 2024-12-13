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
    
    
