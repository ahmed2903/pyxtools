import matplotlib.pyplot as plt
#import pyvista as pv
import numpy as np 
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
import pyvista as pv
from matplotlib.patches import Rectangle
from IPython.display import display
from PIL import Image 


def create_gif_from_arrays(array_list, gif_name, fps=10, cmap="viridis"):
    """
    Create a GIF from a list of 2D arrays.

    Parameters:
        array_list (list of np.ndarray): List of 2D arrays to plot.
        gif_name (str): Name of the output GIF file (e.g., 'output.gif').
        fps (int): Frames per second for the GIF.
        cmap (str): Colormap for the plots.
    """
    # Store file paths for each frame
    frame_files = []
    
    # Create frames
    for i, array in enumerate(array_list):
        plt.figure(figsize=(6, 6))
        plt.imshow(array, cmap=cmap)
        plt.colorbar()
        plt.title(f"Frame {i + 1}")
        
        # Save the frame
        frame_file = f"frame_{i}.png"
        plt.savefig(frame_file)
        frame_files.append(frame_file)
        plt.close()
    
    # Create the GIF using Pillow
    frames = [Image.open(file) for file in frame_files]
    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,  # in milliseconds
        loop=0
    )
    
    # Cleanup temporary frame files
    for file in frame_files:
        import os
        os.remove(file)

def plot_roi_from_numpy(array, roi, name, vmin=None, vmax=None, save = False, **kwargs):

    """
    Plots a region of interest from a 2D numpy array

    input: 
        array : numpy array (2D or 3D)
        roi : list, Region of interest, [min vertical axis, max vertical axis, min horizontal axis, max horizontal axis] 
        name: str, name to be used for title and saving the figure
        vmin: float, color map minimum
        vmax: float, color map maximum
        save: bool, if True, saves the plot
        frame: int, frame number if array is 3D 
    """

    # Plot the pupil
    if len(array.shape) == 3 and 'frame' in kwargs:
        frame = kwargs['frame']
        data_roi = array[frame, roi[0]:roi[1], roi[2]:roi[3]]
    elif len(array.shape) == 2:
        data_roi = array[roi[0]:roi[1], roi[2]:roi[3]]
    else:
        raise ValueError("Either pass in a 2D array or a 3D array with 'frame' in the kwargs.")

    if vmin is None:
        me = np.mean(data_roi) 
        vmin = me - 0.5 *me
    if vmax is None:
        me = np.mean(data_roi) 
        vmax = me + 0.5 *me
        
    plt.figure()
    plt.imshow(data_roi, vmin = vmin, vmax = vmax,cmap='viridis')
    plt.title(name)
    plt.colorbar()
    plt.show()
    
    if save:
        plt.savefig(name)

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

def plot_map_on_detector(detec_image, k_map, vmin, vmax, title, cmap, **kwargs):

    """

    optional args: 
        color: color of the rectangle
        rec_size: size of the rectangle
        fname: if passed, figure will be saved as the fname
    """

    if 'roi' in kwargs:
        roi = kwargs['roi']
        sx, ex, sy, ey = roi
        detec_image = detec_image[sx:ex, sy:ey]
        
        
    else:
        sx = 0
        sy = 0
        
    fig, ax = plt.subplots()

    if 'color' in kwargs:
        color = kwargs['color']
    else: 
        color = 'white'

    if 'rec_size' in kwargs:
        rec_size = kwargs['rec_size']
    else: 
        rec_size = 1
        
    # Add a white rectangle at each of the indices
    for idx in k_map:
        rect_x = idx[1] - 1 - sy  # X-coordinate of the bottom-left corner
        rect_y = idx[0] - 1 - sx # Y-coordinate of the bottom-left corner
        rect = patches.Rectangle((rect_x, rect_y), rec_size, rec_size, linewidth=2, edgecolor=color, facecolor=color)
        ax.add_patch(rect)

    ax.imshow(detec_image, cmap=cmap,  vmin=vmin, vmax=vmax, origin = 'lower')
    plt.title(title)
    if 'fname' in kwargs:
        name = kwargs['fname']
        plt.savefig(name)
    
    plt.show()

from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output

def initialize_live_plot(hr_obj_image):
    """
    Initializes the live plot with two subplots: one for amplitude and one for phase.
    
    Returns:
        fig, ax: Matplotlib figure and axes.
        img_amp, img_phase: Image objects for real-time updates.
    """

    # Initialize empty images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
    # Initialize the plots with the initial image
    img_amp = axes[0].imshow(np.abs(hr_obj_image), vmin =.2, vmax = 1, cmap='viridis')
    axes[0].set_title("Amplitude of Object")
    plt.colorbar(img_amp, ax=axes[0])

    img_phase = axes[1].imshow(np.angle(hr_obj_image), vmin = -np.pi, vmax = np.pi, cmap='viridis')
    axes[1].set_title("Phase of Object")
    plt.colorbar(img_amp, ax=axes[1])

    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    plt.show()

    return fig, axes, img_amp, img_phase

def update_live_plot(img_amp, img_phase, hr_obj_image, fig):
    """
    Updates the live plot with new amplitude and phase images.

    Args:
        img_amp: Matplotlib image object for amplitude.
        img_phase: Matplotlib image object for phase.
        hr_obj_image: The complex object image to be plotted.
    """
    amplitude = np.abs(hr_obj_image)
    phase = np.angle(hr_obj_image)
    
    img_amp.set_data(amplitude)  # Normalize for visibility
    img_phase.set_data(phase)

    amp_mean = np.mean(amplitude)
    vmin = amp_mean - 0.1 * amp_mean
    vmax = amp_mean + 0.2 * amp_mean
    
    #img_amp.autoscale()  # Reset autoscaling
    img_amp.set_clim(vmin, vmax)
    #img_phase.auto_scale()
    
    #plt.colorbar(img_amp, ax=axes[0])
    #clear_output(wait=True)
    display(fig)
    #fig.canvas.draw()
    fig.canvas.flush_events()
    
def plot_images_side_by_side(image1, image2, 
                             vmin1= None, vmax1=None, 
                             vmin2= None, vmax2=None, 
                             title1="Image 1", title2="Image 2", cmap1="gray", cmap2="gray", figsize=(10, 5), show = False):
    """
    Plots two images side by side.

    Parameters:
    - image1: First image (2D numpy array).
    - image2: Second image (2D numpy array).
    - title1: Title for the first image (default: "Image 1").
    - title2: Title for the second image (default: "Image 2").
    - cmap1: Colormap for the first image (default: "gray").
    - cmap2: Colormap for the second image (default: "gray").
    - figsize: Size of the figure (default: (10, 5)).
    """
    # Create a figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    
    if vmin1 is None:
        me = np.mean(image1) 
        vmin1 = me - 0.5 *me
    if vmax1 is None:
        me = np.mean(image1) 
        vmax1 = me + 0.5 *me
    
    if vmin2 is None:
        me = np.mean(image2) 
        vmin2 = me - 0.5 *me
    if vmax2 is None:
        me = np.mean(image2) 
        vmax2 = me + 0.5 *me
    
    # Plot the first image
    im1 = axes[0].imshow(image1, vmin = vmin1, vmax = vmax1, cmap=cmap1)    
    axes[0].set_title(title1)
    #axes[0].axis('off')  # Hide axes
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot the second image
    im2 = axes[1].imshow(image2, vmin = vmin2, vmax = vmax2, cmap=cmap2)
    axes[1].set_title(title2)
    #axes[1].axis('off')  # Hide axes
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Adjust layout and display
    plt.tight_layout()
    if show:
        plt.show()
        