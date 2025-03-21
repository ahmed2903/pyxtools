import matplotlib.pyplot as plt
import numpy as np 
from utils import time_it
from PIL import Image 
import matplotlib.patches as patches

def plot_roi_from_numpy(array, roi=None, name=None, vmin=None, vmax=None, save = False, **kwargs):

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
    if roi is None:
        roi = [0,-1,0,-1]

    if name is None:
        name = "Plot"

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
        
@time_it
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

def two_lists_one_gif(array_list1, array_list2, gif_name, title1 = "Image 1", title2 = "Image 2", fps=10, cmap="viridis"):
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
    num_images = len(array_list1)

    # Create frames
    for i, (array1, array2) in enumerate(zip(array_list1, array_list2)):

        # Create figure & axis once
        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        
        # Initial image
        im1 = axes[0].imshow(array1, cmap=cmap)    
        #axes[0].axis('off')  # Hide axes
        #plt.colorbar(im1, ax=axes[0]) #, fraction=0.046, pad=0.04)
        
        # Plot the second image
        im2 = axes[1].imshow(array2, cmap=cmap)
        #axes[1].axis('off')  # Hide axes
        #plt.colorbar(im2, ax=axes[1]) #, fraction=0.046, pad=0.04)
        
        axes[0].set_title(title1)
        axes[1].set_title(title2)
        #plt.tight_layout()
        fig.suptitle("Frame {i}/{num_images}", fontsize=16, y=1.05)
        
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
        
def plot_pixel_space(ordered_pixels, connection=False):
    """
    Plots the pixel coordinates before and after reordering.
    
    Args:
        ordered_pixels (list of tuples): Reordered pixel coordinates.
    """
    # Convert lists to NumPy arrays for easy plotting
    ordered_pixels = np.array(ordered_pixels)

    # Create figure
    fig, ax = plt.subplots(1,1 ,figsize=(12, 6))

    # Plot original pixel positions (unordered)
    ax.scatter(ordered_pixels[:, 0], ordered_pixels[:, 1], color='red', label="Pixels")
    if connection:
        ax.plot(ordered_pixels[:, 0], ordered_pixels[:, 1], color='red', label="Pixels")

    ax.set_title("Original Pixel Coordinates")
    ax.invert_yaxis()  # Invert to match image coordinates

    plt.show()
    
def plot_map_on_detector(detec_image, k_map, vmin, vmax, title, cmap, crop=False,**kwargs):

    """

    optional args: 
        color: color of the rectangle
        rec_size: size of the rectangle
        fname: if passed, figure will be saved as the fname
    """

    if 'roi' in kwargs:
        roi = kwargs['roi']
        sx, ex, sy, ey = roi
    else:
        sx = 0
        sy = 0
        
    
    if crop:
        detec_image = detec_image[sx:ex, sy:ey]    
    
        
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