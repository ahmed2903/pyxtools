import matplotlib.pyplot as plt
import numpy as np 
from .utils import time_it
from PIL import Image 
import matplotlib.patches as patches
from IPython.display import display
import ipywidgets as widgets

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
    fig, ax = plt.subplots(1,1 ,figsize=(6, 6))

    # Plot original pixel positions (unordered)
    ax.scatter(ordered_pixels[:, 1], ordered_pixels[:, 0], color='red', label="Pixels")
    if connection:
        ax.plot(ordered_pixels[:, 1], ordered_pixels[:, 0], color='red', label="Pixels")

    ax.set_title("Scatter")
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
        rect_row = idx[0] - 1 - sx  # row-coordinate of the bottom-left corner
        rect_col = idx[1] - 1 - sy # col-coordinate of the bottom-left corner
        rect = patches.Rectangle((rect_col, rect_row), rec_size, rec_size, linewidth=2, edgecolor=color, facecolor=color)
        ax.add_patch(rect)

    im = ax.imshow(detec_image, cmap=cmap,  vmin=vmin, vmax=vmax, origin = 'lower')
    fig.colorbar(im, orientation='vertical')
    plt.title(title)
    if 'fname' in kwargs:
        name = kwargs['fname']
        plt.savefig(name)
    
    plt.show()


def two_lists_one_slider(img_list1, img_list2, scale_fac=0.3, vmin1 = None, vmax1 = None, vmin2=None, vmax2=None, cmap1 = 'viridis', cmap2 = 'viridis'):
    """Displays a list of coherent images and allows scrolling through them via a slider."""
    

    num_images = len(img_list1)  # Number of images in the list
    
    # Create a slider for selecting the image index
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    # Create figure & axis once
    fig, axes = plt.subplots(1,2, figsize=(10, 5))
    
    # Initial image
    im1 = axes[0].imshow(img_list1[0], vmin = vmin1, vmax = vmax1, cmap=cmap1)    
    axes[0].set_title(f"Image 1 {0}/{num_images - 1}")
    #axes[0].axis('off')  # Hide axes
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot the second image
    im2 = axes[1].imshow(img_list2[0], vmin = vmin2, vmax = vmax2, cmap=cmap2)
    axes[1].set_title(f"Image 1 {0}/{num_images - 1}")
    #axes[1].axis('off')  # Hide axes
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    def update_image(img_idx):
        """Updates the displayed image when the slider is moved."""
        img1 = img_list1[img_idx]
        # img_mean1 = np.mean(img1)

        img2 = img_list2[img_idx]
        # img_mean2 = np.mean(img2)
        
        # vmin1 = max(img_mean1 - scale_fac * img_mean1, 0)
        # vmax1 = img_mean1 + scale_fac * img_mean1

        # vmin2 = img_mean2 - scale_fac * img_mean2
        # vmax2 = img_mean2 + scale_fac * img_mean2
        
        im1.set_data(img1)  # Update image data
        im1.set_clim(vmin1, vmax1)

        im2.set_data(img2)  # Update image data
        im2.set_clim(vmin2, vmax2)
        
        axes[0].set_title(f"Image 1 {img_idx}/{num_images - 1}")  # Update title
        axes[1].set_title(f"Image 2 {img_idx}/{num_images - 1}")  # Update title
        fig.canvas.draw_idle()  # Efficient redraw

    # Create interactive slider
    interactive_plot = widgets.interactive(update_image, img_idx=img_slider)

    display(interactive_plot)  # Show slider
    #display(fig)  # Display the figure

def plot_list_slider(img_list, scale_fac=0.3, vmin1 = None, vmax1 = None):
    """Displays a list of coherent images and allows scrolling through them via a slider."""
    

    num_images = len(img_list)  # Number of images in the list
    
    # Create a slider for selecting the image index
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    # Create figure & axis once
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Initial image
    vmin, vmax = np.min(img_list[0]), np.max(img_list[0])  # Normalize color scale
    im = ax.imshow(img_list[0], cmap='viridis', vmin=vmin1, vmax=vmax1)
    ax.set_title(f"Image {0}/{num_images - 1}")
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.tight_layout()
    
    def update_image(img_idx):
        """Updates the displayed image when the slider is moved."""
        img = img_list[img_idx]
        img_mean = np.mean(img)

        vmin1 = max(img_mean - scale_fac * img_mean,0)
        
        vmax1 = img_mean + scale_fac * img_mean
        
        im.set_data(img)  # Update image data
        im.set_clim(vmin1, vmax1)
        
        ax.set_title(f"Image {img_idx}/{num_images - 1}")  # Update title
        fig.canvas.draw_idle()  # Efficient redraw

    # Create interactive slider
    interactive_plot = widgets.interactive(update_image, img_idx=img_slider)

    display(interactive_plot)  # Show slider
    #display(fig)  # Display the figure