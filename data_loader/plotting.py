import matplotlib.pyplot as plt
import numpy as np 
from .utils import time_it
from PIL import Image 
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from IPython.display import display
import ipywidgets as widgets
import os
import h5py
from .cbd_loader import ROI
from .data_fs import mask_hot_pixels, make_coherent_image, make_detector_image, stack_4d_data_old, compute_histograms

def plot_detector_roi(roi:ROI, file_no, frame_no, title=None, vmin=None, vmax=None,mask_hot = True, save=False):
    """Plots a region of interest (ROI) on the detector for a given frame.

    Args:
        roi_name (str): The name of the region of interest (ROI) to be plotted.
        file_no (int): The file number to retrieve the frame from.
        frame_no (int): The frame number within the specified file.
    """
    direc = roi.data_dir
    
    if roi.beamtime == 'new':
        file_no_st = (6-len(str(file_no)))*'0' + str(file_no)
        
        file_name = os.path.join(direc,f'Scan_{roi.scan_num}_data_{file_no_st}.h5')

        with h5py.File(file_name,'r') as f:
            data_test = f['/entry/data/data'][frame_no,:,:]
    else:
        file_no_st = '0'*5 + str(1)
        filename = os.path.join(direc,f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
        data_test = stack_4d_data_old(filename, roi.coords, roi.fast_axis_steps, roi.slow_axis)[file_no,frame_no, :,:]

    if mask_hot:
        data_test = mask_hot_pixels(data_test)
    if title is None:
        title = f"Detector image in Frame ({file_no}, {frame_no})"
    plot_roi_from_numpy(data_test, roi.coords, title, vmin=vmin, vmax=vmax, save = save)



def plot_full_detector(roi:ROI, file_no, frame_no, 
                      vmin1=None, vmax1=None, 
                      vmin2=None, vmax2=None):
    
    """
    Plots a full frame of the detector.
    Args:
        file_no (int): The file number to retrieve the frame from.
        frame_no (int): The frame number within the specified file.
    """
    direc = roi.data_dir
    try:
        file_no_st = (6-len(str(file_no)))*'0' + str(file_no)
        
        file_name = os.path.join(direc , f'Scan_{roi.scan_num}_data_{file_no_st}.h5')

        with h5py.File(file_name,'r') as f:
            data_test = f['/entry/data/data'][frame_no,:,:]
    except:
        file_no_st = '0'*5 + str(1)
        filename = os.path.join(direc,f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
        data_test = stack_4d_data_old(filename, [0,-1,0,-1], roi.fast_axis_steps, roi.slow_axis)[file_no,frame_no, :,:]

    data_test = mask_hot_pixels(data_test)
    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    log_data = np.log1p(data_test)

    intensity = axes[0].imshow(data_test, cmap='jet',  vmin = vmin1, vmax = vmax1)
    axes[0].set_title(f'Full Detector')
    axes[0].axis('on')  
    plt.colorbar(intensity, ax=axes[0], fraction=0.046, pad=0.04)
    log_intensity = axes[1].imshow(log_data, cmap='jet',  vmin=vmin2, vmax=vmax2)
    axes[1].set_title(f'Log Scale')
    axes[1].axis('on')
    plt.colorbar(log_intensity, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

    
def plot_4d_dataset(roi : ROI, vmin1 = None, vmax1 = None,vmin2 = None, vmax2 = None, pixel_threshold=None):
    """Plots the 4D dataset for a specified region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) to visualize. The ROI 
                         should be present in the `self.ptychographs` dictionary.
    """
    # Get dataset dimensions
    coherent_shape = roi.data_4d.shape[:2]  
    detector_shape = roi.data_4d.shape[2:]  

    
    
    # Set slider limits
    pcol_slider = widgets.IntSlider(min=0, max= detector_shape[1] - 1, value=detector_shape[1]//2, description="px")
    prow_slider = widgets.IntSlider(min=0, max= detector_shape[0] - 1, value=detector_shape[0]//2, description="py")
    
    lcol_slider = widgets.IntSlider(min=0, max= coherent_shape[1] - 1, value=coherent_shape[1]//2, description="lx")
    lrow_slider = widgets.IntSlider(min=0, max= coherent_shape[0] - 1, value=coherent_shape[0]//2, description="ly")


    rectangle_size_det = 4 
    rectangle_size_coh = 2
    
    # Create the figure and axes **only once**
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    coherent_image = make_coherent_image(roi.data_4d, np.array([prow_slider.value, pcol_slider.value]))
    detector_image = make_detector_image(roi.data_4d, np.array([lrow_slider.value, lcol_slider.value]))

    vmin_c = vmin1 #np.mean(coherent_image) - 0.05 * np.mean(coherent_image)
    vmax_c = vmax1 #np.mean(coherent_image) + 0.05 * np.mean(coherent_image)
    vmin_d = vmin2 #np.mean(detector_image) - 0.3 * np.mean(detector_image)
    vmax_d = vmax2 #np.mean(detector_image) + 2 * np.mean(detector_image)
    
    im0 = axes[0].imshow(coherent_image, cmap='plasma', vmin=vmin_c, vmax=vmax_c)
    
    axes[0].set_title(f"Coherent Image (lx={lrow_slider.value}, ly={lcol_slider.value})")
    rect_coherent = Rectangle(( lcol_slider.value - rectangle_size_coh / 2, lrow_slider.value - rectangle_size_coh / 2), 
                              rectangle_size_coh, rectangle_size_coh, 
                              edgecolor='white', facecolor='none', lw=2)
    
    axes[0].add_patch(rect_coherent)
    plt.colorbar(im0, ax=axes[0], label="Intensity")

    im1 = axes[1].imshow(detector_image, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    axes[1].set_title(f"Detector Image (px={pcol_slider.value}, py={prow_slider.value})")
    rect_detector = Rectangle((pcol_slider.value - rectangle_size_det / 2, prow_slider.value - rectangle_size_det / 2), 
                              rectangle_size_det, rectangle_size_det, 
                              edgecolor='white', facecolor='none', lw=2)
    axes[1].add_patch(rect_detector)
    plt.colorbar(im1, ax=axes[1], label="Detector Intensity")

    plt.tight_layout()
    
    def update_plot(prow, pcol, lrow, lcol):
        """ Updates the plot based on slider values. """
        coherent_image = make_coherent_image(roi.data_4d, np.array([prow, pcol]))
        detector_image = make_detector_image(roi.data_4d, np.array([lrow, lcol]))

        vmin_c = np.mean(coherent_image) - 0.05 * np.mean(coherent_image)
        vmax_c = np.mean(coherent_image) + 0.05 * np.mean(coherent_image)
        vmin_d = np.mean(detector_image) - 0.3 * np.mean(detector_image)
        vmax_d = np.mean(detector_image) + .2 * np.mean(detector_image)

        im0.set_data(coherent_image)
        im1.set_data(detector_image)

        axes[0].set_title(f"Coherent Image from Pixel ({pcol}, {prow})")
        im0.set_clim(vmin_c, vmax_c)
        axes[1].set_title(f'Detector Image at Location ({lcol},{lrow})')
        im1.set_clim(vmin_d, vmax_d)
        
        # Update rectangles
        rect_coherent.set_xy((lcol - rectangle_size_coh / 2, lrow - rectangle_size_coh / 2))
        rect_detector.set_xy((pcol - rectangle_size_det / 2, prow - rectangle_size_det / 2))


        fig.canvas.draw_idle()
        
    # Create interactive widget
    interactive_plot = widgets.interactive(update_plot, prow=prow_slider, pcol=pcol_slider, lrow=lrow_slider, lcol=lcol_slider)
    
    display(interactive_plot)  # Display the interactive widget

def plot_coherent_sequence(roi: ROI, scale_factor = .4, vmin = None, vmax = None):
    """Displays a sequence of coherent images and allows scrolling through them via a slider.

    Args:
        roi_name (str): The name of the region of interest (ROI) to visualize. The ROI 
                         should be present in the `self.coherent_imgs` dictionary.
        scale_factor (float, optional): A factor to scale the maximum color intensity 
                                         for the images. Default is 0.4.
    """
        
    img_list = roi.coherent_imgs  # List of coherent images

    num_images = len(img_list)  # Number of images in the list
    
    # Create a slider for selecting the image index
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")
    
    vmin_box =  widgets.FloatText(value=.1,description='Vmin:',disabled=False)
    
    vmax_box = widgets.FloatText(value=1,description='Vmax:',disabled=False)
    
    
    # Create figure & axis once
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img_list[0], cmap='viridis') #, vmin=vmin, vmax=vmax)
    ax.set_title(f"Coherent Image {0}/{num_images - 1}")
    plt.colorbar(im, ax=ax, label="Intensity")
    ax.set_ylabel('Scan-Y')
    ax.set_xlabel('Scan-X')
    plt.tight_layout()
    
    def update_image(img_idx, vmin, vmax):
        """Updates the displayed image when the slider is moved."""
        img = img_list[img_idx]
        
        im.set_data(img)  # Update image data
        im.set_clim(vmin, vmax)
        
        ax.set_title(f"Coherent Image {img_idx}/{num_images - 1}")  # Update title
        fig.canvas.draw_idle()  # Efficient redraw

    # Create interactive slider
    interactive_plot = widgets.interactive(update_image, img_idx=img_slider, vmin = vmin_box, vmax = vmax_box)

    display(interactive_plot)  # Show slider
    #display(fig)  # Display the figure


def plot_averag_coh_imgs(roi:ROI, vmin=None, vmax=None, cmap = 'viridis', title=None):
    """Plots the average of coherent images for a given region of interest (ROI).

    Args:
        roi: The name of the region of interest (ROI) for which the 
                         average coherent image will be plotted.
    """
    avg = roi.averaged_coherent_images
    #avg = np.mean(np.array(self.coherent_imgs[roi_name]), axis = 0)

    if title is None:
        title = f"Average Coherent Images: Scan {roi.scan_num}"
        
    # plot_roi_from_numpy(avg, name=title, vmin=vmin, vmax=vmax)

    if roi.slow_axis == 'scan_x':
        x_add = "Slow Axis"
        y_add = "Fast Axis"

    elif roi.slow_axis == 'scan_y':
        x_add = "Fast Axis"
        y_add = "Slow Axis"
    plt.figure()
    plt.imshow(avg, vmin = vmin, vmax = vmax, cmap=cmap)
    plt.title(title)
    
    
        
    plt.xlabel(': '.join(('Scan-X', x_add)))
    plt.ylabel(': '.join(('Scan-Y', y_add)))
    
    plt.colorbar()
    plt.show()
    
def plot_detected_objects(roi:ROI):
    """Displays the detected objects in coherent images with a slider for navigation.

    Args:
        roi_name (str): The name of the region of interest (ROI) containing the detected objects.

    """
    
    img_list = roi.detected_objects  # List of coherent images

    num_images = len(img_list)  # Number of images in the list
    
    # Create a slider for selecting the image index
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

    # Create figure & axis once
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Initial image
    vmin, vmax = 0,1 # Normalize color scale
    im = ax.imshow(img_list[0], cmap='viridis') #, vmin=vmin, vmax=vmax)
    ax.set_title(f"Coherent Image {0}/{num_images - 1}")
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.tight_layout()
    
    def update_image(img_idx):
        """Updates the displayed image when the slider is moved."""
        img = img_list[img_idx]
        im.set_data(img)  # Update image data
        
        ax.set_title(f"Coherent Image {img_idx}/{num_images - 1}")  # Update title
        fig.canvas.draw_idle()  # Efficient redraw

    # Create interactive slider
    interactive_plot = widgets.interactive(update_image, img_idx=img_slider)

    display(interactive_plot)  # Show slider
    #display(fig)  # Display the figure
    

def plot_intensity_histograms(roi:ROI, bins = 256):
    """Displays intensity histograms of images in a given ROI and allows scrolling through them via a slider.

    Args:
        roi_name (str): The name of the region of interest (ROI) for which histograms 
                         are computed.
        bins (int, optional): The number of bins to use for the histogram computation. 
                               Default is 256.

    """
    histograms = compute_histograms(roi.coherent_imgs, bins=bins)

    num_images = len(histograms)  # Number of images in the list
    
    # Create a slider for selecting the image index
    img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")
    
    # Create figure & axis once
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Initial image
    line, = ax.plot(histograms[0])  
    ax.set_yscale("log")  # Set y-axis to log scale
    ax.set_title(f"Intensity Histogram {0}/{num_images - 1}")
    ax.set_ylabel("Log Frequency")
    plt.tight_layout()
    
    def update_image(img_idx):
        """Updates the displayed image when the slider is moved."""
        line.set_ydata(histograms[img_idx])  # Update image data
        
        ax.set_title(f"Intensity historgram{img_idx}/{num_images - 1}")  # Update title
        fig.canvas.draw_idle()  # Efficient redraw
    
    # Create interactive slider
    interactive_plot = widgets.interactive(update_image, img_idx=img_slider)
    
    display(interactive_plot)  # Show slider
    

def plot_average_roi(roi:ROI, vmin=None, vmax=None, title=None):
    """Plots the averaged frames for the specified region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose averaged data 
                         is to be plotted. The averaged data should be stored in 
                         `self.averaged_det_data`.
    """
    
    plot_roi_from_numpy(roi.averaged_det_images, [0,-1,0,-1], 
                        f"Averaged Frames ", 
                        vmin=vmin, vmax=vmax )


def plot_calculated_kins(roi:ROI, pupil_roi:ROI, vmin=None, vmax = None, title="Mapped kins onto pupil", cmap = "viridis"):
    """Plots the calculated k-in coordinates mapped onto the pupil function.
    Args:
        roi_name (str): The name of the region of interest.

    """
    plot_map_on_detector(pupil_roi.averaged_det_images, roi.kin_coords, 
                         vmin, vmax, title, cmap, crop=False, roi= pupil_roi.coords)

def plot_kouts(roi:ROI, vmin=None, vmax = None, title="Mapped kouts", cmap = "viridis"):
    
    """Plots the calculated k-out coordinates mapped onto the detector.

    Args:
        roi_name (str): The name of the region of interest.

    """
    plot_map_on_detector(roi.averaged_det_images, roi.kout_coords, 
                         vmin, vmax, title, cmap, crop=False,roi= roi.coords)



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

        img2 = img_list2[img_idx]
 
        
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