# Imports data from Ptychography experiments
# 
# Author: Ahmed H. Mokhtar 
# Email: ahmed.mokhtar@desy.de
# Date : Feb 2025

import numpy as np 
import h5py
import matplotlib.pyplot as plt
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.widgets import Button, TextBox
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display

from ..data_fs import list_datafiles, make_2dimensions_even, stack_4d_data, make_coherent_image, make_detector_image, average_data, mask_hot_pixels, make_coordinates, filter_images, load_hdf_roi
from ..xrays_fs import compute_vectors
from  ..plotting_fs import plot_roi_from_numpy

class load_data:
    
    def __init__(self, directory:str, scan_num: int, 
                 
                 det_psize:float, det_distance:float, centre_pixel:tuple[int], wavelength:float, slow_axis:int):
        
        self.det_psize = det_psize
        self.det_distance = det_distance
        self.centre_pixel = centre_pixel
        self.wavelength = wavelength 
        self.slow_axis = slow_axis
        
        self.dir = f"{directory.rstrip('/')}/Scan_{scan_num}/" 
        self.scan_num = scan_num
        self.rois_dict = {}
        self.ptychographs = {}
        self.averaged_data = {}
        
        self.coords = {}
        self.kouts = {}
        self.coherent_imgs = {}

        self.fnames = list_datafiles(self.dir)[:-2]        
        

    def make_4d_dataset(self, roi_name: str):
        
        """
        Makes the 4D data set around from a ROI on detector.
        """
        self.ptychographs[roi_name] = stack_4d_data(self.dir, self.fnames, self.rois_dict[roi_name], conc=True)
        
        self.ptychographs[roi_name] = mask_hot_pixels(self.ptychographs[roi_name])
    def plot_full_detector(self, file_no, frame_no, 
                          
                          vmin1=None, vmax1=None, 
                          vmin2=None, vmax2=None):
        """
        Plots one full frame of the detector
        """
        file_no_st = (6-len(str(file_no)))*'0' + str(file_no)
        
        file_name = self.dir+f'Scan_{self.scan_num}_data_{file_no_st}.h5'

        with h5py.File(file_name,'r') as f:
            data_test = f['/entry/data/data'][frame_no,:,:]
        
        fig, axes = plt.subplots(1, 2, figsize=(10,5))

        log_data = np.log1p(data_test)

        axes[0].imshow(data_test, cmap='jet',  vmin = vmin1, vmax = vmax1)
        axes[0].set_title(f'Full Detector in Frame ({file_no},{frame_no})')
        axes[0].axis('on')  

        axes[1].imshow(log_data, cmap='jet',  vmin=vmin2, vmax=vmax2)
        axes[1].set_title(f'Log Scale in Frame ({file_no},{frame_no})')
        axes[1].axis('on')

        plt.tight_layout()
        
        
    def plot_detector_roi(self, roi_name, file_no, frame_no, title=None, vmin=None, vmax=None,mask_hot = False, save=False):
        """
        Plots a detector roi in one frame
        """
        
        file_no_st = (6-len(str(file_no)))*'0' + str(file_no)
        
        file_name = self.dir+f'Scan_{self.scan_num}_data_{file_no_st}.h5'

        xs,xe,ys,ye = self.rois_dict[roi_name]
        with h5py.File(file_name,'r') as f:
            data_test = f['/entry/data/data'][frame_no,xs:xe,ys:ye]

        if mask_hot:

            data_test = mask_hot_pixels(data_test)
        if title is None:
            title = f"Detector image at {roi_name} in Frame ({file_no}, {frame_no})"
        plot_roi_from_numpy(data_test, [0,-1,0,-1], title, vmin=vmin, vmax=vmax, save = save)
        
    
    def add_roi(self, roi_name:str , roi:list):
        """
        Add ROI to the dictionary
        
        {name: [xstart, xend, ystart, yend]}
        """
        self.rois_dict[roi_name] = roi 
     
        
    def plot_4d_dataset(self, roi_name: str):
        # Get dataset dimensions
        coherent_shape = self.ptychographs[roi_name].shape[:2]  
        detector_shape = self.ptychographs[roi_name].shape[2:]  

        
        
        # Set slider limits
        px_slider = widgets.IntSlider(min=0, max= detector_shape[1] - 1, value=detector_shape[0]//2, description="px")
        py_slider = widgets.IntSlider(min=0, max= detector_shape[0] - 1, value=detector_shape[1]//2, description="py")
        lx_slider = widgets.IntSlider(min=0, max= coherent_shape[1] - 1, value=coherent_shape[1]//2, description="lx")
        ly_slider = widgets.IntSlider(min=0, max= coherent_shape[0] - 1, value=coherent_shape[1]//2, description="ly")


        rectangle_size_det = 4# Adjust the rectangle size as needed
        rectangle_size_coh = 2
        
        # Create the figure and axes **only once**
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        coherent_image = make_coherent_image(self.ptychographs[roi_name], np.array([px_slider.value, py_slider.value]))
        detector_image = make_detector_image(self.ptychographs[roi_name], np.array([lx_slider.value, ly_slider.value]))

        vmin_c = np.mean(coherent_image) - 0.05 * np.mean(coherent_image)
        vmax_c = np.mean(coherent_image) + 0.05 * np.mean(coherent_image)
        vmin_d = np.mean(detector_image) - 0.3 * np.mean(detector_image)
        vmax_d = np.mean(detector_image) + 2 * np.mean(detector_image)

        im0 = axes[0].imshow(coherent_image, cmap='plasma', vmin=vmin_c, vmax=vmax_c)
        axes[0].set_title(f"Coherent Image (lx={lx_slider.value}, ly={ly_slider.value})")
        rect_coherent = Rectangle((lx_slider.value - rectangle_size_coh / 2, ly_slider.value - rectangle_size_coh / 2), 
                                  rectangle_size_coh, rectangle_size_coh, 
                                  edgecolor='white', facecolor='none', lw=2)
        axes[0].add_patch(rect_coherent)
        plt.colorbar(im0, ax=axes[0], label="Intensity")

        im1 = axes[1].imshow(detector_image, cmap='plasma', vmin=vmin_d, vmax=vmax_d)
        axes[1].set_title(f"Detector Image (px={px_slider.value}, py={py_slider.value})")
        rect_detector = Rectangle((px_slider.value - rectangle_size_det / 2, py_slider.value - rectangle_size_det / 2), 
                                  rectangle_size_det, rectangle_size_det, 
                                  edgecolor='white', facecolor='none', lw=2)
        axes[1].add_patch(rect_detector)
        plt.colorbar(im1, ax=axes[1], label="Detector Intensity")

        plt.tight_layout()
        
        def update_plot(px, py, lx, ly):
            """ Updates the plot based on slider values. """
            coherent_image = make_coherent_image(self.ptychographs[roi_name], np.array([px, py]))
            detector_image = make_detector_image(self.ptychographs[roi_name], np.array([lx, ly]))

            vmin_c = np.mean(coherent_image) - 0.05 * np.mean(coherent_image)
            vmax_c = np.mean(coherent_image) + 0.05 * np.mean(coherent_image)
            vmin_d = np.mean(detector_image) - 0.3 * np.mean(detector_image)
            vmax_d = np.mean(detector_image) + .2 * np.mean(detector_image)

            im0.set_data(coherent_image)
            im1.set_data(detector_image)

            axes[0].set_title(f"Coherent Image from Pixel ({px}, {py})")
            im0.set_clim(vmin_c, vmax_c)
            axes[1].set_title(f'Detector Image at Location ({lx},{ly})')
            im1.set_clim(vmin_d, vmax_d)
            
            # Update rectangles
            rect_coherent.set_xy((lx - rectangle_size_coh / 2, ly - rectangle_size_coh / 2))
            rect_detector.set_xy((px - rectangle_size_det / 2, py - rectangle_size_det / 2))


            fig.canvas.draw_idle()
            
        # Create interactive widget
        interactive_plot = widgets.interactive(update_plot, px=px_slider, py=py_slider, lx=lx_slider, ly=ly_slider)
        
        display(interactive_plot)  # Display the interactive widget
        
    def plot_coherent_sequence(self, roi_name: str):
        """Displays a list of coherent images and allows scrolling through them via a slider."""
        
        img_list = self.coherent_imgs[roi_name]  # List of coherent images

        num_images = len(img_list)  # Number of images in the list
        
        # Create a slider for selecting the image index
        img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

        # Create figure & axis once
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Initial image
        vmin, vmax = np.min(img_list[0]), np.max(img_list[0])  # Normalize color scale
        im = ax.imshow(img_list[0], cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_title(f"Coherent Image {0}/{num_images - 1}")
        plt.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout()
        
        def update_image(img_idx):
            """Updates the displayed image when the slider is moved."""
            img = img_list[img_idx]
            img_mean = np.mean(img)
            vmin = img_mean - 0.05 * img_mean
            vmax = img_mean + 0.05 * img_mean
            
            im.set_data(img)  # Update image data
            im.set_clim(vmin, vmax)
            
            ax.set_title(f"Coherent Image {img_idx}/{num_images - 1}")  # Update title
            fig.canvas.draw_idle()  # Efficient redraw

        # Create interactive slider
        interactive_plot = widgets.interactive(update_image, img_idx=img_slider)

        display(interactive_plot)  # Show slider
        #display(fig)  # Display the figure
        
    def average_frames_roi(self, roi_name):
        
        self.averaged_data[roi_name] = average_data(self.dir, self.fnames, self.rois_dict[roi_name], conc=True)
        self.averaged_data[roi_name] =  mask_hot_pixels(self.averaged_data[roi_name])
    
    def mask_roi(self, roi_name, hot_pixels = True, mask_val=1):
        
        if hot_pixels:
            self.ptychographs[roi_name] = mask_hot_pixels(self.ptychographs[roi_name])
            
    def plot_average_roi(self, roi_name, vmin=None, vmax=None, title=None):
            
            plot_roi_from_numpy(self.averaged_data[roi_name], [0,-1,0,-1], f"Averaged Frames for {roi_name}", vmin=vmin, vmax=vmax )
            
    def make_coherent_images(self, roi_name):
        
        """
        makes a list of coherent images for a given roi from all coordinates.
        """
        
        coherent_imgs = []
        for i, coord in enumerate(self.coords[roi_name]):

            xp =  coord[0] - self.rois_dict[roi_name][0]
            yp =  coord[1] - self.rois_dict[roi_name][2]
            
            coh_img = make_coherent_image(self.ptychographs[roi_name], np.array([xp,yp]))

            coherent_imgs.append(coh_img)
    
        self.coherent_imgs[roi_name] = coherent_imgs
        
    def even_dims_cohimages(self, roi_name):
        """
        Makes the dimensions of the coherent images even
        """
        
        self.coherent_imgs[roi_name] = make_2dimensions_even(self.coherent_imgs[roi_name])
        
    def filter_coherent_images(self, roi_name:str, variance_threshold):
        
        
        cleaned_coh_images, cleaned_coords, cleaned_kxky = filter_images(self.coherent_imgs[roi_name], 
                                                                         coords = self.coords[roi_name],
                                                                         kin_coords=self.kouts[roi_name], 
                                                                         variance_threshold = variance_threshold)
    

        self.coherent_imgs[roi_name] = cleaned_coh_images
        self.kouts[roi_name] = cleaned_kxky
        self.coords[roi_name] = cleaned_coords
        
        
    def make_kvector(self, roi_name, mask_val):
        
        self.coords[roi_name] = make_coordinates(self.averaged_data[roi_name], mask_val, self.rois_dict[roi_name], crop=False)
        
        self.kouts[roi_name] = compute_vectors(self.coords[roi_name], self.det_distance, self.det_psize, self.centre_pixel, self.wavelength)
    
    
    def prepare_roi(self, roi_name:str, mask_val: float, variance_threshold:float):
        """
        full preparating of the roi, after running add_roi.
        roi_name (string): name of the roi. 
        mask_val (float): masks values on detector to include in the coords array.
        variance_threshold (float): value of the threshold for filtering the coherent images. 
        """
        self.make_4d_dataset(roi_name=roi_name)
        self.average_frames_roi(roi_name=roi_name)
        self.make_kvector(roi_name=roi_name,mask_val= mask_val)
        self.make_coherent_images(roi_name=roi_name)
        self.filter_coherent_images(roi_name=roi_name, variance_threshold=variance_threshold)
        self.even_dims_cohimages(roi_name=roi_name)
        
    
