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

from ..data_fs import list_datafiles, stack_4d_data, make_coherent_image, make_detector_image, average_data, mask_hot_pixels, make_coordinates, filter_images
from ..xrays_fs import compute_vectors
from  ..plotting_fs import plot_roi_from_numpy

class load_data:
    
    def __init__(self, directory , scan_num, 
                 
                 det_psize, det_distance, centre_pixel, wavelength, slow_axis):
        
        self.det_psize = det_psize
        self.det_psize = det_distance
        self.centre_pixel = centre_pixel
        self.wavelength = wavelength 
        self.slow_axis = slow_axis
        
        self.dir = directory
        self.scan_num = scan_num
        self.rois_dict = {}
        self.ptychographs = {}
        self.average_data = {}
        
        self.coords = {}
        self.kouts = {}
        self.cohernt_imgs = {}
      
      
    def get_file_names(self):
        
        self.fnames = list_datafiles(self.dir)[:-2]
    
    def load_files(self):
        
        pass 
    
    def make_4d_dataset(self, roi_name: str):
        
        """
        Makes the 4D data set around from a ROI on detector.
        """
        self.ptychographs[roi_name] = stack_4d_data(self.dir, self.fnames, self.rois_dict[roi_name], conc=True)
        

    def plot_full_detector(self, file_no, frame_no ):
        """
        Plots one full frame of the detector
        """
        frame_no = (6-len(str(frame_no)))*'0' + str(frame_no)
        
        file_name = self.dir+f'Scan_{self.scan_num}_data_{file_no}.h5'

        with h5py.File(file_name,'r') as f:
            data_test = f['/entry/data/data'][frame_no,:,:]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,5))

        log_data = np.log1p(data_test)

        axes[0].imshow(data_test, cmap='jet',  vmin = 0, vmax = 3)
        axes[0].set_title('Full Detector in One Frame')
        axes[0].axis('on')  

        axes[1].imshow(log_data, cmap='jet',  vmin=0, vmax=0.5)
        axes[1].set_title('Log Scale in One Frame')
        axes[1].axis('on')

        plt.tight_layout()
        
        
    def plot_detector_roi(self, roi_name:str, title:str, vmin, vmax, save=False):
        """
        Plots a detector roi
        """
        
        #plot_roi_from_numpy(averaged_data, all_detector, title, vmin=vmin, vmax=vmax, save = save)
        
        pass
    
    def add_roi(self, roi_name:str , roi:list):
        """
        Add ROI to the dictionary
        
        {name: [xstart, xend, ystart, yend]}
        """
        self.rois_dict[roi_name] = roi 
     
    def plot_4d_dataset(self, roi_name:str):
        
        px = 20 #pxs[0] 
        py = 20 #pys[0]

        lx=20
        ly=27

        rectangle_size = 3

        coherent_image = make_coherent_image(self.ptychographs[roi_name], np.array([px,py]))
        detector_image = make_detector_image(self.ptychographs[roi_name], np.array([lx,ly]))

        vmin_c = np.mean(coherent_image) - 0.05*np.mean(coherent_image)
        vmax_c = np.mean(coherent_image) + 0.05*np.mean(coherent_image)

        vmin_d = np.mean(detector_image) - 0.3*np.mean(detector_image)
        vmax_d = np.mean(detector_image) + 2*np.mean(detector_image)

        # Function to update the plot
        def update_plot():
            
            coherent_image = make_coherent_image(self.ptychographs[roi_name], np.array([px, py]))
            detector_image = make_detector_image(self.ptychographs[roi_name], np.array([lx, ly]))
            
            im0.set_data(coherent_image)
            axes[0].set_title(f"Coherent Image from Pixel ({px}, {py})")
            
            vmin_c = np.mean(coherent_image) - 0.1*np.mean(coherent_image)
            vmax_c = np.mean(coherent_image) + 2*np.mean(coherent_image)
            
            im0.set_clim(vmin_c, vmax_c)

            im1.set_data(detector_image)
            axes[1].set_title(f'Detector Image at Location ({lx},{ly})')
            
            vmin_d = np.mean(detector_image) - 0.5*np.mean(detector_image)
            vmax_d = np.mean(detector_image) + 2*np.mean(detector_image)
            
            im1.set_clim(vmin_d, vmax_d)
            
            # Update rectangles
            rect_coherent.set_xy((lx - rectangle_size, ly - rectangle_size))  # Positioning based on pixel center
            rect_detector.set_xy((px - rectangle_size, py - rectangle_size))


            fig.canvas.draw_idle()
            
        # Callback functions for buttons
        def increment_px(event):
            global px
            px = (px + 1) % self.ptychographs[roi_name].shape[2]
            update_plot()

        def decrement_px(event):
            global px
            px = (px - 1) % self.ptychographs[roi_name].shape[2]
            update_plot()

        def increment_py(event):
            global py
            py = (py + 1) % self.ptychographs[roi_name].shape[3]
            update_plot()

        def decrement_py(event):
            global py
            py = (py - 1) % self.ptychographs[roi_name].shape[3]
            update_plot()

        def increment_lx(event):
            global lx
            lx = (lx + 1) % self.ptychographs[roi_name].shape[1]
            update_plot()

        def decrement_lx(event):
            global lx
            lx = (lx - 1) % self.ptychographs[roi_name].shape[1]
            update_plot()

        def increment_ly(event):
            global ly
            ly = (ly + 1) % self.ptychographs[roi_name].shape[0]
            update_plot()

        def decrement_ly(event):
            global ly
            ly = (ly - 1) % self.ptychographs[roi_name].shape[0]
            update_plot()
        
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        im0 = axes[0].imshow(coherent_image, cmap='plasma', vmin = vmin_c, vmax = vmax_c)
        axes[0].set_title(f"Coherent Image from Pixel ({px}, {py})")
        axes[0].set_xlabel("N")
        axes[0].set_ylabel("M")
        plt.colorbar(im0, ax=axes[0], label="Intensity Coherent Images")


        im1 = axes[1].imshow(detector_image, cmap='plasma',  vmin=vmin_d, vmax=vmax_d)
        axes[1].set_title(f'Detector Image at location ({lx},{ly})')
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[1], label = "Detector Intensity")

        # Add white rectangles
        rect_coherent = Rectangle((lx - rectangle_size, ly - rectangle_size), 1, 1, edgecolor='white', facecolor='white', lw=2)
        rect_detector = Rectangle((px - rectangle_size, py - rectangle_size), 1, 1, edgecolor='white', facecolor='white', lw=2)
        axes[0].add_patch(rect_coherent)
        axes[1].add_patch(rect_detector)


        # Add buttons for each parameter
        button_ax_px_inc = plt.axes([0.2, 0.05, 0.1, 0.04])  # [left, bottom, width, height]
        button_ax_px_dec = plt.axes([0.1, 0.05, 0.1, 0.04])

        button_ax_py_inc = plt.axes([0.2, 0.01, 0.1, 0.04])
        button_ax_py_dec = plt.axes([0.1, 0.01, 0.1, 0.04])

        button_ax_lx_inc = plt.axes([0.7, 0.05, 0.1, 0.04])
        button_ax_lx_dec = plt.axes([0.6, 0.05, 0.1, 0.04])

        button_ax_ly_inc = plt.axes([0.7, 0.01, 0.1, 0.04])
        button_ax_ly_dec = plt.axes([0.6, 0.01, 0.1, 0.04])

        btn_px_inc = Button(button_ax_px_inc, 'px +')
        btn_px_dec = Button(button_ax_px_dec, 'px -')
        btn_py_inc = Button(button_ax_py_inc, 'py +')
        btn_py_dec = Button(button_ax_py_dec, 'py -')
        btn_lx_inc = Button(button_ax_lx_inc, 'lx +')
        btn_lx_dec = Button(button_ax_lx_dec, 'lx -')
        btn_ly_inc = Button(button_ax_ly_inc, 'ly +')
        btn_ly_dec = Button(button_ax_ly_dec, 'ly -')

        # Connect buttons to their callbacks
        btn_px_inc.on_clicked(increment_px)
        btn_px_dec.on_clicked(decrement_px)
        btn_py_inc.on_clicked(increment_py)
        btn_py_dec.on_clicked(decrement_py)
        btn_lx_inc.on_clicked(increment_lx)
        btn_lx_dec.on_clicked(decrement_lx)
        btn_ly_inc.on_clicked(increment_ly)
        btn_ly_dec.on_clicked(decrement_ly)

        plt.tight_layout()
        plt.show()
        
    def average_frames_roi(self, roi_name):
        
        self.averaged_rois[roi_name] = average_data(self.dir, self.fnames, self.rois_dict[roi_name], conc=True)
        self.averaged_data[roi_name] =  mask_hot_pixels(self.averaged_data[roi_name])
    
    def mask_roi(self, roi_name, hot_pixels = True, mask_val=1):
        
        if hot_pixels:
            self.ptychographs[roi_name] = mask_hot_pixels(self.ptychographs[roi_name])
    
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
    
        self.cohernt_imgs[roi_name] = coherent_imgs
        
        
    def filter_coherent_images(self, roi_name:str, variance_threshold):
        
        
        cleaned_coh_images, cleaned_coords, cleaned_kxky = filter_images(self.cohernt_imgs[roi_name], 
                                                                         coords = self.coords[roi_name],
                                                                         kin_coords=self.kouts[roi_name], 
                                                                         variance_threshold = variance_threshold)
    

        self.cohernt_imgs[roi_name+"_clean"] = cleaned_coh_images
        self.kouts[roi_name+"_clean"] = cleaned_kxky
        self.coords[roi_name+"_clean"] = cleaned_coords
        
        
    def make_kvector(self, roi_name):
        
        self.coords[roi_name] = make_coordinates(self.average_data, self.mask_val, self.rois_dict[roi_name], crop=False)
        self.kouts[roi_name] = compute_vectors(self.coords[roi_name], self.crys_distance, self.det_psize, self.centre_pixel, self.wavelength)
    
    
    def prepare_roi(self, roi_name, variance_threshold):
        
        self.make_4d_dataset(roi_name=roi_name)
        self.average_frames_roi(roi_name=roi_name)
        self.make_kvector(roi_name=roi_name)
        self.make_coherent_images(roi_name=roi_name)
        self.filter_coherent_images(roi_name=roi_name, variance_threshold=variance_threshold)
        
    