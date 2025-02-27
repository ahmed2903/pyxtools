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

from ..data_fs import list_datafiles, stack_4d_data
from  ..plotting_fs import plot_roi_from_numpy

class load_data:
    
    def __init__(self, directory , scan_num):
        
        self.dir = directory
        self.scan_num = scan_num
        self.rois_dict = {}
        self.ptychographs = {}
    
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
        pass
    
    def average_frames_roi(self):
        pass
    
    def make_mask(self):
        pass
    
    def make_coherent_images(self):
        pass
    
    def filter_coherent_images(self):
        pass
    
    def make_kvector(self):
        pass
    
    