# Imports data from Ptychography experiments
# 
# Author: Ahmed H. Mokhtar 
# Email: ahmed.mokhtar@desy.de
# Date : Feb 2025

import numpy as np 
import h5py
import matplotlib.pyplot as plt
import copy
from tqdm.notebook import tqdm
from functools import wraps
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import os 

from matplotlib.patches import Rectangle

from IPython.display import display

from .data_fs import *
from .kvectors import make_coordinates, compute_vectors, optimise_kin, reverse_kins_to_pixels, rotation_matrix, calc_qvec, extract_parallel_line, extract_streak_region
from .plotting import plot_roi_from_numpy, plot_pixel_space, plot_map_on_detector
from .utils import time_it, energy2wavelength_a
from .geometry import *

try:
    import ipywidgets as widgets
except:
    print("ipywdigets wont work outisde jupyter notebook")
    

@dataclass
class Exp:
    directory : str
    scan_num: int
    
    # Needs to be passed in
    det_psize: int  
    energy: float #keV
    
    
    step_size: float # Angstroms
    slow_axis: int  # 0 for x, 1 for y
    fast_axis_steps: int # steps
    
    beamtime: str # new or old
    
    det_distance: float = field(init=False) # microns
    wavelength: float = field(init=False)

    
    def __post_init__(self):

        self.wavelength = energy2wavelength_a(self.energy)
        # wavelength: float # A

        base_path = Path(self.directory)
        new_dir = f"Scan_{self.scan_num}"

        self.directory = os.path.join(base_path, new_dir)

@dataclass
class ROI:

    exp: Exp 
    kind: str # Bragg or Pupil
    
    coords: np.ndarray = field(default=None, repr=False)

    # acquired later
    data_4d: np.ndarray = field(default=None, repr=False) # 4D Data set (x, y, fx, fy)
    centre_pixel: np.ndarray = field(default=None, repr=False)
    
    # to be set automatically
    __averaged_det_images: np.ndarray = field(default=None, repr=False)  # Average of all detector frames 
    __averaged_coherent_images: np.ndarray = field(default=None, repr=False)
    
    # Data
    # needs to be set manually
    detected_objects: np.ndarray = field(default=None, repr=False) # Detected Objects
    coherent_imgs: np.ndarray = field(default=None, repr=False)
    
    # k-space params
    kout_coords: np.ndarray = field(default=None, repr=False)  # Coords
    kouts: np.ndarray = field(default=None, repr=False)  # Kout vectors
    kin_coords: np.ndarray = field(default=None, repr=False)  # Coords
    kins: np.ndarray = field(default=None, repr=False)  # Kin vectors
    
    kins_avg: np.ndarray = field(default=None, repr=False)  # average Kins 
    # For computing kins 
    optimal_angles: list = field(default=None, repr=False)  # Euler angles that rotate Kout to Kin
    est_ttheta: float = field(default=None, repr=False)
    g_init: np.ndarray = field(default=None, repr=False)  # Initial G vector for a Signal
        
    log_params: list = field(default_factory=list, repr = False)
    
    checkpoint_stack:list = field(default_factory=list, repr = False)


    # _______________ functionality with Exp ___________

    def __post_init__(self):
        # Copy attributes from Exp into ROI
        for f in fields(self.exp):
            setattr(self, f.name, getattr(self.exp, f.name))

        
    # _______________ Properties ________________
    @property
    def averaged_det_images(self):
        
        if self.__averaged_det_images is None:
            self.__averaged_det_images = np.mean(self.data_4d, axis=(0,1))
            
        return self.__averaged_det_images

    def update_averaged_det_images(self):
        self.__averaged_det_images = np.mean(self.data_4d, axis=(0,1))

    @averaged_det_images.setter
    def averaged_det_images(self, value):
        self.__averaged_det_images = value
        
    @property
    def averaged_coherent_images(self):
        
        if self.__averaged_coherent_images is None:
            self.__averaged_coherent_images = np.mean(self.data_4d, axis=(2,3))
        
        return self.__averaged_coherent_images
    
    @averaged_coherent_images.setter
    def averaged_coherent_images(self, value):
        self.__averaged_coherent_images = value
        
    def update_averaged_coherent_images(self):
        self.__averaged_coherent_images = np.mean(self.data_4d, axis=(2,3))
    
    
    # _______________ Saving ___________________
    
    @time_it
    def save(self, file_path):
    
        """
        Save the processed data and metadata to an HDF5 file.
    
        Args:

            file_path (str): The path where the HDF5 file will be saved.
        """

        with h5py.File(file_path, "w") as h5f:

            # Save metadata as attributes in the root group
            process_params = h5f.create_group("processing_params")
            
            process_params.attrs['roi'] = json.dumps(self.roi_coords)
            
            for func in self.log_params:
                # self.log_roi_params[roi_name] is a list, each element correponds to a function 
                # func is then a dictionary
                # func["functions"] will be the function name
                func_name = func['function']
                func.pop('function')
                process_params.attrs[func_name] = json.dumps(func)

            images = h5f.create_group("processed_images")
            images.create_dataset("coherent_images", data=self.coherent_imgs, compression="gzip",chunks=True)
                
            kvectors = h5f.create_group("kvectors")
            kvectors.create_dataset("kins", data=self.kins, compression="gzip")
            kvectors.create_dataset("kouts", data = self.kouts, compression='gzip')
            kvectors.create_dataset("kin_coords", data=self.kin_coords, compression="gzip")
            kvectors.create_dataset("kout_coords", data = self.kout_coords, compression='gzip')
                
            print(f"Data saved at {file_path}")
    

    @classmethod
    def load_data(cls, file_path):
        """
        Load processed data and metadata for a specific ROI from an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be loaded.
            file_path (str): Path to the HDF5 file containing saved data.
        """
        self = cls.__new__(cls)
        
        with h5py.File(file_path, "r") as h5f:
                                        
            # Load Processed Images 
            images = h5f["processed_images"]
            
            self.coherent_imgs = images["coherent_images"][...]
            
            # Load K-Vectors 
            kvectors = h5f["kvectors"]
            
            self.kins = kvectors['kins'][...]
            self.kouts = kvectors['kouts'][...]
            self.kin_coords = kvectors['kin_coords'][...]
            self.kout_coords = kvectors['kout_coords'][...]
                
        return self
    
    
    @time_it
    def save_4d(self, file_path):
    
        """
        Save the 4D data set for an roi to an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be saved.

            file_path (str): The path where the HDF5 file will be saved.
        """

        with h5py.File(file_path, "w") as h5f:
            data = h5f.create_group("data")
            # Save metadata as attributes in the root group
            data.create_dataset("data", data = self.data_4d, compression="gzip",chunks=True)

        print(f"Data saved at {file_path}")
    @time_it
    def load_4d(self, file_path):
    
        """
        Load the 4D data set for an roi from an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be loaded.

            file_path (str): The path where the HDF5 is stored.
        """
        print("Loading data ...")
        with h5py.File(file_path, "r") as h5f:
            data = h5f["data"]
            # Save metadata as attributes in the root group
            self.data_4d = np.array(data["data"])

        print("Data loaded")

    # _____________ Checkpointing _____________
    
    def checkpoint_state(self):
        """Create a deep copy of the current data state for potential rollback.

        This method saves the current state of all main data dictionaries 
        (`kins`, `kouts`, `kin_coords`, `kout_coords`) by creating deep copies.
        It allows restoring this exact state later using `restore_checkpoint()`.
        """
        state = copy.deepcopy(self.rois)
        
        self._checkpoint_stack.append(state)
        
        print(f"Checkpoint #{len(self._checkpoint_stack)} created.")
    
    def restore_checkpoint(self):
        """Restore the most recently saved checkpoint.

        Replaces the current data state with the previously saved checkpoint.
        This effectively undoes any operations performed since the last 
        `checkpoint_state()` call.

        If no checkpoint has been saved, this method prints a warning and does nothing.
        """
        if not self._checkpoint_stack:
            print("No checkpoints to restore.")
            return
        
        state = self._checkpoint_stack.pop()
        self.rois = copy.deepcopy(state)
        
        print("Restored to previous checkpoint.")
    
    def auto_checkpoint(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.checkpoint_state()
            return func(self, *args, **kwargs)
        return wrapper
    
    
    
    
    
class processor:
    """
    Class for loading and organizing experimental scan data.

    This class handles the loading of scan data files, manages metadata, 
    and stores various processing results such as regions of interest (ROIs), 
    ptychographic 4d data sets, sigle pixel coherent images, and coordinate mappings.

    It also does preprocessing of the coherent images in preparation for phase retrieval. 

    Args:
        directory (str): Path to the directory containing scan data.
        scan_num (int): Scan number used to identify the dataset.
        det_psize (float): Detector pixel size in meters.
        det_distance (float): Distance between the detector and sample in meters.
        centre_pixel (tuple[int]): Coordinates of the center pixel in the detector.
        wavelength (float): Wavelength of the incident X-rays in meters.
        slow_axis (int): Index representing the slow axis in the scanning system.
        beamtime (str, optional): Identifier for the beamtime ('new' or 'old'). Defaults to 'new'.
        fast_axis_steps (optional): Step values for the fast axis in the scan. Defaults to None.
    """
    
    def __init__(self):

        
        self.averaged_full_det_data = None
        
   
    ################### Logging ##################
    @staticmethod
    def log_roi_params(func):
        """
        Decorator for class methods that logs function calls and parameters per ROI.
        Assumes all methods have `roi_name` as the first positional argument after `self`
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get roi_name (first arg after self)
            mod_kwargs = kwargs.copy()
            
            try:
                roi = args[0]
            except:
                roi = kwargs['roi']
                mod_kwargs.pop("roi")
                assert len(args) == 0 

            entry = {
                "function": func.__name__,
                "args": args[1:],  
                "kwargs": mod_kwargs
            }

            roi.log_params.append(entry)
        
            return func(*args, **kwargs)
    
        return wrapper
    
    ################### Loading data ###################
    @time_it
    @staticmethod
    def make_4d_dataset(roi: ROI, num_jobs = 64):
        
        """Creates a 4D dataset from a region of interest (ROI) on the detector.

        This method extracts a 4D dataset from the scan data, based on the selected ROI.
        It applies hot pixel masking if specified.
    
        Args:
            roi_coords (list): The region of interest (ROI) to process.
    
        Returns:
            The 4D dataset for the given ROI.
        """
        
        if roi.beamtime == 'old':
            # Just for naming
            direc = os.path.join(roi.directory , "Scan_{roi.scan_num}_data_000001.h5")
            if roi.fast_axis_steps is None:
                raise ValueError("fast_axis_steps is required")            
            
            data_4d = stack_4d_data_old(direc, 
                                        roi.coords, 
                                        roi.fast_axis_steps, 
                                        roi.slow_axis)
            
        else:
            direc = roi.directory
            fnames = list_datafiles(direc)[:-2]       
            data_4d = stack_4d_data(direc, 
                                                        fnames, 
                                                        roi.coords, 
                                                        slow_axis = roi.slow_axis, 
                                                        conc=True, 
                                                        num_jobs=num_jobs)
        
        roi.data_4d = mask_hot_pixels(data_4d, 
                                    mask_max_coh = False, 
                                    mask_min_coh = False)
        

     ################### Plotting ###################



        
    @staticmethod
    def estimate_pupil_size(pupil_roi:ROI, mask_val):
        """
        Estimates the pupil size from the averaged pupil data and updates the instance attribute.
    
        Args:
            mask_val (float or int): A threshold value used to mask or filter the pupil data
                before estimating its size.
        
        Updates:
            self.pupil_size: The estimated pupil size, computed using the averaged pupil data
            and the given mask threshold.
        """
        pupil_size = estimate_pupil_size(pupil_roi.averaged_det_images, 
                           mask_val=mask_val,
                           pixel_size=pupil_roi.det_psize)

        return pupil_size
    
    @staticmethod
    def add_lens(focal_length:float, height:float):
        """
        Adds a lens with given parameters to the internal lens parameter dictionary.
    
        Args:
            lens_name (str): The name/key to identify the lens.
            focal_length (float): The focal length of the lens in microns.
            height (float): The physical aperture height of the lens in microns.
        
        Updates:
            self.lens_params: Adds a dictionary entry containing focal length, height,
            and calculated numerical aperture (NA) under the given lens name.
        """
        lens_na = calculate_NA(focal_length, height)

        return lens_na

    @staticmethod
    def estimate_detector_distance_from_NA(na1, na2, pupil_size):
        """
        Estimates the average detector distance based on the range of numerical apertures (NA)
        from the added lenses and current pupil size.
    
        The function identifies the smallest and largest NA values from the lens parameters,
        computes detector distances corresponding to the minimum and maximum pupil sizes,
        and averages them to determine a representative detector distance.
    
        Updates:
            self.exp_params['det_distance']: Estimated detector distance in microns based on optical geometry.
        """
        largest_na = 0
        smallest_na = 1e4
        
        for na in [na1, na2]:

            largest_na = max(largest_na, na)
            smallest_na = min(smallest_na, na)    
        
        distance1 = estimate_detector_distance_from_NA(largest_na, max(pupil_size))
        distance2 = estimate_detector_distance_from_NA(smallest_na, min(pupil_size))

        print(f"First detector distance is {distance1} microns")
        print(f"Second detector distance is {distance2} mircons")
        
        return (distance1+distance2)/2
    
    
    ################### prepare detector roi ###################

    @log_roi_params
    @staticmethod
    def pool_detector_space(roi:ROI, kernel_size, padding=0):
        """Performs pooling on the detector space for a given region of interest (ROI).

        This method applies a 2D sum pooling operation on the ptychograph data of a 
        specific region of interest (ROI). It reduces the spatial resolution of the 
        detector space by pooling with a given kernel size, stride, and padding. 
        If the stride is not provided, it will default to the kernel size.

        The efective detector pixel is modified accordingly.
        
        Args:
            roi_name (str): The name of the region of interest (ROI) to be processed.
            kernel_size (int): The size of the pooling kernel, determining the window 
                                over which the sum pooling is applied.
            stride (int, optional): The stride of the pooling operation, i.e., the 
                                     step size between each pooling operation. If 
                                     not provided, defaults to `kernel_size`.
            padding (int, optional): The amount of zero padding to add around the 
                                      edges of the image before pooling. Defaults to 0.
        """
        print("Pooling detector ...")
        roi.data_4d = sum_pool2d_array(roi.data_4d, 
                                                       kernel_size=kernel_size,
                                                       stride=None,
                                                       padding=0)
                
        
        roi.det_psize *= kernel_size

        if roi.kout_coords is not None:
            roi.kout_coords = np.unique(roi.kout_coords // kernel_size)
        
        roi.coords = np.array(roi.coords) // kernel_size
        
        print("Done.")

        
    @log_roi_params    
    @classmethod
    def normalise_detector(cls, roi:ROI, reference_roi:ROI):
        """Normalizes the detector data based on the reference ROI's peak intensity.

        This method normalizes the ptychograph data for a given ROI (region of interest) 
        by scaling the intensity of the detector data from a reference ROI. The normalization 
        is performed by adjusting the intensity of the operating ROI to match the average 
        intensity of the reference ROI.
    
        If the ptychograph data for the reference ROI is not already available, it is 
        loaded from the specified directory, and any hot pixels are masked before proceeding.
    
        Args:
            roi_name_ref (str): The name of the reference region of interest (ROI) used 
                                 for calculating the peak intensity.
            roi_name (str): The name of the operating region of interest (ROI) to be normalized.
    
        """
        try:
            peak_intensity = np.sum(reference_roi.data_4d, axis=(-2,-1))
        except: 
            cls.make_4d_dataset(reference_roi)
            peak_intensity = np.sum(reference_roi.data_4d, axis=(-2,-1))
            print(peak_intensity.shape)
        
        
        avg_intensity = np.mean(peak_intensity)
        roi.data_4d = roi.data_4d/peak_intensity[...,np.newaxis, np.newaxis] * avg_intensity

    
    @log_roi_params
    @staticmethod
    def mask_region_detector(roi:ROI, region, mode = 'median'):
        """Masks a specific region of the detector data within the given ROI.

        This method applies a median mask to a specified region within the detector data 
        (ptychographs) of a particular region of interest (ROI). The region is defined by the 
        `region` argument, which specifies the start and end indices for both the x and y 
        dimensions. The region is replaced by the median value of the corresponding region 
        across all detector data.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose detector data 
                             will be modified.
            region (tuple): A tuple (sx, ex, sy, ey) specifying the start and end indices 
                            for the region to be masked:
                            - sx, ex: x-axis start and end indices
                            - sy, ey: y-axis start and end indices
            mode (str, optional): A string specifying how to replace the masked region. 
                            - 'zeros'. Default.
                            - 'median'
                            - 'ones'
        """
        sx,ex,sy,ey = region
        if mode == 'zeros':
            roi.data_4d[:,:,sx:ex,sy:ey] = 0.0
            roi.update_averaged_det_images()
        elif mode == 'median':
            roi.data_4d[:,:,sx:ex,sy:ey] = np.median(roi.data_4d, axis = (2,3))[:,:,np.newaxis, np.newaxis]
            roi.update_averaged_det_images()
        elif mode == 'ones':
            roi.data_4d[:,:,sx:ex,sy:ey] = 1.0
            roi.update_averaged_det_images()

    
    ################### Compute k_in vectors ###################
    
    @log_roi_params
    @staticmethod
    def make_kouts( roi:ROI, mask_val):
        """Computes the k-space vectors for a given region of interest (ROI).

        This method calculates the pixel coordinates and k-space vectors for a specified 
        region of interest (ROI) using the averaged data. It uses the given mask value 
        to filter out unwanted pixels and computes the k-space vectors based on the 
        coordinates, detector distance, pixel size, center pixel, and wavelength.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) for which the 
                             k-space vectors are to be computed.
            mask_val (float): The value used for masking pixels during coordinate 
                              computation.
    
        """
        roi.kout_coords = make_coordinates(roi.averaged_det_images, 
                                                                mask_val, 
                                                                roi.coords, 
                                                                crop=False)
        if roi.centre_pixel is None and roi.kind == 'pupil':
            roi.centre_pixel = np.array([(roi.kout_coords[:,0].max() + roi.kout_coords[:,0].min() )/2, (roi.kout_coords[:,1].max() + roi.kout_coords[:,1].min() )/2])

        
        roi.kouts = compute_vectors(roi.kout_coords, 
                                                roi.det_distance, 
                                                roi.det_psize, 
                                                roi.centre_pixel, 
                                                roi.wavelength)

        if roi.kind == 'pupil':
            
            roi.kins = roi.kouts
            roi.kin_coords = roi.kout_coords
            roi.kins_avg = np.mean(roi.kouts, axis = 0, keepdims = True )
    
    
    @staticmethod
    def estimate_ttheta(roi:ROI, pupil_roi:ROI):
        """Estiamtes the two theta value for the signal in the roi, assuming there is one incident wavevector which is specified by the average of all k_in vectors in the pupil. 
        Args:
            roi_name (str): The name of the region of interest (ROI) for which the two_theta value is to be estimated.

        Updates:
            self.est_ttheta[roi_name]: Estimated 2 theta angle for that signal
        
        """
        try:
            pupil_roi.kins_avg = np.mean(pupil_roi.kouts, axis = 0, keepdims = True )
        except:
            raise ValueError("Must compute pupil kouts first")

        kouts_avg = np.mean(roi.kouts, axis = 0, keepdims = True )
        kouts_avg /= np.linalg.norm(kouts_avg)
        
        kins_avg = pupil_roi.kins_avg/np.linalg.norm(pupil_roi.kins_avg)


        
        angle = np.arccos(np.sum(kins_avg* kouts_avg))
        angle = np.rad2deg(angle)
        
        roi.est_ttheta = angle
        
        print(f"the initial 2theta angle is: {angle:.2f}")
        
    
        
    @log_roi_params
    @staticmethod
    def compute_kins(roi:ROI, pupil_roi:ROI, est_ttheta, method = "BFGS", gtol = 1e-6):
        """
        Computes the incident wavevectors (kins) for a given region of interest (ROI) 
        by optimizing the initial q-vector estimate.

        Args:
            roi_name (str): 
                The name of the region of interest (ROI) for which kins are computed.
            est_ttheta (float): 
                Estimated scattering angle (two-theta) in radians.
            method (str, optional): 
                Optimization method to use for minimizing the k-vector difference. 
                Defaults to "BFGS".
            gtol (float, optional): 
                Gradient tolerance for optimization. Defaults to 1e-6.

        Raises:
            ValueError: If the pupil kouts have not been computed before calling this function.

        Updates the following attributes of the instance:
            - `self.kins[roi_name]`: Optimized incident wavevectors.
            - `self.optimal_angles[roi_name]`: Optimized rotation angles (alpha, beta, gamma).
            - `self.kin_coords[roi_name]`: Pixel coordinates corresponding to `self.kins[roi_name]`.
        """

        
            

        
        roi.g_init = calc_qvec(roi.kouts, pupil_roi.kins_avg)
    
        roi.kins, roi.optimal_angles = optimise_kin(roi.g_init, 
                                                                          est_ttheta, 
                                                                          roi.kouts, 
                                                                          roi.wavelength, 
                                                                          method, gtol)
        
        roi.kin_coords = reverse_kins_to_pixels(roi.kins, 
                                                           roi.det_psize, 
                                                           roi.det_distance, 
                                                           roi.centre_pixel)

    
        
    @log_roi_params
    @staticmethod
    def refine_kins(roi:ROI, shifts):
        """
        Refine the estimated incident wavevectors (kins) by adjusting the optimal angles.

        Args:
            roi_name (str): The region of interest name.
            shifts (tuple): A tuple of (alpha_shift, beta_shift, gamma_shift) to refine angles.
        """
        alpha,beta,gamma = (roi.optimal_angles[0] + shifts[0],
                            roi.optimal_angles[1] + shifts[1],
                            roi.optimal_angles[2] + shifts[2] )
        
        R = rotation_matrix(alpha, beta, gamma)
        try:
            g_update =  R @ roi.g_init
        except:
            g_update =  R @ roi.g_init[0]
        kin_opt = (roi.kouts-g_update)
        kin_opt /= np.linalg.norm(kin_opt, axis = 1)[:,np.newaxis]
        kin_opt *= 2*np.pi/roi.wavelength
        roi.kins = kin_opt
        roi.kin_coords = reverse_kins_to_pixels(roi.kins,   roi.det_psize, 
                                                            roi.det_distance, 
                                                            roi.centre_pixel)
    


    @log_roi_params
    @staticmethod
    def select_single_pixel_streak(roi:ROI, width=1, offset=0, inplace = False):

        """
        Selects a single-pixel-wide streak along a specified direction in the region of interest (ROI).

        This method extracts a narrow streak from the `kin_coords` of the specified ROI and applies the
        same selection mask to `kins`, `kin_coords`, `kouts`, and `kout_coords`.
    
        Args:
            roi_name (str): The name of the region of interest.
            width (int, optional): The width of the streak to extract. Default is 1.
            position (str, optional): The position of the streak relative to the main region.
                Options: 'center', 'top', 'bottom'. Default is 'center'.
            offset (int, optional): An offset value to shift the streak selection up or down. Default is 0.
            inplace (bool): if True, it changes the kin_coords, kins, kouts, and kout_coords dict for that roi, if false, if returns the mask. (default = False)

        """
        mask = extract_parallel_line(roi.kin_coords, width=width, offset=offset)
        
        if inplace:
            roi.kins = roi.kins[mask]
            roi.kin_coords = roi.kin_coords[mask]
            roi.kouts = roi.kouts[mask]
            roi.kout_coords = roi.kout_coords[mask]
            try:
                roi.coherent_imgs = roi.coherent_imgs[mask]
            except:
                print("coherent images are not calculated yet")
            
            return  
        else:
            print(f'shape of mask is {mask.shape}')
            return mask
        
    @log_roi_params
    @staticmethod
    def select_streak_region(roi:ROI, percentage=10, start_position='lowest', start_idx=None, inplace = False):
        """
        Selects a region of the streak based on a percentage of its total size.

        This method extracts a portion of the streak from the `kin_coords` of the specified ROI,
        starting from a defined position or index, and applies the same selection mask to `kins`, 
        `kin_coords`, `kouts`, and `kout_coords`.
    
        Args:
            roi_name (str): The name of the region of interest.
            percentage (float, optional): The percentage of the streak to retain. Default is 10%.
            start_position (str, optional): The starting position for selection.
                Options: 'lowest' (bottom), 'highest' (top), or 'middle'. Default is 'lowest'.
            start_idx (int, optional): The specific index to start selection from. Overrides `start_position` if provided.
        """
        
        mask = extract_streak_region(roi.kin_coords, percentage=percentage, start_position=start_position, 
                                     start_idx=start_idx, seed=42)
        if inplace:
            roi.kins = roi.kins[mask]
            roi.kin_coords = roi.kin_coords[mask]
            roi.kouts = roi.kouts[mask]
            roi.kout_coords = roi.kout_coords[mask]
            roi.coherent_imgs = roi.coherent_imgs[mask]
        else:
            return mask
    
    @staticmethod
    def order_pixels(roi:ROI):
        """Reorders the pixels of coherent images and corresponding k-vectors based on their distance from the center.
    
        This method reorders the coherent images and updates the corresponding k-vectors
        based on their distance from the center. The reordering is done to facilitate specific analysis or
        visualization by aligning the data in a certain order.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) for which the pixels 
                             and k-vectors will be reordered.
        """
        sorted_indices = reorder_pixels_from_center(roi.kouts, connected_array=np.array(roi.coherent_imgs))
        
        # Debugging: Print types and shapes
        print(f"sorted_indices dtype: {sorted_indices.dtype}, shape: {sorted_indices.shape}")
        print(f"kouts shape before indexing: {roi.kouts.shape}")

        roi.kouts = roi.kouts[sorted_indices]
        roi.kout_coords = roi.kout_coords[sorted_indices]
        roi.coherent_imgs = roi.coherent_imgs[sorted_indices]

        try:
            roi.kins = roi.kins[sorted_indices]
            roi.kin_coords = roi.kin_coords[sorted_indices]
        except:
            print("Did not order kins and kin coords.")

        try:
            roi.detected_objects = roi.detected_objects[sorted_indices]
        except:
            print("Did not order detected objects.")   
            
    ################### preprocessing coherent images ###################
    @time_it
    @staticmethod
    def make_coherent_images(roi:ROI):
        
        """
        Generates a list of coherent images for a given region of interest (ROI).

        This method creates coherent images by processing the coordinates of the specified 
        ROI and calling the `make_coherent_image` function for each coordinate. The results 
        are stored in `self.coherent_imgs` for further use.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) for which the coherent 
                             images will be generated. The coordinates for the ROI are 
                             retrieved from `self.kout_coords[roi_name]`, and the corresponding 
                             ptychographic data is accessed from `self.ptychographs[roi_name]`.
    
        """
        
        coherent_imgs = []
        
        for i, coord in enumerate(roi.kout_coords):
            
            
            xp =  coord[0] - roi.coords[0]
            yp =  coord[1] - roi.coords[2]

            coherent_imgs.append(make_coherent_image(roi.data_4d, np.array([xp,yp])))

        roi.coherent_imgs = np.array(coherent_imgs)
    
    
    @log_roi_params  
    @staticmethod
    def filter_coherent_images(roi:ROI, variance_threshold):
        """Filters coherent images based on variance threshold.

        This method filters out noisy or low-variance coherent images for a given 
        region of interest (ROI) by using a variance threshold. It removes images 
        with variance below the specified threshold and updates the coherent images, 
        coordinates, and kx, ky values accordingly.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) for which the 
                             coherent images will be filtered.
            variance_threshold (float): The threshold value for variance. Images with 
                                        variance below this threshold will be filtered out.
    
        """
        
        cleaned_coh_images, cleaned_kins, cleaned_kxky = filter_images(images = roi.coherent_imgs, 
                                                                         coords = roi.kins,
                                                                         kin_coords=roi.kouts, 
                                                                         variance_threshold = variance_threshold)
    

        roi.coherent_imgs = cleaned_coh_images.copy()
        roi.kouts = cleaned_kxky.copy()
        roi.kins = cleaned_kins.copy()
        
    @log_roi_params
    @staticmethod
    def remove_coh_background(roi:ROI, sigma, num_jobs = -1):
        """Removes the background noise from coherent images for a given region of interest (ROI).

        This method applies a background removal technique to the coherent images 
        of a specific region of interest (ROI) using a Gaussian filter with a given 
        standard deviation (sigma). The process helps to reduce unwanted background 
        noise in the coherent images, improving the signal-to-noise ratio.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose coherent 
                             images will be processed.
            sigma (float): The standard deviation of the Gaussian filter used for 
                           background removal. A higher value will smooth the image 
                           more, removing more background noise but potentially 
                           blurring fine details.
        """
        
        roi.coherent_imgs = remove_background_parallel(roi.coherent_imgs, sigma=sigma, n_jobs=num_jobs)
        
    @staticmethod
    def even_dims_cohimages(roi:ROI, mode = 'constant', num_jobs = -1, **kwargs):
        """Adjusts the dimensions of coherent images to be even.

        This method takes the coherent images for a given region of interest (ROI) 
        and ensures that their dimensions are even by calling the `make_2dimensions_even` function.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) for which the coherent 
                             images will have their dimensions adjusted. The coherent images 
                             are accessed from `self.coherent_imgs[roi_name]`.
    
        """
        
        roi.coherent_imgs = make_2dimensions_even(roi.coherent_imgs, mode=mode, num_jobs=num_jobs, **kwargs)
        roi.averaged_coherent_images = make_2dimensions_even([roi.averaged_coherent_images],mode=mode,
                                                                        num_jobs=num_jobs)[0]
    
    
    @log_roi_params
    @staticmethod
    def apply_median_filter(roi:ROI, kernel_size, stride, threshold, num_jobs = -1):
        """Applies a median filter to coherent images for a given region of interest (ROI).

        This method applies a median filter to the coherent images of a specified 
        region of interest (ROI). The filter helps to remove noise and outliers by 
        replacing each pixel's value with the median of its neighbors within a 
        specified kernel. The filter can also apply a threshold to the pixel values 
        for further noise reduction. The filtering is done in parallel using multiple 
        CPU cores to speed up the process.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose coherent 
                             images will be processed.
            kernel_size (int): The size of the square kernel to be used for the 
                                median filter. A larger kernel size will result in 
                                stronger smoothing.
            stride (int): The step size for applying the kernel across the image. 
                           A higher stride will apply the filter less frequently.
            threshold (float): The threshold value used to filter out small intensity 
                               variations, helping to remove noise and improve the 
                               quality of the coherent images.
        """
        roi.coherent_imgs = median_filter_parallel(roi.coherent_imgs, 
                                                              kernel_size = kernel_size, 
                                                              stride = stride, 
                                                              threshold=threshold, 
                                                              n_jobs=num_jobs)
        
    @log_roi_params
    @staticmethod
    def apply_bilateral_filter(self, roi:ROI, sigma_spatial, sigma_range, kernel_size, num_jobs = -1):
        """Applies a bilateral filter to the coherent images for a given region of interest (ROI).

        This method applies a bilateral filter to the coherent images of a specified 
        region of interest (ROI). The bilateral filter smooths the image while preserving 
        edges by considering both spatial proximity and intensity similarity. It uses 
        two parameters: `sigma_spatial` controls the spatial smoothing, and `sigma_range` 
        controls the intensity smoothing. The kernel size defines the size of the neighborhood 
        for the filter.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose coherent 
                             images will be processed.
            sigma_spatial (float): The standard deviation of the spatial Gaussian filter. 
                                   Larger values result in greater spatial smoothing.
            sigma_range (float): The standard deviation of the range (intensity) Gaussian 
                                 filter. Larger values preserve more intensity variation.
            kernel_size (int): The size of the square kernel to be used for the bilateral 
                            filter. A larger kernel size will result in stronger filtering.
    """
        roi.coherent_imgs = bilateral_filter_parallel(roi.coherent_imgs, 
                                                                      sigma_spatial, 
                                                                      sigma_range, 
                                                                      kernel_size, n_jobs=num_jobs)
    @log_roi_params
    @staticmethod
    def detect_object(roi:ROI, threshold, min_val, max_val, num_jobs = -1):
        """Detects objects within the specified region of interest (ROI) and filters based on intensity.

        This method applies an object detection algorithm to the coherent images in the given 
        region of interest (ROI), based on a specified intensity threshold. The detected objects 
        are then filtered based on their total intensity being within the specified `min_val` 
        and `max_val` range. Only objects that meet these criteria are retained.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) to process.
            threshold (float): The intensity threshold for object detection. Objects with intensity 
                               below this threshold will not be detected.
            min_val (float): The minimum intensity threshold for filtering detected objects.
            max_val (float): The maximum intensity threshold for filtering detected objects.
    
        """
        roi.detected_objects = detect_obj_parallel(roi.coherent_imgs, threshold=threshold, n_jobs=num_jobs)
        
        
        mask = [np.sum(im) > min_val and np.sum(im) < max_val for im in roi.detected_objects]
        print(f'length of the mask is {np.sum(mask)}')
        
        roi.detected_objects = np.array(roi.detected_objects)[mask].copy()
        roi.coherent_imgs = np.array(roi.coherent_imgs)[mask].copy()

        print(f'length of detected objects {roi.detected_objects.shape}')
        print(f'length of coherent images {roi.coherent_imgs.shape}')
        
        roi.kouts = np.array(roi.kouts)[mask].copy()
        roi.kins = np.array(roi.kins)[mask].copy()
        
    @log_roi_params    
    @staticmethod
    def blur_detected_objs(roi:ROI, sigma):
        """Applies Gaussian blur to the detected objects in the specified region of interest (ROI).

        This method applies a Gaussian blur to the images of detected objects within a 
        given region of interest (ROI). The blur is controlled by the `sigma` parameter, 
        which determines the standard deviation of the Gaussian kernel. A larger `sigma` 
        results in a stronger blur.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose detected object 
                             images will be blurred.
            sigma (float): The standard deviation of the Gaussian filter. Larger values 
                           result in more blurring.
    
        """
        roi.detected_objects = apply_gaussian_blur(roi.detected_objects, sigma=sigma)
        
    @log_roi_params
    @staticmethod
    def mask_cohimgs_threshold(roi:ROI, threshold_value):
        """Applies a threshold mask to the coherent images in the given ROI.

        This method thresholds the coherent images for the specified region of interest (ROI) 
        by applying a threshold value. Any values below the threshold will be masked or set 
        to zero, and values above the threshold will remain unchanged.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose coherent images 
                             will be thresholded.
            threshold_value (float): The threshold value used for masking the coherent images. 
                                      Values below this threshold will be masked.
    
        """
        roi.coherent_imgs = threshold_data(roi.coherent_imgs, threshold_value)
        
    @log_roi_params
    @staticmethod
    def mask_region_cohimgs(roi:ROI, region, mode = 'zeros'):
        """Masks a specific region of coherent images within the given ROI.

        This method applies a median mask to a specified region within the coherent images 
        of a particular region of interest (ROI). The region is defined by the `region` argument, 
        which specifies the start and end indices for both the x and y dimensions. The region is 
        replaced by the median value of the corresponding region across all images.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose coherent images 
                             will be modified.
            region (tuple): A tuple (sx, ex, sy, ey) specifying the start and end indices 
                            for the region to be masked:
                            - sx, ex: x-axis start and end indices
                            - sy, ey: y-axis start and end indices
            mode (str, optional): A string specifying how to replace the masked region. 
                            - 'zeros'. Default.
                            - 'median'
                            - 'ones'
        """
        sx,ex,sy,ey = region
        roi.coherent_imgs = np.array(roi.coherent_imgs)
        if mode == 'zeros':
            roi.coherent_imgs[:,sx:ex,sy:ey] = 0.0
            roi.update_averaged_coherent_images()
        elif mode == 'median':
            roi.coherent_imgs[:,sx:ex,sy:ey] = np.median(roi.coherent_imgs, 
                                                                        axis = (1,2))[:,np.newaxis, np.newaxis]
            roi.update_averaged_coherent_images()
        elif mode == 'ones':
            roi.coherent_imgs[:,sx:ex,sy:ey] = 1.0
            roi.update_averaged_coherent_images()
            
    @staticmethod
    def align_coherent_images(roi:ROI):
        """Aligns a list of coherent images for a given region of interest (ROI).

        This method uses the first image in the list as a reference and aligns the subsequent 
        images in the list by applying phase correlation to estimate and correct the shifts.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose coherent images
                             will be aligned.
    
        """
        roi.coherent_imgs = align_images(roi.coherent_imgs)
        

    
   

class plotter:
    @staticmethod
    def plot_detector_roi(roi:ROI, file_no, frame_no, title=None, vmin=None, vmax=None,mask_hot = True, save=False):
        """Plots a region of interest (ROI) on the detector for a given frame.

        This method retrieves a specific ROI from the detector data and visualizes it.
        It supports optional hot pixel masking and saving of the generated plot.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) to be plotted.
            file_no (int): The file number to retrieve the frame from.
            frame_no (int): The frame number within the specified file.
            title (str, optional): The title of the plot. Defaults to an auto-generated title.
            vmin (float, optional): Minimum intensity value for visualization. Defaults to None.
            vmax (float, optional): Maximum intensity value for visualization. Defaults to None.
            mask_hot (bool, optional): Whether to apply hot pixel masking. Defaults to False.
            save (bool, optional): Whether to save the plotted image. Defaults to False.
    
        Raises:
            KeyError: If the specified ROI name is not found in `self.rois_dict`.
    
        Displays:
            A plot of the selected ROI on the detector.
        """
        direc = roi.directory
        
        if roi.beamtime == 'new':
            file_no_st = (6-len(str(file_no)))*'0' + str(file_no)
            
            file_name = os.path.join(direc,f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
    
            with h5py.File(file_name,'r') as f:
                data_test = f['/entry/data/data'][frame_no,:,:]
        else:
            
            data_test = stack_4d_data_old(direc, roi.coords, roi.fast_axis_steps, roi.slow_axis)[file_no,frame_no, :,:]

        if mask_hot:
            data_test = mask_hot_pixels(data_test)
        if title is None:
            title = f"Detector image in Frame ({file_no}, {frame_no})"
        plot_roi_from_numpy(data_test, roi.coords, title, vmin=vmin, vmax=vmax, save = save)
    
    
    @staticmethod
    def plot_full_detector(roi:ROI, file_no, frame_no, 
                          vmin1=None, vmax1=None, 
                          vmin2=None, vmax2=None):
        
        """Plots a full frame of the detector.

        This method retrieves and visualizes a single frame from the detector, 
        displaying both the raw and log-transformed versions of the data.
    
        Args:
            file_no (int): The file number to retrieve the frame from.
            frame_no (int): The frame number within the specified file.
            vmin1 (float, optional): Minimum intensity value for raw data visualization. Defaults to None.
            vmax1 (float, optional): Maximum intensity value for raw data visualization. Defaults to None.
            vmin2 (float, optional): Minimum intensity value for log-transformed visualization. Defaults to None.
            vmax2 (float, optional): Maximum intensity value for log-transformed visualization. Defaults to None.
    
        Raises:
            KeyError: If the required dataset is not found in the HDF5 file.
    
        Displays:
            A matplotlib figure with two subplots:
            - Left: Raw intensity data.
            - Right: Log-transformed intensity data.
        """
        direc = roi.directory
        if roi.beamtime == 'new':
            file_no_st = (6-len(str(file_no)))*'0' + str(file_no)
            
            file_name = os.path.join(direc , f'Scan_{roi.scan_num}_data_{file_no_st}.h5')
    
            with h5py.File(file_name,'r') as f:
                data_test = f['/entry/data/data'][frame_no,:,:]
        else:
            data_test = stack_4d_data_old(direc, [0,-1,0,-1], roi.fast_axis_steps, roi.slow_axis)[file_no,frame_no, :,:]

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
        
    @staticmethod
    def plot_4d_dataset(roi : ROI, vmin1 = None, vmax1 = None,vmin2 = None, vmax2 = None):
        """Plots the 4D dataset for a specified region of interest (ROI).
    
        This method generates an interactive plot showing both coherent and detector images 
        from the 4D dataset. The plot includes sliders for selecting pixel positions in both 
        coherent and detector images, and updates dynamically as the sliders are adjusted.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) to visualize. The ROI 
                             should be present in the `self.ptychographs` dictionary.
    
        Displays:
            An interactive plot with two subplots:
                - The first subplot displays the coherent image with a rectangle indicating 
                  the selected scan pixel position.
                - The second subplot displays the detector image with a rectangle indicating 
                  the selected detector pixel.
    
        Interactive Widgets:
            - A slider for the row and column positions on the coherent image.
            - A slider for the row and column positions on the detector image.
    
        Updates:
            The images are dynamically updated as the slider values change.
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
    
    @staticmethod
    def plot_coherent_sequence(roi: ROI, scale_factor = .4):
        """Displays a sequence of coherent images and allows scrolling through them via a slider.
        
        This method generates an interactive plot that displays a series of coherent images. 
        A slider is provided to allow the user to scroll through the images. The color scale 
        of the images is dynamically adjusted based on the mean intensity of each image.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) to visualize. The ROI 
                             should be present in the `self.coherent_imgs` dictionary.
            scale_factor (float, optional): A factor to scale the maximum color intensity 
                                             for the images. Default is 0.4.
    
        Displays:
            An interactive plot with a slider to scroll through the images. The current image 
            and its intensity scale are updated as the slider is moved.
        """
        
        img_list = roi.coherent_imgs  # List of coherent images

        num_images = len(img_list)  # Number of images in the list
        
        # Create a slider for selecting the image index
        img_slider = widgets.IntSlider(min=0, max=num_images - 1, value=0, description="Image")

        # Create figure & axis once
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Initial image
        vmin, vmax = np.min(img_list[0]), np.max(img_list[0])  # Normalize color scale
        im = ax.imshow(img_list[0], cmap='viridis') #, vmin=vmin, vmax=vmax)
        ax.set_title(f"Coherent Image {0}/{num_images - 1}")
        plt.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout()
        
        def update_image(img_idx):
            """Updates the displayed image when the slider is moved."""
            img = img_list[img_idx]
            img_mean = np.mean(img)
            vmin = img_mean - 0.05 * img_mean
            vmax = img_mean + scale_factor * img_mean
            
            im.set_data(img)  # Update image data
            im.set_clim(vmin, vmax)
            
            ax.set_title(f"Coherent Image {img_idx}/{num_images - 1}")  # Update title
            fig.canvas.draw_idle()  # Efficient redraw

        # Create interactive slider
        interactive_plot = widgets.interactive(update_image, img_idx=img_slider)

        display(interactive_plot)  # Show slider
        #display(fig)  # Display the figure
    
    
    @staticmethod
    def plot_averag_coh_imgs(roi:ROI, vmin=None, vmax=None, title=None):
        """Plots the average of coherent images for a given region of interest (ROI).

        This method computes the average of all coherent images for a specific region of 
        interest (ROI) and plots the resulting average image. The plot can be customized 
        with optional color scale limits (`vmin`, `vmax`) and an optional title.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) for which the 
                             average coherent image will be plotted.
            vmin (float, optional): The minimum value for the color scale. If not specified, 
                                    the minimum value of the average image will be used.
            vmax (float, optional): The maximum value for the color scale. If not specified, 
                                    the maximum value of the average image will be used.
            title (str, optional): The title for the plot. If not specified, a default title 
                                   including the ROI name will be used.
    
        """
        avg = roi.averaged_coherent_images
        #avg = np.mean(np.array(self.coherent_imgs[roi_name]), axis = 0)

        if title is None:
            title = f"Average Coherent Images"
            
        plot_roi_from_numpy(avg, name=title, vmin=vmin, vmax=vmax)
        
    @staticmethod
    def plot_detected_objects(roi:ROI):
        """Displays the detected objects in coherent images with a slider for navigation.

        This method visualizes the list of detected objects in the coherent images for the 
        specified region of interest (ROI). The user can scroll through the images using a 
        slider and view each image with its corresponding intensity range.
    
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
        
    
    @staticmethod
    def plot_kin_space(roi:ROI, connection=False):
        """Plots the pixel space of the given region of interest (ROI) using k-vectors.

        This method visualizes the pixel space of a specific ROI by plotting its k-vectors.
        Optionally, the method can show the connections between the pixels, depending on the 
        `connection` argument.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) to plot.
            connection (bool, optional): Whether to display connections between the pixels.
                                         Defaults to False.
    
        """
        plot_pixel_space(roi.kins, connection=connection)
        
    
    @staticmethod
    def plot_intensity_histograms(roi:ROI, bins = 256):
        """Displays intensity histograms of images in a given ROI and allows scrolling through them via a slider.

        This method calculates and visualizes intensity histograms for a set of images 
        corresponding to a region of interest (ROI). The histograms are displayed on a 
        logarithmic scale, and an interactive slider allows the user to scroll through 
        the images and view their corresponding histograms.
    
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
        
    
    @staticmethod
    def plot_average_roi(roi:ROI, vmin=None, vmax=None, title=None):
        """Plots the averaged frames for the specified region of interest (ROI).

        This method displays the averaged data for a given ROI by plotting the mean of 
        the frames stored in `self.averaged_det_data`. The plot is displayed using the 
        `plot_roi_from_numpy` function.
    
        Args:
            roi_name (str): The name of the region of interest (ROI) whose averaged data 
                             is to be plotted. The averaged data should be stored in 
                             `self.averaged_det_data`.
            vmin (float, optional): The minimum value for the color scale. If None, 
                                     the minimum of the data is used. Default is None.
            vmax (float, optional): The maximum value for the color scale. If None, 
                                     the maximum of the data is used. Default is None.
            title (str, optional): The title of the plot. If None, a default title is used. 
                                   Default is None.
    
        Displays:
            A plot of the averaged frames for the specified ROI, with the given color scale.
        """
        
        plot_roi_from_numpy(roi.averaged_det_images, [0,-1,0,-1], 
                            f"Averaged Frames ", 
                            vmin=vmin, vmax=vmax )
    
    
    @staticmethod
    def plot_calculated_kins(roi:ROI, pupil_roi:ROI, vmin=None, vmax = None, title="Mapped kins onto pupil", cmap = "viridis"):
        """Plots the calculated k-in coordinates mapped onto the pupil function.

        This method visualizes the k-in coordinates by mapping them onto the pupil function of the 
        detector using a specified colormap.
    
        Args:
            roi_name (str): The name of the region of interest.
            vmin (float, optional): Minimum value for color scaling. Default is None (auto-scale).
            vmax (float, optional): Maximum value for color scaling. Default is None (auto-scale).
            title (str, optional): Title of the plot. Default is "Mapped kins onto pupil".
            cmap (str, optional): Colormap to use for visualization. Default is "viridis".
        """
        plot_map_on_detector(pupil_roi.averaged_det_images, roi.kin_coords, 
                             vmin, vmax, title, cmap, crop=False, roi= pupil_roi.coords)

    @staticmethod
    def plot_kouts(roi:ROI, vmin=None, vmax = None, title="Mapped kouts", cmap = "viridis"):
        
        """Plots the calculated k-out coordinates mapped onto the detector.

        This method visualizes the k-out coordinates by mapping them onto the detector data, allowing 
        for analysis of diffraction patterns.
    
        Args:
            roi_name (str): The name of the region of interest.
            vmin (float, optional): Minimum value for color scaling. Default is None (auto-scale).
            vmax (float, optional): Maximum value for color scaling. Default is None (auto-scale).
            title (str, optional): Title of the plot. Default is "Mapped kouts".
            cmap (str, optional): Colormap to use for visualization. Default is "viridis".
        """
        plot_map_on_detector(roi.averaged_det_images, roi.kout_coords, 
                             vmin, vmax, title, cmap, crop=False,roi= roi.coords)
    
    # def plot_average_full_detector(self, beamtime='new', fast_axis_steps=None, slow_axis = 0, directory=None, vmin1=None, vmax1=None, 
    #                                     vmin2=None, vmax2=None):
    #     """
    #     Plots the averaged full detector data in both linear and logarithmic scales.

    #     If the averaged full detector data is not already computed, it is calculated
    #     by averaging over all data files in `self.fnames`, depending on the beamtime mode.
    #     Hot pixels are masked from the final averaged data.

    #     Args:
    #         vmin1 (float, optional): Minimum value for the linear scale color map. Defaults to None.
    #         vmax1 (float, optional): Maximum value for the linear scale color map. Defaults to None.
    #         vmin2 (float, optional): Minimum value for the logarithmic scale color map. Defaults to None.
    #         vmax2 (float, optional): Maximum value for the logarithmic scale color map. Defaults to None.

    #     Displays:
    #         A matplotlib figure with two subplots: 
    #             - Left: Averaged detector data with linear intensity scale.
    #             - Right: Logarithmically scaled intensity plot.
    #     """

    #     if self.averaged_full_det_data is None:
    #         if beamtime == 'new':
            
    #             f0_name = os.path.join(directory,self.fnames[0])
    #             with h5py.File(f0_name,'r') as f:
    #                 data_avg = np.mean(f['/entry/data/data'][:,:,:], axis = 0)
                    
    #             for fname in tqdm(self.fnames[1:]):
    #                 file_name = os.path.join(directory,fname)
            
    #                 with h5py.File(file_name,'r') as f:
    #                     data_avg += np.mean(f['/entry/data/data'][:,:,:], axis = 0)
    #         else:
    #             data_avg = np.mean(stack_4d_data_old(directory, [0,-1,0,-1], 
    #                                                  fast_axis_steps, slow_axis), axis = (0,1))
    
    #         self.averaged_full_det_data = data_avg
    #         self.averaged_full_det_data = mask_hot_pixels(self.averaged_full_det_data)
            
    #     fig, axes = plt.subplots(1, 2, figsize=(10,5))

    #     log_data = np.log1p(self.averaged_full_det_data)

    #     intensity = axes[0].imshow(self.averaged_full_det_data, cmap='jet',  vmin = vmin1, vmax = vmax1)
    #     axes[0].set_title(f'Full Detector - Average')
    #     axes[0].axis('on')  
    #     plt.colorbar(intensity, ax=axes[0], fraction=0.046, pad=0.04)
    #     log_intensity = axes[1].imshow(log_data, cmap='jet',  vmin=vmin2, vmax=vmax2)
    #     axes[1].set_title(f'Log Scale')
    #     axes[1].axis('on')
    #     plt.colorbar(log_intensity, ax=axes[1], fraction=0.046, pad=0.04)

    #     plt.tight_layout()
    #     plt.show()
    
        
    # ################### Prepares the data ###################
    # def prepare_roi(self, roi_name:str, 
    #                 mask_val: float, 
    #                 mask_max_coh:bool = False,
    #                 mask_min_coh: bool = False,
    #                 pool_det = None, 
    #                 normalisation_roi = None
    #                 ):
    #     """
    #     Full preparation of the region of interest (ROI) after running add_roi.
    
    #     This function processes the given ROI by performing various steps, including:
    #     - Creating a 4D dataset for the ROI.
    #     - Normalizing the detector data (if applicable).
    #     - Pooling the detector space (if specified).
    #     - Averaging the frames of the ROI.
    #     - Calculating the k-vector for the ROI.
    
    #     Args:
    #         roi_name (str): The name of the ROI to prepare.
    #         mask_val (float): The value used for masking on the detector to include in the coordinates array.
    #         mask_max_coh (bool, optional): If True, masks the maximum coherent values. Defaults to False.
    #         mask_min_coh (bool, optional): If True, masks the minimum coherent values. Defaults to False.
    #         pool_det (tuple, optional): If passed, contains the kernel size, stride, and padding for pooling the detector space. Defaults to None.
    #         normalisation_roi (str, optional): The name of an ROI for normalization, if specified. Defaults to None.
        
    #     """
        
    #     self.make_4d_dataset(roi_name)

    #     if normalisation_roi is not None:
    #         self.normalise_detector(roi_name, normalisation_roi)
            
    #     if pool_det is not None:
    #         self.pool_detector_space(roi_name, *pool_det)
    #     self.make_kouts(roi_name=roi_name,mask_val= mask_val)

    # def prepare_kins(self, roi_name:str, 
    #                 ttheta: float, # Two Theta value of the reflection in degrees
    #                 streak_width = 1, 
    #                 streak_position = 'center', 
    #                 streak_offset = 0, 
    #                 percentage = 10, 
    #                 start_position = 'lowest', 
    #                 start_idx = None
    #                 ):
    #     """
    #     Prepares the k-in vectors for a given region of interest (ROI).

    #     This method processes the k-in vectors by computing their values based on the provided
    #     two-theta angle, then extracting a narrow streak that represents a '2D' projection, and selecting a specified portion of the streak.
    
    #     Args:
    #         roi_name (str): The name of the region of interest.
    #         ttheta (float): The two-theta angle of the reflection in degrees.
    #         streak_width (int, optional): The width of the extracted streak in pixels. Default is 1.
    #         streak_position (str, optional): The position of the streak relative to the main region.
    #             Options: 'center', 'top', 'bottom'. Default is 'center'.
    #         streak_offset (int, optional): An offset value to shift the streak selection. Default is 0.
    #         percentage (float, optional): The percentage of the streak to retain. Default is 10%.
    #         start_position (str, optional): The starting position for selection.
    #             Options: 'lowest' (bottom), 'highest' (top), or 'middle'. Default is 'lowest'.
    #         start_idx (int, optional): The specific index to start selection from. If provided, it overrides `start_position`.
    #     """
    #     ttheta_rad = np.deg2rad(ttheta) #Gold (400) reflection
    #     self.compute_kins(roi_name, est_ttheta = ttheta_rad)
    #     self.select_streak_region(self, roi_name, percentage=percentage, start_position=start_position, start_idx=start_idx)
    #     self.select_single_pixel_streak(self, roi_name, width=streak_width, position=streak_position, offset=streak_offset)

    
    # def prepare_coherent_images(self, roi_name:str, 
    #                             mask_region = None,
    #                             variance_threshold = None, 
    #                             order_imgs = True,
    #                             background_sigma = None, 
    #                             median_params = None, # tuple (kernel_size, stride, frac threshold)
    #                             bilateral_params = None, # tuple (sigma_spatial, sigma_range, kernel_size)
    #                             detect_params = None, #tuple (threshold, min_val, mask_val)
    #                             blur_sigma = None, # Gaussian blur the detected object
    #                             align_cohimgs = False,
    #                             mask_threshold = None
    #                             ):
    #     """
    #     Prepares the coherent images for a given region of interest (ROI) by applying a series of processing steps.
    
    #     This function performs multiple operations to process the ROI's coherent images, including:
    #     - Creating coherent images.
    #     - Masking regions (if specified).
    #     - Filtering images based on variance.
    #     - Ordering the pixels.
    #     - Applying background removal.
    #     - Making image dimensions even.
    #     - Applying median and bilateral filters.
    #     - Detecting objects and optionally blurring them.
    #     - Aligning the coherent images.
    
    #     Args:
    #         roi_name (str): The name of the ROI for which to prepare the coherent images.
    #         mask_region (tuple, optional): The region to mask from the coherent images. Defaults to None.
    #         variance_threshold (float, optional): Threshold for filtering the coherent images based on variance. Defaults to None.
    #         order_imgs (bool, optional): Whether to reorder the images' pixels. Defaults to True.
    #         background_sigma (float, optional): The standard deviation for Gaussian background removal. Defaults to None.
    #         median_params (tuple, optional): Parameters for median filtering (kernel_size, stride, threshold). Defaults to None.
    #         bilateral_params (tuple, optional): Parameters for bilateral filtering (sigma_spatial, sigma_range, kernel_size). Defaults to None.
    #         detect_params (tuple, optional): Parameters for object detection (threshold, min_val, mask_val). Defaults to None.
    #         blur_sigma (float, optional): The standard deviation for Gaussian blur applied to detected objects. Defaults to None.
    #         align_cohimgs (bool, optional): Whether to align the coherent images. Defaults to False.
    #         mask_threshold (float, optional): The threshold value for masking coherent images. Defaults to None.
    
    #     """
    #     self.make_coherent_images(roi_name=roi_name)
    #     if mask_region is not None:
    #         self.mask_region_cohimgs(roi_name, mask_region, mode = 'median')
    #     if variance_threshold is not None:
    #         self.filter_coherent_images(roi_name=roi_name, variance_threshold=variance_threshold)
            
    #     if order_imgs:
    #         self.order_pixels(roi_name)
    #     if mask_threshold is not None:
    #         self.mask_cohimgs_threshold(roi_name=roi_name, threshold_value= mask_threshold)
    #     if background_sigma is not None:
    #         self.remove_coh_background(roi_name, background_sigma) 
        
    #     self.even_dims_cohimages(roi_name=roi_name)
    #     if median_params is not None:
    #         self.apply_median_filter(roi_name, *median_params)

    #     if bilateral_params is not None:
    #         self.apply_bilateral_filter(roi_name, *bilateral_params)
            
    #     if detect_params is not None:
    #         self.detect_object(roi_name, *detect_params)

    #         if blur_sigma is not None:
    #             self.blur_detected_objs(roi_name, blur_sigma)

    #     if align_cohimgs:
    #         self.align_coherent_images(roi_name)
        
