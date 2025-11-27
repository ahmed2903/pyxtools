# Imports data from Ptychography experiments
# 
# Author: Ahmed H. Mokhtar 
# Email: ahmed.mokhtar@desy.de
# Date : Feb 2025

import numpy as np 
import h5py
import matplotlib.pyplot as plt
import copy
from functools import wraps
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import os 

from .data_fs import *
from .kvectors import make_coordinates, compute_vectors, optimise_kin, reverse_kins_to_pixels, rotation_matrix, calc_qvec, extract_parallel_line, extract_streak_region

from .utils import time_it, energy2wavelength_a
from .geometry import *

    

@dataclass
class Exp:
    
    directory : str
    scan_num: int

    det_psize: int  
    energy: float #keV
    
    
    step_size: float # Angstroms
    slow_axis: int  # 0 for x, 1 for y
    
    beamtime: str = field(default = 'new', repr = False) # new or old
    
    det_distance: float = field(default = None, repr=True) # microns
    wavelength: float = field(init=False)
    fast_axis_steps: int = field(default = None, repr = True) # steps

    detector: str = field(default = 'Eiger', repr = True)
    
    def __post_init__(self):

        self.wavelength = energy2wavelength_a(self.energy)
        base_path = Path(self.directory)
        if self.detector == 'Eiger':
            
            new_dir = f"Scan_{self.scan_num}"

        elif self.detector == 'Lambda':
            new_dir = f"Scan_{self.scan_num}_Lambda"

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
            
            process_params.attrs['roi'] = json.dumps(self.coords)
            process_params.attrs['step_size'] = self.step_size
            process_params.attrs['det_distance'] = self.det_distance

            # for i, func in enumerate(self.log_params):
            #     # self.log_roi_params[roi_name] is a list, each element correponds to a function 
            #     # func is then a dictionary
            #     # func["functions"] will be the function name
            #     # func_name = func['function']
            #     # func.pop('function')
            #     # process_params.attrs[func['function']] = json.dumps(func[args]) 
            #     process_params.attrs[func['function']] = json.dumps(func['args']) +' '+ json.dumps(func['kwargs'])

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
            params = h5f["processing_params"]
            
            self.step_size = params.attrs['step_size']
            
            
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
        """Save a deep copy of the current state for potential rollback."""
        state = copy.deepcopy({
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
            if key not in ("checkpoint_stack", "log_params")  # don't save logs or stack itself
        })
        self.checkpoint_stack.append(state)
        print(f"[ROI] Checkpoint #{len(self.checkpoint_stack)} created.")

    def restore_checkpoint(self):
        """Restore the most recent checkpoint."""
        if not self.checkpoint_stack:
            print("[ROI] No checkpoints to restore.")
            return

        last_state = self.checkpoint_stack.pop()
        for key, value in last_state.items():
            setattr(self, key, value)
    
    def auto_checkpoint(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.checkpoint_state()
            return func(self, *args, **kwargs)
        return wrapper
    
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
def make_4d_dataset(roi: ROI, num_jobs = 64):
    
    """Creates a 4D dataset from a region of interest (ROI) on the detector.

    Args:
        roi_coords (list): The region of interest (ROI) to process.

    Returns:
        The 4D dataset for the given ROI.
    """

    
    if roi.beamtime == 'old':
        # Just for naming
        direc = os.path.join(roi.directory , f"Scan_{roi.scan_num}_data_000001.h5")
        if roi.fast_axis_steps is None:
            raise ValueError("fast_axis_steps is required")            
        
        data_4d = stack_4d_data_old(direc, 
                                    roi.coords, 
                                    roi.fast_axis_steps, 
                                    roi.slow_axis)
        
    else:
        
        direc = roi.directory
        print(direc)
        fnames = list_datafiles(direc)[:-2]     
        if roi.detector == 'Eiger':
            data_4d = stack_4d_data(direc, fnames, 
                                                    roi.coords, 
                                                    slow_axis = roi.slow_axis, 
                                                    conc=True, 
                                                    num_jobs=num_jobs)
            
        elif roi.detector == 'Lambda':
            data_4d = stack_4d_data_lambda(direc, fnames, 
                                                    roi.coords, 
                                                    slow_axis = roi.slow_axis, 
                                                    conc=True, 
                                                    num_jobs=num_jobs)
            
    roi.data_4d = mask_hot_pixels(data_4d, 
                                mask_max_coh = False, 
                                mask_min_coh = False)
    
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
    # pupil_size = estimate_pupil_size(pupil_roi.averaged_det_images, 
    #                    mask_val=mask_val,
    #                    pixel_size=pupil_roi.det_psize)

    array = pupil_roi.averaged_det_images
    pixel_size = pupil_roi.det_psize

    min_lx = array.shape[0] /3
    min_ly = array.shape[1] /3
    
    nx , ny = np.where(array>mask_val)

    x_lengths = []
    for x in np.unique(nx):
        y_indices = np.where(array[x,:] > mask_val)[0]
        if len(y_indices) > min_lx:
            x_lengths.append(y_indices[-1] - y_indices[0] + 1)

    
    y_lengths = []
    for y in np.unique(ny):
        x_indices = np.where(array[:,y] > mask_val)[0]
        if len(x_indices) > min_ly:
            y_lengths.append(x_indices[-1] - x_indices[0] + 1)

    x_lengths = np.array(x_lengths) 
    y_lengths = np.array(y_lengths)

    avg_x = np.max(x_lengths) * pixel_size
    avg_y = np.max(y_lengths) * pixel_size

    return avg_x, avg_y
    
    # return pupil_size

def add_lens(focal_length:float, height:float):
    """
    Adds a lens with given parameters to the internal lens parameter dictionary.

    Args:
        lens_name (str): The name/key to identify the lens.
        focal_length (float): The focal length of the lens in microns.
        height (float): The physical aperture height of the lens in microns.
    """
    lens_na = calculate_NA(focal_length, height)

    return lens_na

def estimate_detector_distance(na1, na2, pupil_size):
    """
    Estimates the average detector distance based on the range of numerical apertures (NA)
    from the added lenses and current pupil size.

    Updates:
        self.exp_params['det_distance']: Estimated detector distance in microns based on optical geometry.
    """
    largest_na = 0
    smallest_na = 1e4
    
    for na in [na1, na2]:

        largest_na = max(largest_na, na)
        smallest_na = min(smallest_na, na)    
    
    distance1 = estimate_detector_distance_from_NA(largest_na, pupil_size=max(pupil_size))
    distance2 = estimate_detector_distance_from_NA(smallest_na, pupil_size=min(pupil_size))

    print(f"First detector distance is {distance1} microns")
    print(f"Second detector distance is {distance2} mircons")
    
    return (distance1+distance2)/2


################### prepare detector roi ###################

@log_roi_params
def pool_detector_space(roi:ROI, kernel_size, padding=0):
    """Performs pooling on the detector space for a given region of interest (ROI).
    
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
def normalise_detector(cls, roi:ROI, reference_roi:ROI):
    """Normalizes the detector data based on the reference ROI's peak intensity.

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
def mask_region_detector(roi:ROI, region, mode = 'median'):
    """Masks a specific region of the detector data within the given ROI.

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
def make_kouts( roi:ROI, mask_val):
    """Computes the k-space vectors for a given region of interest (ROI).

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



def select_single_pixel_streak(roi:ROI, width=1, offset=0, inplace = False):

    """
    Selects a single-pixel-wide streak along a specified direction in the region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest.
        width (int, optional): The width of the streak to extract. Default is 1.
        position (str, optional): The position of the streak relative to the main region.
            Options: 'center', 'top', 'bottom'. Default is 'center'.
        offset (int, optional): An offset value to shift the streak selection up or down. Default is 0.
        inplace (bool): if True, it changes the kin_coords, kins, kouts, and kout_coords dict for that roi, if false, if returns the mask. (default = False)

    """
    mask = extract_parallel_line(roi.kins[:,:2], width=width, offset=offset)
    
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
        return mask
    
def select_streak_region(roi:ROI, percentage=10, start_position='lowest', start_idx=None, inplace = False):
    """
    Selects a region of the streak based on a percentage of its total size.

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

def order_pixels(roi:ROI):
    """Reorders the pixels of coherent images and corresponding k-vectors based on their distance from the center.

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
def make_coherent_images(roi:ROI):
    
    """
    Generates a list of coherent images for a given region of interest (ROI).
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
def filter_coherent_images(roi:ROI, variance_threshold):
    """Filters coherent images based on variance threshold.

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
def remove_coh_background(roi:ROI, sigma, num_jobs = -1):
    """Removes the background noise from coherent images for a given region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent 
                         images will be processed.
        sigma (float): The standard deviation of the Gaussian filter used for 
                       background removal. A higher value will smooth the image 
                       more, removing more background noise but potentially 
                       blurring fine details.
    """
    
    roi.coherent_imgs = remove_background_parallel(roi.coherent_imgs, sigma=sigma, n_jobs=num_jobs)
    
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
def apply_median_filter(roi:ROI, kernel_size, stride, threshold, num_jobs = -1):
    """Applies a median filter to coherent images for a given region of interest (ROI).

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
def apply_bilateral_filter(self, roi:ROI, sigma_spatial, sigma_range, kernel_size, num_jobs = -1):
    """Applies a bilateral filter to the coherent images for a given region of interest (ROI).

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
    roi.coherent_imgs = bilateral_filter(roi.coherent_imgs, sigma_spatial, 
                                                                  sigma_range, 
                                                                  kernel_size, n_jobs=num_jobs)
    
@log_roi_params
def detect_object(roi:ROI, threshold, min_val, max_val, num_jobs = -1):
    """Detects objects within the specified region of interest (ROI) and filters based on intensity.
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
def blur_detected_objs(roi:ROI, sigma):
    """Applies Gaussian blur to the detected objects in the specified region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose detected object 
                         images will be blurred.
        sigma (float): The standard deviation of the Gaussian filter. Larger values 
                       result in more blurring.

    """
    roi.detected_objects = apply_gaussian_blur(roi.detected_objects, sigma=sigma)
    
@log_roi_params
def mask_cohimgs_threshold(roi:ROI, threshold_value):
    """Applies a threshold mask to the coherent images in the given ROI.
    
    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent images 
                         will be thresholded.
        threshold_value (float): The threshold value used for masking the coherent images. 
                                  Values below this threshold will be masked.

    """
    roi.coherent_imgs = threshold_data(roi.coherent_imgs, threshold_value)
    
@log_roi_params
def mask_region_cohimgs(roi:ROI, region, mode = 'zeros'):
    """Masks a specific region of coherent images within the given ROI.

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
        
def align_coherent_images(roi:ROI, ref_idx = None):
    """Aligns a list of coherent images for a given region of interest (ROI).

    Args:
        roi_name (str): The name of the region of interest (ROI) whose coherent images
                         will be aligned.

    """
    print('aligned coherent images')
    roi.coherent_imgs = align_images(roi.coherent_imgs, ref_idx = ref_idx)

def flip_images(roi:ROI, flip_mode):
    """
    Flips images based on the selected mode.
    
    Args:
        images (np.ndarray): A batch of images with shape (N, H, W).
        flip_mode (str): The flipping mode, one of:
                        - "xy"  (original)
                        - "x_neg_y" (flip y-axis)
                        - "neg_x_y" (flip x-axis)
                        - "neg_x_neg_y" (flip both axes)

    Returns:
        np.ndarray: The transformed batch of images.
    """
    if flip_mode == "xy":
        return roi.coherent_imgs  # No flip
    elif flip_mode == "x_neg_y":
        return np.flip(roi.coherent_imgs, axis=1)  # Flip along y-axis
    elif flip_mode == "neg_x_y":
        return np.flip(roi.coherent_imgs, axis=2)  # Flip along x-axis
    elif flip_mode == "neg_x_neg_y":
        return np.flip(roi.coherent_imgs, axis=(1, 2))  # Flip along both axes
    else:
        raise ValueError(f"Invalid flip_mode: {flip_mode}")
        