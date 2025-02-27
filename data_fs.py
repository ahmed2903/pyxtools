import numpy as np
import time
from tqdm.notebook import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import concurrent.futures
import h5py
import os
from skimage.measure import shannon_entropy
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from scipy.ndimage import zoom 
from joblib import Parallel, delayed
import torch 
import torch.functional as F

from . import xrays_fs as xf
from . import general_fs as gf


def load_hdf_roi(data_folder, f_name, roi):
    """
    loads the data from a h5 file into a numpy array for a region of interest
    """
    
    file_path = data_folder + "/" + f_name
    
    with h5py.File(file_path,'r') as f:
        
        data = f['/entry/data/data'][:,roi[0]:roi[1], roi[2]:roi[3]]
        
    return data

def load_hdf(data_folder, f_name):
    """
    loads the data from a h5 file into a numpy array for a region of interest
    """
    
    file_path = data_folder + "/" + f_name
    
    with h5py.File(file_path,'r') as f:
        
        data = f['/entry/data/data']
        
    return data

def list_datafiles(data_folder):
    """
    lists all h5 data files in a folder 
    """
    f_names = []
    
    for filename in sorted(os.listdir(data_folder)):
        if filename.endswith(".h5"):
            f_names.append(filename)
                
    return f_names



def average_data(data_folder, names_array, roi, conc=False):
    
    """
    averages all the data in a NxM scan
    first averages the data vertically in a roi
    then horizontally
    """
    
    n = len(names_array) # number of slow scans 
    
    nx = roi[1] - roi[0] # roi vertical size
    ny = roi[3] - roi[2] # roi horizontal size
    
    t1=time.perf_counter()

    if conc: 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_data = list(executor.map(load_hdf_roi, [data_folder]*len(names_array), names_array, [roi]*len(names_array)))

    else: 
        all_data = []
        for i in range(len(names_array)):
            all_data.append(  load_hdf_roi(data_folder, names_array[i], roi)  )

            
    stacked_data = np.concatenate(all_data,axis=0)
    average_data = np.mean(stacked_data, axis=0)

    t2=time.perf_counter()

    print(f"finsihed in {t2-t1}")
    
    return average_data

    
def stack_data(data_folder, names_array, roi, conc = True):
    """
    Stacks all the data along the first dimension
    such that output ha shape (NxM, x,y)
    where N is the number of files, and M is the number of frames per file
    """
    
    nx = roi[1] - roi[0] # roi vertical size
    ny = roi[3] - roi[2] # roi horizontal size
    t1=time.perf_counter()

    if conc:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_data = list(executor.map(load_hdf_roi, [data_folder]*len(names_array), names_array, [roi]*len(names_array)))

    else:
        all_data = []
        for i in range(len(names_array)):
            all_data.append(  load_hdf_roi(data_folder, names_array[i], roi)  )
    
    stacked_data = np.concatenate(all_data,axis=0)

    t2=time.perf_counter()

    print(f"finsihed in {t2-t1}")
    
    return stacked_data
    

def stack_4d_data(data_folder,names_array,roi, conc = False):

    
    nx = roi[1] - roi[0] # roi vertical size
    ny = roi[3] - roi[2] # roi horizontal size
    

    t1=time.perf_counter()
    if conc:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_data = list(executor.map(load_hdf_roi, [data_folder]*len(names_array), names_array, [roi]*len(names_array)))

    else:
        all_data = []
        for i in range(len(names_array)):
            all_data.append(  load_hdf_roi(data_folder, names_array[i], roi)  )
        all_data = np.array(all_data)
    t2=time.perf_counter()

    print(f"finsihed in {t2-t1}")
    
        
    stacked_data = np.stack(all_data, axis=0)
    
    return stacked_data

def make_coherent_image(data: np.ndarray, pixel_idx:np.ndarray):
    
    """
    Makes a single coherent image from a 4D data set that is size (M,N,x,y) using one pixel 
    
    M is the slow scan 
    N is the fast scan
    x,y are detector size
    
    pixel_idx: the index of the pixel in the detector to be used in the coherent image
    """
    
    px = pixel_idx[0]
    py = pixel_idx[1]
    
    coherent_image = data[:,:,py,px]

    return coherent_image

def sum_pool2d_array(input_array, kernel_size, stride=None, padding=0):
    """
    Perform sum pooling on a 4D NumPy array or PyTorch tensor.
    
    Parameters:
    - input_array: 4D NumPy array or PyTorch tensor (batch_size, channels, height, width)
    - kernel_size: Size of the pooling kernel (e.g., 2 for 2x2 pooling).
    - stride: Stride of the pooling operation. Default is kernel_size.
    - padding: Amount of zero-padding added to both sides of the input. Default is 0.
    
    Returns:
    - output: Pooled array as a NumPy array.
    """
    if stride is None:
        stride = kernel_size  # Default stride equals kernel size
    
    is_numpy = isinstance(input_array, np.ndarray)
    
    # Convert NumPy array to PyTorch tensor if necessary
    if is_numpy:
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
    else:
        input_tensor = input_array  
    
    # Perform sum pooling
    pooled = F.avg_pool2d(input_tensor, kernel_size, stride, padding) * (kernel_size**2)
    
    # Convert back to NumPy array if input was NumPy
    if is_numpy:
        return pooled.numpy()
        
    return pooled

def make_detector_image(data: np.ndarray, position_idx:np.ndarray):
    
    """
    Makes a single detector image from a 4D data set that is size (M,N,x,y) using one location 
    
    M is the slow scan 
    N is the fast scan
    x,y are detector size
    
    position_idx: the index of the location in the detector to be used in the detector image
    """
    
    px = position_idx[0]
    py = position_idx[1]
    
    detector_image = data[py,px,:,:]

    return detector_image

def make_coordinates(array, mask_val, roi, crop):

    
    if crop:
        array = array[roi[0]:roi[1], roi[2]:roi[3]]
        

    indices = np.where(array > mask_val)
    coords = np.array([(int(i)+ roi[0], int(j)+roi[2]) for i, j in zip(indices[0], indices[1])])

    return coords

def mask_hot_pixels(array):
    """
    Masks hot pixels (maximum values in the last two dimensions) in a 4D array.
    
    Parameters:
    - array: 4D numpy array with shape (N, M, x, y) where the last two dimensions
      represent the detector image (x, y).
      
    Returns:
    - masked_array: The array with hot pixels masked (replaced with NaN or 0).
    """
    if len(array.shape)>3:
        # Find the maximum value along the last two dimensions (x, y) for each (N, M)
        max_values = np.max(array, axis=(2, 3), keepdims=True)
    if len(array.shape) == 2:
        max_values = np.max(array, axis=(0, 1), keepdims=True)
    else: 
        max_values = np.max(array, axis=(1, 2), keepdims=True)
    # Mask all the values equal to the maximum value in the 2D slice with NaN (or 0)
    masked_array = np.where(array == max_values, 0.0, array)
    
    return masked_array

def estimate_pupil_size(array, mask_val, pixel_size, pupil_roi, crop=True):
    """
    Computes the average length of a box with jagged edges in x and y directions.

    Args:
        array (2D ndarray): A 2D array where the box is represented by non-zero values.
        mask_val (float): value of the mask
        pixel_size (float): size of the detector pixel
        pupil_roi (list): a 4 element list containing the roi of the pupil [xs, xe, ys, ye]

    Returns:
        tuple: (average_x_length, average_y_length)
    """
    if crop: 
        array = array[pupil_roi[0]:pupil_roi[1], pupil_roi[2]:pupil_roi[3]]

    min_lx = array.shape[0] /2
    min_ly = array.shape[1] /2
    
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

    avg_x = np.mean(x_lengths) * pixel_size
    avg_y = np.mean(y_lengths) * pixel_size

    return avg_x, avg_y

def get_kin(array, kins, roi, mask_val, ttheta_real, crop, det_focus_distance, det_crys_distance, det_psize, det_cen_pixel, wavelength, method, gtol):

    if crop:
        array = array[:,roi[0]:roi[1],roi[2]:roi[3]]

    coords = make_coordinates(array, mask_val, roi, crop=crop)

    kouts = xf.compute_vectors(coords, det_crys_distance, det_psize, det_cen_pixel, wavelength)

    kins_avg = np.mean(kins, axis = 0, keepdims = True )
    
    ttheta_est = np.mean(xf.calc_ttheta_from_kout(kouts, kins_avg))
    print(f"The estimated two theta value for this configuration is: {np.rad2deg(ttheta_est)}")
    qvec_0 = xf.calc_qvec(kouts, kins_avg, ttheta = ttheta_real, wavelength= wavelength)

    ki, opt_angles = xf.optimise_kin(qvec_0, ttheta_real, kouts, wavelength, method, gtol)
    
    map_kins = xf.reverse_kins_to_pixels(ki, det_psize, det_focus_distance,  det_cen_pixel)

    return map_kins, ki, opt_angles


def downsample_array(arr, new_shape):
    """
    Downsample a 2D array by taking mean of non-overlapping rectangular blocks
    
    Args:
        arr (np.ndarray): Input 2D array
        new_shape (tuple): Desired output shape
    
    Returns:
        np.ndarray: Downsampled array
    """
    # Calculate exact block sizes
    orig_rows, orig_cols = arr.shape
    new_rows, new_cols = new_shape
    
    # Calculate block dimensions
    row_factor = orig_rows // new_rows
    col_factor = orig_cols // new_cols
    
    # Trim array to ensure exact division
    trimmed_arr = arr[:new_rows*row_factor, :new_cols*col_factor]
    
    # Reshape and compute mean
    downsampled = trimmed_arr.reshape(
        new_rows, row_factor, 
        new_cols, col_factor
    ).max(axis=(1, 3))
    
    return downsampled

def make_2dimensions_even(array_list):
    """
    Takes a list of NumPy arrays and ensures that all arrays have even dimensions.
    If either dimension is odd, the array is padded at the end to make it even.

    Parameters:
        array_list (list of numpy.ndarray): List of arrays to process.

    Returns:
        list of numpy.ndarray: List of arrays with even dimensions.
    """
    # Get the shape of the first array as a reference
    ref_shp = array_list[0].shape

    # Determine the target shape (even dimensions)
    target_shape = list(ref_shp)
    
    for i in range(len(target_shape)):
        if target_shape[i] % 2 != 0:  # If dimension is odd
            target_shape[i] += 1  # Make it even by adding 1

    # Calculate the padding required for each dimension
    padding = (0,target_shape[0] - ref_shp[0], 
               0, target_shape[1] - ref_shp[1] ) 
        
    # Pad the arrays if necessary
    padded_arrays = []
    for array in array_list:
        
        # Apply padding
        padded_array = gf.pad_2d(array, *padding)
        padded_arrays.append(padded_array)

    return padded_arrays


def filter_images(images, coords, variance_threshold, kin_coords=None, **kwargs):
    """
    Filters images based on variance, with an optional entropy threshold. 
    Can also filter additional coordinates (kinematic coordinates) if provided.

    Parameters:
    -----------
    images : List of 2D images to filter.
    coords : List of coordinates associated with the images.
    variance_threshold : Minimum variance required to keep an image.
    kin_coords : Additional coordinates to filter along with `coords`.
    entropy_threshold : Minimum Shannon entropy required to keep an image.

    Returns:
    --------
    filtered_images : The filtered images.
    filtered_coords : The filtered coordinates.
    filtered_kin_coords : The filtered kinematic coordinates (if `kin_coords` is provided).
    entropies : The entropy values for the filtered images (only if `entropy_threshold` is provided).
    """
    entropy_threshold = kwargs.get('entropy_threshold', None)  # Check for entropy threshold
    filtered_images = []
    filtered_coords = []
    filtered_kin_coords = [] if kin_coords is not None else None
    entropies = [] if entropy_threshold is not None else None

    for i, (img, coord) in enumerate(zip(images, coords)):
        variance = np.var(img)
        keep = variance >= variance_threshold  # Initial condition based on variance

        # If entropy threshold is provided, calculate entropy and check the condition
        if entropy_threshold is not None:
            entropy = shannon_entropy(img)
            if keep and entropy >= entropy_threshold:
                keep = True
                if entropies is not None:
                    entropies.append(entropy)
            else:
                keep = False

        # Add image and coordinates to the filtered lists if conditions are met
        if keep:
            filtered_images.append(img)
            filtered_coords.append(coord)
            if kin_coords is not None:
                filtered_kin_coords.append(kin_coords[i])

    # Return results
    if kin_coords is not None:
        if entropies is not None:
            return filtered_images, filtered_coords, filtered_kin_coords, entropies
        return filtered_images, filtered_coords, filtered_kin_coords
    else:
        if entropies is not None:
            return filtered_images, filtered_coords, entropies
        return filtered_images, filtered_coords
    



def upsample_image(im, zoom_factor):
    """
    Upsample a single image using zoom.
    """
    return zoom(im, zoom_factor, order=3).astype(complex)

def upsample_images(images, zoom_factor, n_jobs=-1):
    """
    Upsample a list of images in parallel.

    Parameters:
        images (list of numpy.ndarray): List of input images.
        zoom_factor (float or tuple): Zoom factor for upsampling.
        n_jobs (int): Number of CPU cores to use. Default is -1 (use all available cores).

    Returns:
        numpy.ndarray: Array of upsampled images.
    """
    # Use joblib to parallelize the upsampling
    up_images = Parallel(n_jobs=n_jobs)(delayed(upsample_image)(im, zoom_factor) for im in images)

    return np.array(up_images)