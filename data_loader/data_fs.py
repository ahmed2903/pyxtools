import numpy as np
import time
from tqdm.notebook import tqdm, trange

    
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import concurrent.futures
from joblib import Parallel, delayed

from multiprocessing import Pool


import h5py
import os
from skimage.measure import shannon_entropy
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from scipy.ndimage import zoom 
import torch 
import torch.nn.functional as F
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift, gaussian_filter, binary_dilation
import numpy as np
import scipy.ndimage as ndimage
#import cv2

import scipy
import scipy.fft
from .utils import time_it

################# Load Data #################
    
def load_hdf_roi(args):
    """
    loads the data from a h5 file into a numpy array for a region of interest
    """
    data_folder, f_name, roi = args
    file_path = os.path.join(data_folder, f_name)
    
    with h5py.File(file_path,'r') as f:
        
        data = f['/entry/data/data'][:,roi[0]:roi[1], roi[2]:roi[3]]
        
    return data
    
def load_hdf(data_folder, f_name):
    """
    loads the data from a h5 file into a numpy array for a region of interest
    """
    
    file_path = os.path.join(data_folder, f_name)
    
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

@time_it
def average_data_old(file_path,roi):
        
    with h5py.File(file_path,'r') as f:
        
        data = np.mean(f['/entry/data/data'][:,roi[0]:roi[1], roi[2]:roi[3]], axis = 0)
        
    return data

@time_it 
def average_data(data_folder, names_array, roi, conc=False, num_jobs=16):
    
    """
    averages all the data in a NxM scan
    first averages the data vertically in a roi
    then horizontally
    """
    
    n = len(names_array) # number of slow scans 
    
    nx = roi[1] - roi[0] # roi vertical size
    ny = roi[3] - roi[2] # roi horizontal size

    args_list = [(data_folder, name, roi) for name in names_array]
    
    if conc: 
        #with concurrent.futures.ProcessPoolExecutor() as executor:
        #    all_data = list(executor.map(load_hdf_roi, [data_folder]*len(names_array), names_array, [roi]*len(names_array)))
        with Pool(processes=num_jobs) as pool:
            all_data = list(pool.imap(load_hdf_roi, args_list))
    
    else: 
        all_data = []
        for i in range(len(names_array)):
            all_data.append(  load_hdf_roi(data_folder, names_array[i], roi)  )

            
    stacked_data = np.concatenate(all_data,axis=0)
    average_data = np.mean(stacked_data, axis=0)

    return average_data

@time_it   
def stack_data(data_folder, names_array, roi, conc = True):
    """
    Stacks all the data along the first dimension
    such that output ha shape (NxM, x,y)
    where N is the number of files, and M is the number of frames per file
    """
    
    nx = roi[1] - roi[0] # roi vertical size
    ny = roi[3] - roi[2] # roi horizontal size
    
    if conc:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_data = list(executor.map(load_hdf_roi, [data_folder]*len(names_array), names_array, [roi]*len(names_array)))

    else:
        all_data = []
        for i in range(len(names_array)):
            all_data.append(  load_hdf_roi(data_folder, names_array[i], roi)  )
    
    stacked_data = np.concatenate(all_data,axis=0)
    
    return stacked_data
    
@time_it 
def stack_4d_data(data_folder, names_array, roi, slow_axis = 0, conc = False, num_jobs = 4):

    
    nx = roi[1] - roi[0] # roi vertical size
    ny = roi[3] - roi[2] # roi horizontal size
    
    args_list = [(data_folder, name, roi) for name in names_array]
    if conc:
        #with concurrent.futures.ProcessPoolExecutor() as executor:
        #    all_data = list(executor.map(load_hdf_roi, [data_folder]*len(names_array), names_array, [roi]*len(names_array)))
        with Pool(processes=num_jobs) as pool:
            all_data = list(pool.imap(load_hdf_roi, args_list))        
    else:
        all_data = []
        for i in range(len(names_array)):
            all_data.append(load_hdf_roi(data_folder, names_array[i], roi)  )
        all_data = np.array(all_data)
        
    stacked_data = np.stack(all_data, axis=0)

    if slow_axis == 0: 

        stacked_data = np.transpose(stacked_data, (1,0,2,3))
    
    return stacked_data
    
def print_hdf5_keys(file_path):
    """
    Prints all keys (groups and datasets) in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
    """
    def print_keys(name, obj):
        """
        Recursive function to print keys in the HDF5 file.
        """
        print(name)  # Print the key (group or dataset name)

    # Open the HDF5 file
    with h5py.File(file_path, "r") as f:
        # Traverse the file and print all keys
        print(f"Keys in {file_path}:")
        f.visititems(print_keys)

@time_it        
def stack_4d_data_old(file_path,roi, fast_axis_steps, slow_axis = 0):

    
    nx = roi[1] - roi[0] # roi vertical size
    ny = roi[3] - roi[2] # roi horizontal size
        
    # Open the input HDF5 file
    with h5py.File(file_path, "r") as f:
        # Assume the dataset is named 'data' (change this if needed)
        if "entry/data/data" not in f:
            raise ValueError("The input file does not contain a dataset named 'data'.")

        # Load the dataset
        data = f["entry/data/data"][:,roi[0]:roi[1],roi[2]:roi[3]]  # Load the entire dataset into memory
        dims = data.shape
        slow_axis_steps = dims[0]//fast_axis_steps
        data = data.reshape(slow_axis_steps,fast_axis_steps,dims[1],dims[2])
    
    if slow_axis == 0: 
        data = np.transpose(data, (1,0,2,3))

    print(f"The shape of the data is {data.shape}")
    return data


################# Process Data #################
@time_it
def sum_pool2d_array(input_array, kernel_size, stride=None, padding=0):
    """
    Perform sum pooling on a 4D NumPy array or PyTorch tensor.
    
    Parameters:
    - input_array: 4D NumPy array or PyTorch tensor
    - kernel_size: Size of the pooling kernel.
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
    pooled = F.avg_pool2d(input_tensor, kernel_size, stride, padding) #* (kernel_size**2)
    
    # Convert back to NumPy array if input was NumPy
    if is_numpy:
        return pooled.numpy()
        
    return pooled
    
def make_coherent_image(data: np.ndarray, pixel_idx:np.ndarray, slow_axis = 0):
    
    """
    Makes a single coherent image from a 4D data set that is size (M,N,x,y) using one pixel 
    
    M is the slow scan 
    N is the fast scan
    x,y are detector size
    
    pixel_idx: the index of the pixel in the detector to be used in the coherent image
    """
    
    px = pixel_idx[0]
    py = pixel_idx[1]
    
    coherent_image = data[:,:,px,py]

    return coherent_image
    


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
    
    detector_image = data[px,py,:,:]

    return detector_image




@time_it   
def mask_hot_pixels(array, mask_max_coh=False, mask_min_coh= False):
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
        if mask_max_coh:
            max_values_coh = np.max(array, axis=(0, 1), keepdims=True)
        if mask_min_coh:
            min_values_coh = np.min(array, axis=(0, 1), keepdims=True)
            
    if len(array.shape) == 2:
        max_values = np.max(array, axis=(0, 1), keepdims=True)
    else: 
        max_values = np.max(array, axis=(1, 2), keepdims=True)
    # Mask all the values equal to the maximum value in the 2D slice with NaN (or 0)
    masked_array = np.where(array == max_values, 0.0, array)

    if mask_max_coh or mask_min_coh:
        med = np.median(masked_array, axis = (0,1))
    
    if mask_max_coh:
        masked_array = np.where(masked_array == max_values_coh, med, masked_array)
    if mask_min_coh:
        masked_array = np.where(masked_array == min_values_coh, med, masked_array)
    
    return masked_array

def estimate_pupil_size(array, mask_val, pixel_size, pupil_roi=None, crop=True):
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
    if pupil_roi is not None: 
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

def make_2dimensions_even(array_list, mode='constant', num_jobs=-1, **kwargs):
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
    padding = ((0,target_shape[0] - ref_shp[0]), (0,target_shape[1] - ref_shp[1]))
    
    def process_array(array):
        padded_array = np.pad(array, padding, mode=mode, **kwargs)
        return padded_array
        
    padded_arrays = Parallel(n_jobs=num_jobs)(delayed(process_array)(arr) for arr in array_list)
    
    return np.array(padded_arrays)

@time_it
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
        return np.array(filtered_images), np.array(filtered_coords), np.array(filtered_kin_coords)
    else:
        return np.array(filtered_images), np.array(filtered_coords)
    



def phase_correlation(ref_img, shifted_img):
    """Compute shift between two images using phase correlation."""
    # Compute Fourier Transforms
    ref_fft = scipy.fft.fft2(ref_img)
    shifted_fft = scipy.fft.fft2(shifted_img)
    
    # Compute cross-power spectrum
    cross_power = (ref_fft * np.conj(shifted_fft)) / np.abs(ref_fft * np.conj(shifted_fft))
    
    # Inverse FFT to get correlation peak
    correlation = scipy.fft.ifft2(cross_power)
    correlation = np.abs(scipy.fft.fftshift(correlation))

    # Find peak location (gives the shift)
    max_loc = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = np.array(max_loc) - np.array(ref_img.shape) // 2  # Shift relative to center
    return shift  # Invert shift to align image
    
def align_images(image_list):
    """Align a list of shifted images based on the first image.

    This function aligns the images by calculating the shift between the reference image 
    (the first image in the list) and each subsequent image using phase correlation.
    The aligned images are returned as a list.

    Args:
        image_list (list of np.ndarray): A list of images (numpy arrays) to be aligned.

    Returns:
        list of np.ndarray: A list of aligned images.
    
    """
    aligned_images = []
    ref_img = image_list[0]  # Use first image as reference
    aligned_images.append(ref_img)
    
    for img in image_list[1:]:
        avg = np.mean(np.array(aligned_images), axis = 0)    
        shift = phase_correlation(avg, img)  # Find shift
        aligned_img = scipy.ndimage.shift(img, shift, mode='constant')  # Apply shift
        aligned_images.append(aligned_img)

    aligned_images = np.array(aligned_images)
    return aligned_images


def remove_background(image, sigma=20):
    """Removes smooth background by subtracting a Gaussian-blurred version of the image."""
    background = ndimage.gaussian_filter(image, sigma=sigma)
    tmp = image-background
    inverted = np.max(tmp) - tmp
    return inverted

def remove_background_parallel(image_list, sigma=20, n_jobs = 8):
    """Applies background removal to all images in a list using multiprocessing."""
    result = Parallel(n_jobs=n_jobs)(delayed(remove_background)(im, sigma) for im in image_list)
    result = np.array(result)
    return result
    
def upsample_image(im, zoom_factor):
    """
    Upsample a single image using zoom.
    """
    return zoom(im, zoom_factor, order=3).astype(complex)

def upsample_images(images, zoom_factor, n_jobs=4):
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

def detect_object(image, threshold_factor=0.5):
    """Detects the object by thresholding and finding the largest connected component."""
    # Normalize the image
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)

    # Apply adaptive thresholding
    threshold = threshold_factor * np.max(image_norm)
    binary_mask = image_norm > threshold  # Object is where intensity is above the threshold

    # Label connected components
    labeled, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return None  # No object detected

    # Find the largest connected component (assuming it's the object)
    object_sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
    largest_object_label = np.argmax(object_sizes[1:]) + 1  # Ignore background (label 0)
    
    return labeled == largest_object_label  # Return object mask

@time_it
def detect_obj_parallel(image_list, threshold=.1, n_jobs = 8):
    """Applies background removal to all images in a list using multiprocessing."""
    result = Parallel(n_jobs=n_jobs)(delayed(detect_object)(im, threshold) for im in image_list)
    return np.array(result)


from scipy.ndimage import uniform_filter

def median_filter(image, kernel_size, stride, threshold):
    """
    Applies a median-based filter to an image to replace bad pixels.

    Parameters:
        image (numpy.ndarray): Input 2D image.
        kernel_size (int): Size of the sliding window (kernel) for computing the median.
        stride (int): Step size for moving the kernel across the image.
        threshold (float): Values outside `median * threshold` are replaced with the median.

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Pad the image to handle edge cases
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Initialize the output image
    filtered_image = np.copy(image)

    # Iterate over the image with the given stride
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            # Define the region of interest (ROI) in the padded image
            roi = padded_image[i:i + kernel_size, j:j + kernel_size]

            # Compute the median of the ROI
            median_value = np.median(roi)

            # Define the bounds for valid pixels
            lower_bound = median_value / threshold
            upper_bound = median_value * threshold

            # Replace outliers in the original image
            roi_original = filtered_image[i:i + kernel_size, j:j + kernel_size]
            outliers = (roi_original < lower_bound) | (roi_original > upper_bound)
            roi_original[outliers] = median_value

    return filtered_image
@time_it
def median_filter_parallel(images, kernel_size, stride, threshold, n_jobs=32):
    """
    Applies the median filter to a list of images in parallel.

    Parameters:
        images (list of numpy.ndarray): List of input 2D images.
        kernel_size (int): Size of the sliding window (kernel) for computing the median.
        stride (int): Step size for moving the kernel across the image.
        threshold (float): Values outside `median * threshold` are replaced with the median.
        n_jobs (int): Number of jobs to run in parallel. Default is -1 (use all available cores).

    Returns:
        list of numpy.ndarray: List of filtered images.
    """
    # Use joblib to parallelize the median filter application
    filtered_images = Parallel(n_jobs=n_jobs)(
        delayed(median_filter)(image, kernel_size, stride, threshold) for image in images
    )
    filtered_images = np.array(filtered_images)

    return filtered_images


def pad_to_double(img):
    """
    Pads each 2D numpy array in a list to double its size.
    The original image will be centered in the padded output.
    
    Args:
        image_list: List of 2D numpy arrays
        
    Returns:
        List of padded 2D numpy arrays, each with double the dimensions
    """
    
    # Get original dimensions
    h, w = img.shape
    
    # Calculate padding for each side
    pad_h = h // 2
    pad_w = w // 2
    
    # Pad the image with zeros
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    return padded_img

@time_it
def pad_to_double_parallel(image_list, n_jobs=8):
    """
    Slices the center of each 2D numpy array in a list,
    keeping only half the size in each dimension.
    
    Args:
        image_list: List of 2D numpy arrays
        
    Returns:
        List of center-sliced 2D numpy arrays, each with half the dimensions
    """
    # Use joblib to parallelize the median filter application
    padded_images = Parallel(n_jobs=n_jobs)(
        delayed(pad_to_double)(image) for image in image_list
    
    )
    return padded_images


def exctract_centres_parallel(image_list, n_jobs=8):
    """
    Slices the center of each 2D numpy array in a list,
    keeping only half the size in each dimension.
    
    Args:
        image_list: List of 2D numpy arrays
        
    Returns:
        List of center-sliced 2D numpy arrays, each with half the dimensions
    """
    # Use joblib to parallelize the median filter application
    sliced_images = Parallel(n_jobs=n_jobs)(
        delayed(median_filter)(image) for image in image_list
    
    )
    return sliced_images
    
def extract_centre(img):
    """
    Slices the center of a 2D numpy array, keeping only half the size in each dimension.
    
    Args:
        img: A 2D numpy array
        
    Returns:
        Center-sliced 2D numpy array with half the dimensions
    """
    # Get original dimensions
    h, w = img.shape
    
    # Calculate start and end indices for slicing
    h_start = h // 4
    h_end = h - h_start
    w_start = w // 4
    w_end = w - w_start
    
    # Slice the center of the image
    sliced_img = img[h_start:h_end, w_start:w_end]
    
    return sliced_img

def shrink_wrap_2d_numpy(input_data, threshold=0.2, sigma=4):
    """
    Applies the shrink-wrap method to create a mask around an object in a 2D NumPy array.
    
    Args:
        input_data (np.ndarray): 2D input array (height, width).
        threshold (float): Threshold for masking (relative to max value).
        sigma (float): Standard deviation for Gaussian blur.
    
    Returns:
        np.ndarray: Binary mask of the same shape as input_data.
    """
    if input_data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")

    # Compute absolute values and normalize
    abs_data = np.abs(input_data)
    max_val = np.max(abs_data)

    # Apply threshold to create initial binary mask
    mask = np.where(abs_data < (threshold * max_val), 0, 1).astype(np.float32)

    # Apply Gaussian smoothing
    mask = gaussian_filter(mask, sigma=sigma)

    # Re-apply threshold to refine the mask
    mask = np.where(mask < threshold, 0, 1).astype(np.uint8)

    return mask

@time_it
def shrink_wrap_parallel(image_list, threshold=0.2, sigma=4, n_jobs=8):
    """
    Slices the center of each 2D numpy array in a list,
    keeping only half the size in each dimension.
    
    Args:
        image_list: List of 2D numpy arrays
        
    Returns:
        List of center-sliced 2D numpy arrays, each with half the dimensions
    """
    # Use joblib to parallelize the median filter application
    mask = Parallel(n_jobs=n_jobs)(
        delayed(shrink_wrap_2d_numpy)(image, threshold, sigma) for image in image_list
    
    )
    return mask


# Function to apply Gaussian blur to a list of objects
def apply_gaussian_blur(arrays, sigma=1):
    """
    Apply Gaussian blur to a list of objects (NumPy arrays).

    Parameters:
    objects (list of numpy arrays): The list of objects to blur.
    sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
    list of numpy arrays: The blurred objects.
    """
    blurred_objects = [gaussian_filter(arr, sigma=sigma) for arr in arrays]
    return np.array(blurred_objects)

def compute_histograms(image_list, bins=256):
    """
    Computes the intensity histogram for each image in a list.

    Args:
        image_list (list of np.ndarray): List of grayscale images (2D NumPy arrays).
        bins (int): Number of bins for the histogram (default: 256 for 8-bit images).

    Returns:
        list of np.ndarray: A list where each element is a histogram array.
    """
    histograms = []
    
    for img in image_list:
        # Compute histogram (normalized)
        maxima = img.max()
        hist, _ = np.histogram(img.ravel(), bins=bins, range=[0, maxima])
        histograms.append(hist)

    return histograms

def threshold_data(image_list, threshold_value):

    """
    threshold the intensity histogram for each image in a list.

    Args:
        image_list (list of np.ndarray): List of grayscale images (2D NumPy arrays).
        threshold_value (float):

    Returns:
        np.ndarray: Each image is filtered 
    """
    filtered_imgs = []
    
    for img in image_list:
        filt_img = np.where(img<threshold_value, 0, img)
        filtered_imgs.append(filt_img)

    return np.array(filtered_imgs)

def bilateral_filter(image, sigma_spatial=3, sigma_range=50, kernel_size=7):
    """
    Applies bilateral filtering to a 2D grayscale image using OpenCV.

    Args:
        image (np.ndarray): 2D grayscale image.
        sigma_spatial (float): Standard deviation for spatial smoothing.
        sigma_range (float): Standard deviation for intensity (range) smoothing.
        kernel_size (int): Size of the filter kernel (determines the neighborhood size).

    Returns:
        np.ndarray: Bilaterally filtered image.
    """

    # Apply OpenCV bilateral filter
    filtered_image = cv2.bilateralFilter(image, d=kernel_size, sigmaColor=sigma_range, sigmaSpace=sigma_spatial)

    return filtered_image
@time_it
def bilateral_filter_parallel(image_list, sigma_spatial=3, sigma_range=50, kernel_size=7, n_jobs=8):

    # Use joblib to parallelize the median filter application
    padded_images = Parallel(n_jobs=n_jobs)(
        delayed(bilateral_filter)(image, sigma_spatial,sigma_range,kernel_size) for image in image_list)
    
    padded_images = np.array(padded_images)
    return padded_images
@time_it
def reorder_pixels_from_center(pixel_coords, connected_array=None):
    """
    Reorders a list of pixel coordinates so that the sequence starts at the center
    and expands outward

    Args:
        pixel_coords: List of (x, y) pixel coordinates.

    Returns:
        list of tuples: Indices of reordered pixel coordinates.
    """
    pixel_coords = np.array(pixel_coords)

    # Compute the centroid (mean of coordinates)
    centroid = np.mean(pixel_coords, axis=0)

    # Find the actual pixel closest to the centroid
    distances = np.linalg.norm(pixel_coords - centroid, axis=1)
    center_idx = np.argmin(distances)  # Index of the closest pixel
    center_pixel = pixel_coords[center_idx]

    # Compute distances from the center pixel
    distances_from_center = np.linalg.norm(pixel_coords - center_pixel, axis=1)

    # Sort pixels by increasing distance
    sorted_indices = np.argsort(distances_from_center)
    # Ensure sorted_indices is a NumPy array and not a tuple
    sorted_indices = np.array(sorted_indices, dtype=int)
    
    return sorted_indices

def flip_images(images, flip_mode):
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
        return images  # No flip
    elif flip_mode == "x_neg_y":
        return np.flip(images, axis=1)  # Flip along y-axis
    elif flip_mode == "neg_x_y":
        return np.flip(images, axis=2)  # Flip along x-axis
    elif flip_mode == "neg_x_neg_y":
        return np.flip(images, axis=(1, 2))  # Flip along both axes
    else:
        raise ValueError(f"Invalid flip_mode: {flip_mode}")