import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import torch
from scipy.ndimage import zoom 
from joblib import Parallel, delayed
import time
from time import strftime
import functools
import multiprocessing as mp
from tqdm import tqdm 

def time_it(func):
    """Decorator to measure execution time of a function."""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Compute execution time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result  # Return original function result
    return wrapper


def run_reconstruction(args):
    
    run_id, recon_class, iterations, fname, zoom_factor, kwargs = args
    
    recon = recon_class(**kwargs)
    
    recon.prepare(extend = 'double', zoom_factor=zoom_factor)
    recon.iterate(iterations)
    recon.save_reconsturction(f"{fname}_id{run_id}.h5")

def reconstruction_repeats(recon_class, iterations, fname, repeats = 5,  n_jobs=4, zoom_factor=1, **kwargs):
    time_str = strftime("%Y-%m-%d_%H.%M")
    fname += f"_{time_str}"
    
    args_list = [(i, recon_class, iterations, fname, zoom_factor, kwargs) for i in range(repeats)]
 
    with mp.Pool(processes=n_jobs) as pool:
        list(tqdm(pool.imap_unordered(run_reconstruction, args_list), total=repeats, desc="Reconstructions"))
    

    
def calc_obj_freq_bandwidth(lr_psize):
    """
    Calculate the Object bandwidth based on the step size in scanning mode.  
    """
    omega_obj_x, omega_obj_y = 2 * np.pi / lr_psize, 2 * np.pi / lr_psize

    return omega_obj_x, omega_obj_y

def prepare_dims(images, pupil_kins, lr_psize, extend = None, band_multiplier = 1):
    """
    Prepare the dimensions of the high resolution fourier space image. 

    High resolution pixel size in fourier space: 2pi/Lx, 2pi/Ly. 
                Where Lx, Ly is the scan length in x and y. # CHECK ME 
    """
    
    # Convert inputs to numpy arrays
    pupil_kins = np.array(pupil_kins)

    # Low-resolution image shape
    coh_img_dim = images[0].shape
    nx_lr, ny_lr = coh_img_dim

    # Object size == Scan length
    Lx, Ly = coh_img_dim[0] * lr_psize, coh_img_dim[1] * lr_psize

    # High-resolution Fourier pixel size 
    dkx, dky = 2 * np.pi / Lx, 2 * np.pi / Ly

    # Define the extents of the wave vectors
    kx, ky = pupil_kins[:, 0], pupil_kins[:, 1]
    kx_min, kx_max = np.min(kx), np.max(kx)
    ky_min, ky_max = np.min(ky), np.max(ky)
    
    if extend == 'double' :
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min
        
        kx_min = kx_min - range_x/2
        kx_max = kx_max + range_x/2
        
        ky_min = ky_min - range_y/2
        ky_max = ky_max + range_y/2

    # FIX ME!!!
    elif extend == 'by_bandwidth':
        # Object bandwidth 
        omega_obj_x, omega_obj_y = calc_obj_freq_bandwidth(lr_psize)
        # Extend the range of kx and ky to fit boundary values
        kx_min = kx_min - omega_obj_x * band_multiplier
        kx_max = kx_max + omega_obj_x * band_multiplier
        
        ky_min = ky_min - omega_obj_y * band_multiplier
        ky_max = ky_max + omega_obj_y * band_multiplier

    return (kx_min,kx_max), (ky_min,ky_max), (dkx,dky)

def init_hr_image(bounds_x, bounds_y, dks):

    kx_min_n, kx_max_n = bounds_x
    ky_min_n, ky_max_n = bounds_y
    
    dkx, dky = dks
    
    kx_sp = np.arange(kx_min_n, kx_max_n, dkx)
    ky_sp = np.arange(ky_min_n, ky_max_n, dky)

    nkx, nky = kx_sp.shape[0], ky_sp.shape[0]
    
    # Round up to the nearest even number
    nkx = int(np.ceil(nkx / 2) * 2)  
    nky = int(np.ceil(nky / 2) * 2) 
        
    # High-resolution Object and Fourier spaces
    hr_obj_image = np.ones((nkx, nky), dtype=complex)
    hr_fourier_image =  np.ones((nkx, nky), dtype=complex) #fftshift(fft2(ifftshift(hr_obj_image)))

    return hr_obj_image, hr_fourier_image


def mask_torch_ctf(outer_size, device=torch.device('cpu')):
    """
    Create a (2N, 2M) array with ones in the center region of size (N, M) and zeros elsewhere.
    
    Parameters:
        outer_size: tuple (2N, 2M) -> total size of the array
        inner_size: tuple (N, M) -> size of the central region filled with ones
    
    Returns:
        mask: (2N, 2M) torch.Tensor
    """
    mask = torch.zeros(outer_size, dtype=torch.float64, device=device)
    
    # Calculate center indices
    N, M = outer_size[0]//2, outer_size[1]//2
    
    start_x, start_y = (outer_size[0] - N) // 2, (outer_size[1] - M) // 2
    end_x, end_y = start_x + N, start_y + M

    # Set central region to ones
    mask[start_x:end_x, start_y:end_y] = 1

    return mask


def make_dims_even(size_tuple):
    """
    Takes a tuple of (x, y) and ensures both values are even.
    If a value is odd, it's increased by 1 to make it even.
    """
    x, y = size_tuple
    
    if x % 2 != 0:
        x += 1
        
    if y % 2 != 0:
        y += 1
        
    return (x, y)

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
    
@time_it
def pad_array_flexible(arr, target_shape, mode='constant', constant_values=0, center=True):
    """
    More flexible padding function using numpy.pad.
    
    Parameters:
    -----------
    arr : array_like
        Input array
    target_shape : tuple
        Target shape for the padded array
    mode : str or function, optional
        Padding mode (see numpy.pad documentation)
    constant_values : scalar, optional
        Value to use for constant padding
    center : bool, optional
        If True, center the array; if False, pad at the end
        
    Returns:
    --------
    np.ndarray
        Padded array
    """
    arr = np.asarray(arr)
    
    if len(arr.shape) != len(target_shape):
        raise ValueError(f"Dimension mismatch: array has {len(arr.shape)}D, target is {len(target_shape)}D")
    
    pad_widths = []
    for current, target in zip(arr.shape, target_shape):
        if target < current:
            raise ValueError(f"Target size {target} is smaller than current size {current}")
        
        pad_total = target - current
        if center:
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
        else:
            pad_before = 0
            pad_after = pad_total
            
        pad_widths.append((pad_before, pad_after))
    
    return np.pad(arr, pad_widths, mode=mode, constant_values=constant_values)