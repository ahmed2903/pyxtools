import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift

from joblib import Parallel, delayed
import multiprocessing as mp
import time

def _fft2( arr):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))

def _ifft2( arrFT):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arrFT)))

def register_images(images, image_ref):
    '''
    image registration with no parallel processing
    '''

    N, _, _ = images.shape

    im_tot = image_ref.copy()

    for k in range(N):
        this_img = images[k]

        shift_val, error, diffphase = phase_cross_correlation(image_ref, this_img, upsample_factor=100)

        this_img_reset = shift( this_img, shift = shift_val )

        if this_img_reset.any() is np.nan:
            print(k)
            raise ValueError

        mask = this_img_reset != 0
        
        combined_image = np.where(mask, np.abs(this_img_reset), image_ref)
        
        im_tot += np.abs(combined_image)

    image_registered = im_tot /(N+1)

    return image_registered


def process_single_image(this_img, image_ref):
    """
    Process a single image: calculate shift and apply it
    """
    
    shift_val, error, diffphase = phase_cross_correlation(
        image_ref, this_img, upsample_factor=10
    )
    this_img_reset = shift(this_img, shift=shift_val)
    
    if np.any(np.isnan(this_img_reset)):
        raise ValueError(f"NaN detected in shifted image")
    
    mask = this_img_reset != 0
    combined_image = np.where(mask, np.abs(this_img_reset), image_ref)
    
    return np.abs(combined_image)


def register_images_parallel(images, image_ref, n_jobs=-1):
    """
    Parallel version of register_images
    
    Parameters:
    -----------
    images : array-like, shape (N, H, W)
        Stack of images to register
    image_ref : array-like, shape (H, W)
        Reference image
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available cores
    
    Returns:
    --------
    image_registered : array-like, shape (H, W)
        Averaged registered image
    """
    N, _, _ = images.shape
    start_time = time.time()
    
    # Process all images in parallel
    print(f"Processing {N} images using {n_jobs if n_jobs > 0 else mp.cpu_count()} cores...")

    
    parallel_start = time.time()
    combined_images = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_image)(images[k], image_ref) 
        for k in range(N)
    )

    parallel_time = time.time() - parallel_start
    
    # Sum all combined images and add reference
    sum_start = time.time()

    im_tot = image_ref.copy()
    for combined_img in combined_images:
        im_tot += combined_img
    
    # Calculate average
    image_registered = im_tot / (N + 1)

    sum_time = time.time() - sum_start
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Timing Summary:")
    print(f"  Parallel processing: {parallel_time:.2f} seconds")
    print(f"  Summing results:     {sum_time:.2f} seconds")
    print(f"  Total time:          {total_time:.2f} seconds")
    print(f"  Time per image:      {total_time/N:.3f} seconds")
    print(f"  Images per second:   {N/total_time:.2f}")
    print(f"{'='*60}\n")
    
    return image_registered


def process_single_upsample(image, target_shape, Nr1, Nc1):
    """Process a single image upsampling"""
    Nx, Ny = target_shape
    
    image_FT = _fft2(image)
    
    new_image_FT = np.zeros((Nx, Ny), dtype=complex)
    
    new_image_FT[Nx//2 - Nr1//2 : Nx//2 + Nr1//2, 
                 Ny//2 - Nc1//2 : Ny//2 + Nc1//2] = image_FT
    
    upsampled = np.abs(_ifft2(new_image_FT)) / (Nr1**2 / Nx**2)
    
    return upsampled


def upsample_coherent_images_parallel(images, target_shape, n_jobs=-1):
    """
    Parallel version of _upsample_coherent_images
    
    Parameters:
    -----------
    images : array-like, shape (N, Nr1, Nc1)
        Stack of coherent images to upsample
    target_shape : tuple (Nx, Ny)
        Target shape for upsampling
    _fft2 : function
        FFT2 function to use
    _ifft2 : function
        IFFT2 function to use
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available cores
    
    Returns:
    --------
    upsamp_img : array-like, shape (N, Nx, Ny)
        Upsampled images
    """
    start_time = time.time()
    
    Ncoherent_imgs, Nr1, Nc1 = images.shape
    Nx, Ny = target_shape
    n_cores = n_jobs if n_jobs > 0 else mp.cpu_count()
    
    print(f"Upsampling {Ncoherent_imgs} images from {Nr1}x{Nc1} to {Nx}x{Ny} using {n_cores} cores...")
    
    # Process all images in parallel
    parallel_start = time.time()
    upsampled_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_upsample)(images[k], target_shape, Nr1, Nc1)
        for k in range(Ncoherent_imgs)
    )
    parallel_time = time.time() - parallel_start
    
    # Stack results into array
    stack_start = time.time()
    upsamp_img = np.stack(upsampled_list, axis=0)
    stack_time = time.time() - stack_start
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Timing Summary:")
    print(f"  Parallel processing: {parallel_time:.2f} seconds")
    print(f"  Stacking results:    {stack_time:.2f} seconds")
    print(f"  Total time:          {total_time:.2f} seconds")
    print(f"  Time per image:      {total_time/Ncoherent_imgs:.3f} seconds")
    print(f"  Images per second:   {Ncoherent_imgs/total_time:.2f}")
    print(f"{'='*60}\n")
    
    return upsamp_img
