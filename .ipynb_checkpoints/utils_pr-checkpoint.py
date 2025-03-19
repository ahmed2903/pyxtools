import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import torch


def calc_obj_freq_bandwidth(lr_psize):
    """
    Calculate the Object bandwidth based on the step size in scanning mode.  
    """
    omega_obj_x, omega_obj_y = 2 * np.pi / lr_psize, 2 * np.pi / lr_psize

    return omega_obj_x, omega_obj_y

def prepare_dims(images, kout_vec, lr_psize, extend_to_double):
    """
    Prepare the dimensions of the high resolution fourier space image. 

    High resolution pixel size in fourier space: 2pi/Lx, 2pi/Ly. 
                Where Lx, Ly is the scan length in x and y. # CHECK ME 
    """
    
    # Convert inputs to numpy arrays
    kout_vec = np.array(kout_vec)

    # Number of coherent images
    n_imgs = images.shape[0]
    
    # Low-resolution image shape
    coh_img_dim = images[0].shape
    nx_lr, ny_lr = coh_img_dim

    # Object size == Scan length
    Lx, Ly = coh_img_dim[0] * lr_psize, coh_img_dim[1] * lr_psize

    

    # High-resolution Fourier pixel size 
    dkx, dky = 2 * np.pi / Lx, 2 * np.pi / Ly

    # Define the extents of the wave vectors
    kx, ky = kout_vec[:, 0], kout_vec[:, 1]
    kx_min, kx_max = np.min(kx), np.max(kx)
    ky_min, ky_max = np.min(ky), np.max(ky)

    
    
    if extend_to_double:
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min
        
        kx_min = kx_min - range_x/2
        kx_max = kx_max + range_x/2
        
        ky_min = ky_min - range_y/2
        ky_max = ky_max + range_y/2
        
    else:
        # Object bandwidth 
        omega_obj_x, omega_obj_y = calc_obj_freq_bandwidth(lr_psize)
        # Extend the range of kx and ky to fit boundary values
        kx_min = kx_min - omega_obj_x
        kx_max = kx_max + omega_obj_x
        
        ky_min = ky_min - omega_obj_y
        ky_max = ky_max + omega_obj_y

    
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
    hr_fourier_image = fftshift(fft2(ifftshift(hr_obj_image)))

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
