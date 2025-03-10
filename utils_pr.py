import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2


def calc_obj_freq_bandwidth(lr_psize):
    """
    Calculate the Object bandwidth based on the step size in scanning mode.  
    """
    omega_obj_x, omega_obj_y = 2 * np.pi / lr_psize, 2 * np.pi / lr_psize

    return omega_obj_x, omega_obj_y

def prepare_dims(images, kout_vec, lr_psize=25, extend = False):
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

    if extend:
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
