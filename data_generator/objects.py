# Date: 28th May, 2025
# Import objects for Fourier Ptychography numerical Experiments

import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
from PIL import Image

from .improcess import *

def create_2dgrid(array_size, pixel_size):
    '''
    
    '''
    N = array_size
    grid_xy = np.arange(-N//2, N//2, pixel_size)
    x, y = np.meshgrid(grid_xy, grid_xy)
    return x, y


def get_vortex(object_size=1024, topo_charge = 1):
    N = object_size
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    m = topo_charge  # Topological charge
    
    amplitude = 1 #np.exp(-20 * R**2)          # Gaussian envelope
    phase =   m * theta          # Vortex phase
    vortex_object = amplitude * np.exp( 1j*phase)       # Complex field
    return vortex_object

def read_complex_images(amp_path, phase_path, target_size):
    '''
    Parameters--
    amp_path : (str) Path of the amplitude pattern
    phase_path : (str) Path of the phase pattern
    target_size : (tuple) shape of the complex image

    Returns---
    f : a complex-valued image with amplitude and phase patterns given in the path
    '''
    
    
    img1 = Image.open(amp_path)
    img2 = Image.open(phase_path)
    
    amp = 0.5 + 0.5*preprocess_image(img1, target_size)
    
    phi = preprocess_image(img2, target_size)
    
    # phi = 0.5 + prep_img(img2)
    # phi = phi/phi.max()
    
    f = amp*np.exp(1j*phi)
    return f

def rect_2d(pixel_size, array_size = 1024, width=1.0, height=1.0, center_x=0.0, center_y=0.0):
    """
    Generate a 2D rectangular function (rect function).
    
    Parameters:
    array_size: computational window over which rect is defined 
    width, height: dimensions of the rectangle
    center_x, center_y: center position of the rectangle
    
    Returns:
    2D array with 1 inside rectangle, 0 outside
    """
    x, y = create_2dgrid(array_size, pixel_size)
    x_condition = np.abs(x - center_x) <= width / 2
    y_condition = np.abs(y - center_y) <= height / 2
    return (x_condition & y_condition).astype(int)

def circ2d( pixel_size, array_size = 1024,radius=10.0, center_x=0.0, center_y=0.0):
    """
    Generate a 2D circular function (circ).
    
    Parameters:
    array_size: computational window over which rect is defined 
    radius: radius of the circular aperture
    center_x, center_y: center position of the rectangle
    
    Returns:
    2D array with 1 inside circle, 0 outside
    """
    x, y =  create_2dgrid(array_size, pixel_size)
    return (np.sqrt((x - center_x)**2 + (y - center_y)**2)< radius)
