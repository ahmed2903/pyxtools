import numpy as np 

from IPython.display import display, clear_output

import os



from PIL import Image

from skimage.restoration import unwrap_phase
from .utils_zernike import *



def _Zernike_proj(self, wavefront):
    '''
    This function impose Zernike constraint on the given phase wavefront
    
    '''
    shape_y, shape_x = wavefront.shape
    
    wavefront_range = wavefront.max() - wavefront.min()

    if wavefront_range > 2*np.pi:
        wavefront = unwrap_phase(wavefront)

    # Constructing the wavefront using Zernike 

    square_poly = SquarePolynomials() 

    # Create coordinate grids
    side_x = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_x)
    side_y = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_y)

    X, Y = np.meshgrid(side_x, side_y)
    xdata = [X, Y]

    coeffs = extract_square_coefficients_vectorized(wavefront)

    all_results = square_poly.evaluate_all(xdata, coeffs)
    new_wavefront = sum(all_results.values())

    return new_wavefront