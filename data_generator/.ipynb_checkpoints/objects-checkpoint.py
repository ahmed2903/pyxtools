# Date: 28th May, 2025
# Import objects for Fourier Ptychography numerical Experiments

import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
from PIL import Image

from .improcess import prep_img


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

def read_complex_images(amp_path, phase_path):
    img1 = Image.open(amp_path)
    img2 = Image.open(phase_path)
    
    amp = 0.5 + 0.5*prep_img(img1)
    
    phi = prep_img(img2)
    
    # phi = 0.5 + prep_img(img2)
    # phi = phi/phi.max()
    
    f = amp*np.exp(1j*phi)
    return f