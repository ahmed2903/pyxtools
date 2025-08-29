import numpy as np 
import imageio
from PIL import Image


# Some utils functions
def create_2dgrid(array_size, pixel_size):
    '''
    
    '''
    Nr = array_size[0]
    Nc = array_size[1]

    grid_x = np.arange(-Nr//2, Nr//2, pixel_size[0])
    grid_y = np.arange(-Nc//2, Nc//2, pixel_size[1])
    x, y = np.meshgrid(grid_x, grid_y)
    return x, y

def prep_img(img, target_size):
    '''
    Parameters:
    img : image that needs to be prepared
    target size : (tuple) target array size eg. (n, m)
    
    Returns: 
    a 2D image array which is
    - resized to the target size
    - normalized
    '''
    img =  img.convert("L")
    # Resize the image
    img = img.resize(target_size)
    img = np.array(img)
    img = img/img.max()
    return img

def binary_grating(array_size, pixel_size, period=1.0, orientation='horizontal', phase=0.0):
    """
    Generate a 2D binary grating pattern.
    
    Parameters:
    array_size: computational window over which grating is defined ( considering a square dim )
    period: spatial period of the grating
    orientation: 'horizontal', 'vertical', or angle in radians
    phase: phase shift of the grating
    
    Returns:
    2D array with values of 0 and 1
    """
    x, y = create_2dgrid(array_size, pixel_size)

    
    if orientation == 'horizontal':
        pattern = np.sin(2 * np.pi * y / period + phase)
    elif orientation == 'vertical':
        pattern = np.sin(2 * np.pi * x / period + phase)
    else:  # orientation is an angle in radians
        rotated_coord = x * np.cos(orientation) + y * np.sin(orientation)
        pattern = np.sin(2 * np.pi * rotated_coord / period + phase)
    
    return (pattern > 0).astype(int)

def rect_2d(array_size, pixel_size, width=1.0, height=1.0, center_x=0.0, center_y=0.0):
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

def circ_2d(array_size, pixel_size, radius=1.0, center_x=0.0, center_y=0.0):
    """
    Generate a 2D circle function
    
    Parameters:
    array_size: computational window over which circle is defined 
    pixel_size: size of each pixel in the grid
    radius: radius of circular aperture
    center_x, center_y: center position of the circle
    
    Returns:
    2D array with 1 inside circular aperture, 0 outside
    """
    x, y = create_2dgrid(array_size, pixel_size)
    circ_aperture = np.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius
    return circ_aperture.astype(int) 

def quadratic_2d(array_size, pixel_size, a=1.0, b=1.0, c=0.0, center_x=0.0, center_y=0.0):
    """
    Generate a 2D quadratic function.
    
    Parameters:
    array_size: computational window size 
    a, b: quadratic coefficients for x and y directions
    c: constant offset
    center_x, center_y: center of the quadratic
    
    Returns:
    2D array with quadratic values
    """
    x, y = create_2dgrid(array_size, pixel_size)
    x_shifted = x - center_x
    y_shifted = y - center_y
    return a * x_shifted**2 + b * y_shifted**2 + c

def exponential_chirp_1d_extended(arraysize, pixel_size, f0=1.0, f1=10.0, t1=1.0, direction='horizontal', amplitude=1.0):
    """
    Generate a 1D exponential chirp extended to 2D (constant along one axis).
    
    In an exponential chirp, the frequency changes exponentially from f0 to f1 
    over the distance from 0 to t1.
    
    Parameters:
    array_size: computational window size 
    f0: initial frequency at position 0
    f1: final frequency at position t1
    t1: position where frequency reaches f1
    direction: 'horizontal' (chirp along x) or 'vertical' (chirp along y)
    amplitude: amplitude of the chirp
    
    Returns:
    2D array with exponential chirp pattern
    """
    x, y = create_2dgrid(array_size, pixel_size)

    if t1 == 0 or f0 <= 0 or f1 <= 0:
        raise ValueError("t1 must be non-zero, f0 and f1 must be positive")
    
    # Calculate the exponential growth rate
    beta = np.log(f1 / f0) / t1
    
    if direction == 'horizontal':
        coord = x
    else:  # vertical
        coord = y
    
    # For exponential chirp: f(t) = f0 * exp(beta * t)
    # Phase: integral of 2π * f(t) dt = 2π * f0 * (exp(beta * t) - 1) / beta
    # Handle case where beta is very small (approximately linear)
    if abs(beta) < 1e-10:
        phase = 2 * np.pi * f0 * coord
    else:
        phase = 2 * np.pi * f0 * (np.exp(beta * coord) - 1) / beta
    
    return amplitude * np.cos(phase)

def quadratic_2d(array_size, pixel_size, a=1.0, b=1.0, c=0.0, center_x=0.0, center_y=0.0):
    """
    Generate a 2D quadratic function.
    
    Parameters:
    array_size: computational window size 
    a, b: quadratic coefficients for x and y directions
    c: constant offset
    center_x, center_y: center of the quadratic
    
    Returns:
    2D array with quadratic values
    """
    x, y = create_2dgrid(array_size, pixel_size)
    x_shifted = x - center_x
    y_shifted = y - center_y
    return a * x_shifted**2 + b * y_shifted**2 + c

def get_multiple_vortices(array_size, pixel_size, vortices=[((-256, -256), 1), ((256, 256), -1), ((0.0, 0.0), 2)]):
   
    X, Y = create_2dgrid(array_size, pixel_size)
    
    total_vortex_field = np.ones(array_size, dtype=complex)  # Start with uniform field

    for (x0, y0), m in vortices:
        X_shifted = X - x0
        Y_shifted = Y - y0
        theta = np.arctan2(Y_shifted, X_shifted)
        phase = m * theta
        vortex_field = np.exp(1j * phase)
        total_vortex_field *= vortex_field  # Multiply fields for coherent combination

    return np.angle(total_vortex_field)

    