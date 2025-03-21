# Define the objective function

import numpy as np
from scipy.optimize import minimize

def rotation_matrix(alpha, beta, gamma):
    # Construct rotation matrix using standard formulas
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    return np.dot(Rz, np.dot(Ry, Rx)) #Rz @ Ry @ Rx

def objective_function(params, kouts, G_initial, ttheta, wavelength ):
    # Rotate G using Euler angles
    alpha, beta, gamma = params
    R = rotation_matrix(alpha, beta, gamma)
    try:
        G = R @ G_initial
    except:
        G = R @ G_initial[0]
    
    C = G/2
    
    k_mag = 2*np.pi/wavelength 
    R_circle = k_mag * np.cos(ttheta)
    Gmag = 4*np.pi / wavelength * np.sin(ttheta/2)
        
    # Update k_in
    kin = kouts - G

    # Compute deviations for k_out (circle condition)
    f_circle = np.mean(np.abs(np.linalg.norm(kin - C, axis=1) - R_circle) / R_circle)

    # Compute error in the 2theta angle 
    dots = np.sum( (kin/np.linalg.norm(kin, axis=1)[:,np.newaxis] ) * (kouts/np.linalg.norm(kouts, axis=1)[:,np.newaxis]), axis = 1)
    f_angle = np.sum( np.abs( (dots - np.cos(ttheta)) / np.cos(ttheta) ) ) # np.std(dots)**2) #
    
    return f_angle

def optimise_kin(G_init, ttheta, kouts, wavelength, method, gtol):

    # Initial guess for optimization
    initial_guess = [0, 0, 0]

    # Perform optimization
    result = minimize(objective_function, initial_guess, args=(kouts, G_init, ttheta, wavelength),
                  method=method, options={'disp': True, 
                                        "gtol" : gtol, 
                                        #'return_all': True
                                        })

    # Optimized orientation
    optimal_angles = result.x

    opt_mat = rotation_matrix(*optimal_angles)

    try:
        G_opt =  opt_mat @ G_init
    except:
        G_opt =  opt_mat @ G_init[0]
        
    kin_opt = (kouts-G_opt)
    kin_opt /= np.linalg.norm(kin_opt, axis = 1)[:,np.newaxis]
    kin_opt *= 2*np.pi/wavelength

    return kin_opt, optimal_angles

def calc_qvec(kout, kin, **kwargs):
    """
    optional args:
        ttheta: the real two theta value
        wavelength : wavelength of the experiment 
    """

    if len(kout.shape)>1:
        kout = np.mean(kout, axis = 0, keepdims = True)
    
    kout /= np.linalg.norm(kout)
    
    kin_oaxis = kin / np.linalg.norm(kin)

    G_0 = kout - kin_oaxis
    G_0 /= np.linalg.norm(G_0, axis = 1, keepdims = True)
    
    if 'ttheta' and 'wavelength' in kwargs:
        ttheta = kwargs['ttheta']
        wavelength = kwargs['wavelength']
        
        G_0 *= 4*np.pi*np.sin(ttheta/2)/wavelength
    
    return G_0

def reverse_kins_to_pixels(kins, pixel_size, detector_distance, central_pixel):
    """
    Reverse map k_out vectors to detector pixel indices.

    Args:
        kouts (np.ndarray): Array of k_out vectors (N, 3), normalized.
        pixel_size (float): Pixel size in micrometers.
        detector_distance (float): Distance from the crystal to the center of the detector in micro meters.
        detector_shape: Tuple (num_rows, num_cols) of the detector
    Returns:
        np.ndarray: Array of pixel indices for k_out vectors.
    """

    #cen_x,cen_y = np.array(detector_shape)/2

    cen_x, cen_y = np.array(central_pixel)
    
    # pixel size in meters
    pixel_size = np.array(pixel_size)
    kins = kins / np.linalg.norm(kins, axis=1)[:, np.newaxis]
    
    # Reverse mapping to pixel indices
    x_pixels = (kins[:, 0] / kins[:, 2]) * detector_distance / pixel_size + cen_x
    y_pixels = (kins[:, 1] / kins[:, 2]) * detector_distance / pixel_size + cen_y
    
    # Convert to integer pixel indices
    x_pixel_indices = np.floor(x_pixels).astype(int)
    y_pixel_indices = np.floor(y_pixels).astype(int)

    coord = np.vstack((x_pixel_indices, y_pixel_indices)).T
    
def compute_vectors(coordinates, detector_distance, pixel_size, central_pixel, wavelength):
    """
    Compute vectors from the origin to each pixel.

    Parameters:
    - coordinates: (N, 2) array of pixel indices (row, col).
    - detector_distance: Distance from origin to detector center.
    - pixel_size: Pixel size.
    - detector_shape: Tuple (num_rows, num_cols) of the detector.

    Returns:
    - vectors: (N, 3) array of vectors from origin to each pixel.
    """
    #num_x, num_y = detector_shape
    
    #center_x = (num_x - 1) * pixel_size / 2
    #center_y = (num_y - 1) * pixel_size / 2

    center_x, center_y = np.array(central_pixel) * pixel_size
    
    vectors = []
    for coord in coordinates:
        i, j = coord
        x = i * pixel_size - center_x
        y = j * pixel_size - center_y
        
        vector = np.array([x, y, detector_distance])
        vectors.append(vector)

    pixel_vectors = np.array(vectors)
    unit_vectors = pixel_vectors/np.linalg.norm(pixel_vectors, axis = 1)[:,np.newaxis]

    k = 2.0*np.pi /wavelength

    unit_vectors = np.array(unit_vectors)
    
    ks = k*unit_vectors

    return ks