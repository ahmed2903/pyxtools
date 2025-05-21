# Define the objective function

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

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

def make_coordinates(array, mask_val, roi, crop=False):

    """
    Args:
        array (ndarray): The array from which to calculate the coordinates. 
        mask_val (float): Only pixels above this value will be considered
        roi (list or tuple): the region of interest in pixels (row_start, row_end, column_start, column_end)
    
    Returns:
        coords (ndarray): (N,2) array that is structured as (rows, columns) 
    """
    if crop:
        array = array[roi[0]:roi[1], roi[2]:roi[3]]
        
    indices = np.where(array > mask_val)
    
    coords = np.array([(int(i)+ roi[0], int(j)+roi[2]) for i, j in zip(indices[0], indices[1])])

    return coords

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

    cen_row, cen_col = np.array(central_pixel)
    
    # pixel size in meters
    pixel_size = np.array(pixel_size)
    kins = kins / np.linalg.norm(kins, axis=1)[:, np.newaxis]
    
    # Reverse mapping to pixel indices
    row_pixels = (kins[:, 0] / kins[:, 2]) * detector_distance / pixel_size + cen_row
    col_pixels = (kins[:, 1] / kins[:, 2]) * detector_distance / pixel_size + cen_col
    
    # Convert to integer pixel indices
    row_pixel_indices = np.floor(row_pixels).astype(int)
    col_pixel_indices = np.floor(col_pixels).astype(int)

    coord = np.vstack((row_pixel_indices, col_pixel_indices)).T

    return coord
    
def compute_vectors(coordinates, detector_distance, pixel_size, central_pixel, wavelength):
    """
    Compute vectors from the origin to each pixel.

    Parameters:
    - coordinates: (N, 2) array of pixel indices (row, col).
    - detector_distance: Distance from origin to detector center.
    - pixel_size: Pixel size [units] # FIX ME!
    - central_pixel: Tuple (row, col) of the detector central pixel.

    Returns:
    - vectors: (N, 3) array of vectors from origin to each pixel.
    """
    #num_x, num_y = detector_shape
    
    #center_x = (num_x - 1) * pixel_size / 2
    #center_y = (num_y - 1) * pixel_size / 2

    center_row, center_col = np.array(central_pixel) * pixel_size
    
    vectors = []
    for coord in coordinates:
        i, j = coord
        row = i * pixel_size - center_row
        col = j * pixel_size - center_col
        
        vector = np.array([row, col, detector_distance])
        vectors.append(vector)

    pixel_vectors = np.array(vectors)
    print(pixel_vectors.shape)
    unit_vectors = pixel_vectors/np.linalg.norm(pixel_vectors, axis = 1)[:,np.newaxis]
    
    k = 2.0*np.pi /wavelength

    unit_vectors = np.array(unit_vectors)
    
    ks = k*unit_vectors

    return ks

def extract_streak_region(kins, percentage=10, start_position='random', start_idx=None, seed=42):
    np.random.seed(seed)  # Seed for reproducibility
    kins = np.asarray(kins)

    if kins.ndim != 2 or kins.shape[1] < 2:
        raise ValueError(f"Expected kins to be (N, 2+) but got {kins.shape}")

    kins = kins[:, :2]
    num_pixels = len(kins)
    num_to_select = max(1, int(num_pixels * (percentage / 100)))

    # Determine starting index
    if start_idx is None:
        sorted_indices = np.argsort(kins[:, 0] + kins[:, 1])  # Sort by sum(x, y)

        if start_position == 'lowest':
            start_idx = sorted_indices[0]
        elif start_position == 'highest':
            start_idx = sorted_indices[-1]
        elif start_position == 'middle':
            start_idx = sorted_indices[len(sorted_indices) // 2]
        elif start_position == 'random':
            start_idx = np.random.choice(num_pixels)
        else:
            raise ValueError("start_position must be 'lowest', 'highest', 'middle', or 'random'")

    # Use KDTree for efficient nearest-neighbor search
    tree = KDTree(kins)
    selected_mask = np.zeros(num_pixels, dtype=bool)
    selected_mask[start_idx] = True

    # Initialize BFS queue
    selected_points = [start_idx]
    
    while len(selected_points) < num_to_select:
        new_candidates = []
        for idx in selected_points:
            # Find nearest neighbors (restrict search to small neighborhood)
            _, neighbors = tree.query(kins[idx], k=min(10, num_pixels))  
            for n_idx in neighbors:
                if not selected_mask[n_idx]:  # Only add unselected points
                    selected_mask[n_idx] = True
                    new_candidates.append(n_idx)
                    if len(selected_points) + len(new_candidates) >= num_to_select:
                        break
            if len(selected_points) + len(new_candidates) >= num_to_select:
                break

        selected_points.extend(new_candidates)

    return selected_mask

def extract_parallel_line(kins, width=1, position='center', offset=0):
    """
    Extracts a parallel thin line (1-2 pixels wide) from a streak in the kin coordinates.

    Args:
        kins (np.ndarray): (N, 2) array of kins coordinates.
        width (int): Thickness of the extracted line (1 or 2 pixels).
        position (str): Where to extract the line from. Options: ['center', 'top', 'bottom'].
        offset (int): Custom offset from the center (in pixels).

    Returns:
        np.ndarray: A boolean mask for selected pixels.
    """

    kins = np.asarray(kins)
    if kins.ndim != 2 or kins.shape[1] != 2:
        raise ValueError(f"Expected kins to be (N, 2), but got {kins.shape}")

    # Compute principal component (streak direction)
    pca = PCA(n_components=2)
    pca.fit(kins)
    direction = pca.components_[0]  

    kins_centered = kins - np.mean(kins, axis=0)  
    projected_dist = np.dot(kins_centered, direction)  # Projection onto principal axis

    perpendicular_vec = np.array([-direction[1], direction[0]])  
    perp_dist = np.dot(kins_centered, perpendicular_vec) 

    if position == 'center':
        lower_bound = -width / 2
        upper_bound = width / 2
    elif position == 'top':
        lower_bound = 2  
        upper_bound = 2 + width
    elif position == 'bottom':
        lower_bound = -2 - width
        upper_bound = -2
    else:
        lower_bound = offset
        upper_bound = offset + width

    selected_mask = (perp_dist >= lower_bound) & (perp_dist <= upper_bound)

    return selected_mask