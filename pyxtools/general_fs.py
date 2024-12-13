import math 
import numpy as np 
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import time
from scipy.ndimage import convolve

def ComputeAngles(a,b,c,hkl1,hkl2):
    
    """
    A function that computes angles between two lattice planes for a given lattice
    
    Input: 
    
        a, b, c: each is a vector representing the real space lattice vector
        hkl1: first miller indices for the first plane
        hkl2: second miller indices for the second plane

    Returns:
        (float): the angle between the two planes, in degrees. 
    """
    
    h1 , k1, l1 = hkl1
    h2 , k2, l2 = hkl2

    a_vec = a 
    b_vec = b
    c_vec = c

    # Calculate reciprocal lattice vector
    a_star = np.cross(b_vec, c_vec) / (np.dot(a_vec, np.cross(b_vec,c_vec)))
    b_star = np.cross(c_vec, a_vec) / (np.dot(b_vec, np.cross(c_vec,a_vec)))
    c_star = np.cross(a_vec, b_vec) / (np.dot(c_vec, np.cross(a_vec,b_vec)))

    # Calculate direction of the vectors of interest 
    q_1 = h1 * a_star + k1 * b_star + l1 * c_star
    q_2 = h2 * a_star + k2 * b_star + l2 * c_star

    # Calculate angle between them
    angle = math.acos(np.dot(q_1, q_2) / (np.linalg.norm(q_1) * np.linalg.norm(q_2)))
    angle_deg = math.degrees(angle)
    
    return angle_deg

def Bin(ar, binx, biny, binz):
    """
    Bin an array in x,y,z 
    """
    shp = ar.shape
    nx = (shp[0]+binx -1)//binx
    ny = (shp[1]+biny -1)//biny
    nz = (shp[2]+binz -1)//binz
    nshp = np.array((nx, ny, nz),dtype=np.int64)
    
    arraybin = np.zeros((nx, ny, nz), dtype = np.cdouble)
    
    for k in range(binz):
        for j in range(biny):
            for i in range(binx):
                subshp = ar[i::binx,j::biny,k::binz].shape
                arraybin[(nx-subshp[0]):nx,(ny-subshp[1]):ny,(nz-subshp[2]):nz] += ar[i::binx,j::biny,k::binz]
    return arraybin

# Define function that creates a mask for array larger than isosurface
def MakeMask(inarray, isurf):   #
    mask = np.abs(inarray) > isurf # compare amplitude with isosurface 
    return mask

# Logical OR function
def LogicalOR(ar1, ar2):
    ar = np.logical_or(ar1, ar2) 
    return ar

def circular_mean(arr1, arr2):
    # Circular mean difference normalisation
    diff = np.arctan2(np.sin(arr1 - arr2), np.cos(arr1 - arr2))
    return diff

def CalcAngle(vec1, vec2):
    """
    The angle between two vectors
    """

    angle = np.rad2deg(math.acos(np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
    return angle
    
def CalcLen(arr, axis, mask_val):
    """
    CAlculate the length of a 3D pbject in a np array along the given axis. 

    Args:
        arr (np.ndarray): The 3D array containing a uniform 3D object
        axis (np.ndarray): The axis along which to compute the length 
        mask_val (float): The mask value of the array 

    Returns:
        float: the length along the axis. 
    """
    
    direction = axis  
    unit = 30
    
    
    Mx, My, Mz = np.unravel_index(np.argmax(arr), arr.shape)
    
    point_in_plane = np.array([Mx - axis[0]*unit, My - axis[1]*unit, Mz- axis[2]*unit])  
    
    random_vec = np.array([0.0,1.0,0.0])
    random_vec /= np.linalg.norm(random_vec)
    
    plane_vector1 = np.cross(axis, random_vec)  
    plane_vector2 = np.cross(axis, plane_vector1)  

    # Define the number of points in the plane
    num_points_x = unit //2
    num_points_y = unit //2

    # Define the maximum line length along the direction
    max_line_length = unit*2//3  

    max_length = 0

    # Iterate through all start points on the plane
    for i in range(-num_points_x,num_points_x, 2):
        for j in range(-num_points_y, num_points_y, 2):
            start_point = point_in_plane + i * plane_vector1 + j * plane_vector2
            start_point = start_point.astype(int)
            
            # Calculate the end point in the direction of interest
            end_point = start_point + direction * max_line_length
            end_point = end_point.astype(int)

            # Create points along the line from start to end
            points_on_line = np.linspace(start_point, end_point, max_line_length).astype(int)

            # Initialize variables for length calculation
            object_length = 0

            # Iterate through the points on the line
            for point in points_on_line:
                x, y, z = point
                amplitude = arr[x, y, z] # Access the amplitude at the point

                if amplitude > mask_val:
                    object_length += 1

            # Update the maximum length
            max_length = max(max_length, object_length)

    return max_length

def GetBounds(amp, mask_val):
    """
    Calculate the bounds of the diffraction pattern along the x,y,z axes
    """

    shp = amp.shape

    axis_x = [1,0,0]
    axis_y = [0,1,0]
    axis_z = [0,0,1]

    amp_mask = np.where(amp>mask_val, 1, 0)

    min_projection_x = float('inf') 
    max_projection_x = -float('inf')

    min_projection_y = float('inf') 
    max_projection_y = -float('inf')

    min_projection_z = float('inf') 
    max_projection_z = -float('inf')

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
            
                if amp_mask[i,j,k] == 0:
                    pass
                else: 
                    position_vector = np.array([i,j,k])
                                            
                    projection_x = np.dot(axis_x, position_vector)
                    projection_y = np.dot(axis_y, position_vector)
                    projection_z = np.dot(axis_z, position_vector)
                    
                    
                    min_projection_x = min(min_projection_x, projection_x)
                    max_projection_x = max(max_projection_x, projection_x)

                    min_projection_y = min(min_projection_y, projection_y)
                    max_projection_y = max(max_projection_y, projection_y)

                    min_projection_z = min(min_projection_z, projection_z)
                    max_projection_z = max(max_projection_z, projection_z)
            
    
    bounds = [(min_projection_x, max_projection_x),
                (min_projection_y, max_projection_y),
                (min_projection_z, max_projection_z)]
    
    bounds = [max_projection_x - min_projection_x, 
                max_projection_y - min_projection_y, 
                max_projection_z - min_projection_z]
    
    return bounds

def scan_line(arr, start, end):
	"""
	scans the phase in a 1D line in a 3D array giving the start and end points
	"""
	nx, ny, nz = arr.shape
	x = np.linspace(0, nx-1, nx)
	y = np.linspace(0, ny-1, ny)
	z = np.linspace(0, nz-1, nz)
	f = RegularGridInterpolator((x, y, z), arr)

	N = 500  # number of points to sample along the line
	t = np.linspace(0, 1, N)
	line = start + t[:, np.newaxis] * (end - start)
	values = f(line)

	return values

def chi_loss(exp,recons):
    
    chi = np.mean((recons-exp)**2)/(np.mean(exp)+1e-40)
    return chi

def pcc_loss(exp, recons):
    
    vx = np.abs(exp - np.mean(exp))
    vy = np.abs(recons - np.mean(recons))
    
    pcc = np.mean(vx*vy) /(np.sqrt(np.mean(vx**2) + np.mean(vy**2)) +1e-40)
    
    return pcc

def TransposeArray(array):
    
    array2 = array.transpose((2,1,0))

    return array2

def pad(array, psx, psy, psz, pex, pey, pez):
    shp = array.shape

    array = np.concatenate((np.zeros((psx,shp[1],shp[2]), dtype = np.cdouble, order = 'C'), array), axis = 0)
    array = np.concatenate((array, np.zeros((pex,shp[1],shp[2]), dtype = np.cdouble, order = 'C')), axis = 0)

    shp = array.shape

    array = np.concatenate((np.zeros((shp[0],psy,shp[2]), dtype = np.cdouble, order = 'C'), array), axis = 1)
    array = np.concatenate((array, np.zeros((shp[0],pey,shp[2]), dtype = np.cdouble, order = 'C')), axis = 1)

    shp = array.shape

    array = np.concatenate((np.zeros((shp[0],shp[1],psz), dtype = np.cdouble, order = 'C'), array), axis = 2)
    array = np.concatenate((array, np.zeros((shp[0],shp[1],pez), dtype = np.cdouble, order = 'C')), axis = 2)

    return array

def PRTF(diff_array, rec_array, ring_size, fs_spacing, mask_val):
    
    """
    Function to compute the Phase Retrieval Transfer Function: PRTF
    
    Inputs:
        diff_array: The experimental diffraction pattern
        rec_array: Square of Magnitude of Fourier Transform of the recosntructed crystal
        ring_size: Thickness of the bin when computing the PRTF
        fs_spacing: The spacing between voxels
        mask_val: The value of the mask of the experimental diffraction pattern, below which everything is zero

    """
    
    # Shape of the arrays
    shp = diff_array.shape

    # Create a Linear Space that such that the center of the array would be at index 0
    x = np.linspace(-shp[0]//2, shp[0]//2, shp[0])
    y = np.linspace(-shp[1]//2, shp[1]//2, shp[1])
    z = np.linspace(-shp[2]//2, shp[2]//2, shp[2])

    # Create the Mesh Grid to include the linear spaces
    # Each element in the resulting mesh grid arrays xyz represents a point in 3D space. 
    # xyz[0] contains the x-coordinates
    # xyz[1] contains the y-coordinates
    # xyz[2] contains the z-coordinates 
    xyz = np.meshgrid(x,y,z, indexing = 'ij')
    
    # Compute the distance between the center of the array (at 0,0,0) and each point in the space
    magnitude = np.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2)

    # Compute the maximum distance from the center 
    max_dist = np.sqrt((shp[0]/2)**2 + (shp[1]/2)**2 + (shp[2]/2)**2 )
    
    # Create a linearly spaced array of distances
    # This serves as a pointing vector from the center of the array to the distance of interest
    seps = int((max_dist)/ring_size)
    point_vecs = np.linspace(0,max_dist+1, seps)
    
    
    diff_array =np.abs(diff_array)
    rec_array = np.abs(rec_array)
    
    # Apply masking based on a specified threshold value
    diff_array = np.where(diff_array<mask_val, 0.0, diff_array)
    
    # Normalize the arrays by their maximum values
    diff_array /= np.max(diff_array)
    rec_array /= np.max(rec_array)
    
    # Convert to amplitudes by taking the square root
    diff_array = np.sqrt(diff_array)
    rec_array = np.sqrt(rec_array)
    
    # Lists to contain PRTF values and the Number of Voxels in the Ring
    prtf, nv = [],[]
    
    # Iterate through the distances from the centre of the array
    # At each distance, Select only the voxels that = distance +/- ring_size/2
    # For that, a mask is applied when magnitude > distance-ring_size/2 and magnitude < distance+ring_size/2
    # Additionaly, no zero voxels in the experimental diffraction patterns are included
    for vec in point_vecs:
        
        # Half the ring size
        r2 = ring_size /2
        
        # Compute the mask for voxels to include for the current ring
        mask = (magnitude>= (vec-r2)) & (magnitude<= (vec+r2)) & (diff_array>0) & (rec_array>5e-4)
        
        # Apply mask to diffraction and reconstructed arrays
        temp_diff = diff_array[mask]
        temp_rec = rec_array[mask]
        
        # Count the number of voxels within the current mask
        # Will serve as an indication of whether to calculate the PRTF of that ring or not
        # Will also serve as a mask, to not include the rings that have zero voxels
        no_voxels = np.count_nonzero(temp_diff)
        nv.append(no_voxels)
        
        # Calculate the PRTF values if the number of voxels is above zero
        if no_voxels > 0:
            
            prtf_calc = np.divide(temp_rec, temp_diff)
            prtf.append(np.mean(prtf_calc))
            
        else: 
            prtf.append(0)

    # Adjust distances based on diffration pattern spacing
    point_vecs *= fs_spacing * np.sqrt(3)

    # Filter arrays to exclude empty rings
    nv = np.array(nv, dtype = int)

    n_empty_mask = nv != 0 
    
    prtf = np.array(prtf, dtype = np.float32)
    prtf = prtf[n_empty_mask]

    max_freq = prtf.shape[0] * fs_spacing *np.sqrt(3) * ring_size
    
    point_vecs = point_vecs[n_empty_mask]
    
    return prtf, point_vecs, max_freq, nv, n_empty_mask

def create_square_with_uniform_phase(N, square_size, phase_range=(-np.pi, np.pi)):
    """Create a 2D square with a uniform phase distribution."""
    # Initialize the array with zeros
    array = np.zeros((N, N), dtype=complex)
    
    # Calculate the start and end indices for the square
    start_idx = N//2 - square_size//2
    end_idx = start_idx + square_size
    
    # Calculate the phase increment per pixel
    phase_increment = (phase_range[1] - phase_range[0]) / square_size
    
    # Assign amplitude and linearly varying phase to the square region
    for i in range(start_idx, end_idx):
        for j in range(start_idx, end_idx):
            phase = np.pi#phase_range[0] + (j - start_idx) * phase_increment
            array[i, j] = np.exp(1j * phase)  # Amplitude is 1
    
    return array

def compute_fourier_transform(array):
    """Compute the Fourier transform of an array."""
    shp = array.shape
    
    array = pad(array, shp[0]//2, shp[0]//2, shp[1]//2, shp[1]//2, shp[2]//2, shp[2]//2)
    
    ft = np.fft.fftshift(np.fft.fftn(array))
    intensity = np.abs(ft)**2
    return intensity

def gaussian_fit( x,y,z, A, x0, y0, z0, sigma_x, sigma_y, sigma_z):
    
        x_term = ((x - x0) / sigma_x) ** 2
        y_term = ((y - y0) / sigma_y) ** 2
        z_term = ((z - z0) / sigma_z) ** 2
        gauss = A * np.exp((-x_term - y_term - z_term) / 2)
        return gauss 
    
def _gaussian(M, *args):
    
    x,y,z = M
    
    return gaussian_fit(x,y,z,*args)

def FitToGaussian(arr):
    shp = arr.shape
    arr_max = np.max(arr)

    x = np.arange(shp[0])
    y = np.arange(shp[1])
    z = np.arange(shp[2])

    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')

    Mx, My, Mz = np.unravel_index(np.argmax(arr), shp)
    initial_guess = (arr_max, Mx,My,Mz, 1,1,1)

    xdata = np.stack((X.ravel(), Y.ravel(), Z.ravel()))

    popt, _ = curve_fit(_gaussian, xdata, arr.ravel(), initial_guess)

    return popt

def find_aligned_indices(arr1, arr2, tolerance=1e-5):
    
    """
    Finds the aligned indices between two arrays
    """
    
    arr1 = arr1 / np.linalg.norm(arr1, axis=1)[:,np.newaxis]
    arr2 = arr2 / np.linalg.norm(arr2, axis=1)[:,np.newaxis]
    
    dot_products = np.dot(arr2, arr1.T)
    
    aligned_indice= np.where(np.abs(dot_products-1)< tolerance)
    
    return aligned_indice[0]

def filter_vectors_by_direction(arr, kins, direction, max_angle):
    """
    Takes an array of vector (N,3) and filters them such that they are all pointing in one direction
    also takes the corresponding array
    
    Adds anoter constraint of maximum angular spread of the vectors 
    such that the vectors are pointing in a cone
    
    Args:
    arr (np.ndarray): array of vectors N,3
    arr (np.ndarray): corresponding array of vectors N,3

    direction (np.ndarray): principle direction
    max_angle (float): [radians] angle of cone with the direction vector  
    """
    
    direction /= np.linalg.norm(direction)
    
    cos_max_angle = math.cos(max_angle)
        
    cos_vectors = np.dot(arr, direction)  / np.linalg.norm(arr, axis = 1)
    
    arr = arr[cos_vectors>= cos_max_angle]
    kins = kins[cos_vectors>= cos_max_angle]
    
    return arr, kins

from joblib import Parallel, delayed

def compute_norms_chunk(chunk):
    return np.linalg.norm(chunk, axis=1)


def filter_elastic_scatt(kouts, kins, tolerance, wavelength):
    """
    
    Filters the kouts and kins to consider only when:
    |kout| = |kin| +/- 1e-4

    Args:
        kouts (_type_): _description_
        kins (_type_): _description_
    """
    time1 = time.time()
    # Compute the magnitudes of kin and kout
    # kin_magnitudes = np.linalg.norm(kins, axis=1)  
    # kout_magnitudes = np.linalg.norm(kouts, axis=1)  
    
    # Parallelize norm computation
    kout_magnitudes = np.concatenate(Parallel(n_jobs=-1)(delayed(compute_norms_chunk)(chunk) for chunk in np.array_split(kouts, 4)))

    time2 = time.time()
    
    print(f"calculating magnitudes took {time2-time1:.6f} seconds ")
    # Create a mask for filtering based on the magnitude condition
    
    magnitude = 2*math.pi / wavelength
    
    
    magnitude_diff = np.abs(magnitude - kout_magnitudes)
    mask = magnitude_diff < tolerance
    
    print("mask shape is ")
    print(mask.shape)
    
    time5 = time.time()
    # Apply the mask to get the filtered kin and kout pairs
    filtered_kin = kins[mask]
    filtered_kout = kouts[mask]
    time6 = time.time()
    print(f"filtering took {time6 - time5 :.6f} seconds")
    
    return filtered_kout, filtered_kin
    
def calc_detector_max_angle(detector_size, detector_distance):
    
    """
    Calculates the maximum angle that a detector can capture.
    
    Detector is assumed to be centralised, such that centre of the detector is along the optical axis
    
    Returns: max angle [radians]
    """
    
    sx, sy = detector_size[0]/2, detector_size[1]/2
    
    det_max_path = np.array([sx,sy,detector_distance])
    
    det_max_path /= np.linalg.norm(det_max_path)
    
    max_angle = math.acos(np.dot(det_max_path, np.array([0,0,1])))
    
    return max_angle

import numpy as np

def convolve_reciprocal_lattice_with_grid(shape_transform, reciprocal_vectors, kx,ky,kz):
    """
    Convolve reciprocal lattice points with a 3D shape transform, including grid generation.

    Parameters:
        shape_transform (numpy.ndarray): A 3D array representing the shape transform (size: grid_size).
        reciprocal_vectors (numpy.ndarray): An (N, 3) array of reciprocal lattice vectors.

    Returns:
        output_points (numpy.ndarray): An (M, 3) array of convolved reciprocal lattice points (M = N * K).
        combined_intensities (numpy.ndarray): An (M,) array of intensities for each convolved point.
    """    
    
    # Flatten the 3D shape transform into a list of points (K, 3)
    X, Y, Z = np.meshgrid(kx, ky, kz, indexing="ij")
    shape_coords = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T  # (K, 3)
    shape_values = shape_transform.ravel()  # (K,)

    # Number of reciprocal lattice points (N) and shape points (K)
    N = len(reciprocal_vectors)
    K = len(shape_coords)

    # Compute the convolved points (M = N * K)
    output_points = (reciprocal_vectors[:, None, :] + shape_coords[None, :, :]).reshape(-1, 3)  # (M, 3)

    # Compute combined intensities for each point
    reciprocal_intensities = np.ones(N)  # Default to uniform intensity; modify if needed
    combined_intensities = (reciprocal_intensities[:, None] * shape_values[None, :]).ravel()  # (M,)

    return output_points, combined_intensities
