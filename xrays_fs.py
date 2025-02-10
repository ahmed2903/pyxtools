import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from math import sin, cos, pi
import itertools as it
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize

from . import utils as ut
from . import atom_info as af


def energy2wavelength_a(energy_kev: float) -> float:
    """
    Converts energy in keV to wavelength in A

    wavelength_a = energy2wave(energy_kev)

    lambda [A] = h*c/E = 12.3984 / E [keV]

    """
    # Electron Volts:
    E = 1000 * energy_kev * ut.echarge

    # SI: E = hc/lambda
    lam = ut.hplanck * ut.c / E # in meters
    wavelength_a = lam / ut.Ang # in angstroms

    return wavelength_a

def wavelength_a2energy(wavelength):
    """
    Converts wavelength in A to energy in keV
     Energy [keV] = h*c/L = 12.3984 / lambda [A]
    """

    # SI: E = hc/lambda
    lam = wavelength * ut.Ang
    E = ut.hplanck * ut.c / lam

    # Electron Volts:
    Energy = E / ut.echarge
    return Energy / 1000.0

def qmag2dspace(qmag):
    """
    Calculate d-spacing from |Q|
         dspace = q2dspace(Qmag)
    """
    return 2 * np.pi / qmag


def dspace2qmag(dspace):
    """
    Calculate d-spacing from |Q|
         Qmag = q2dspace(dspace)
    """
    return 2 * np.pi / dspace

def wavevector(energy_kev=None, wavelength=None):
    """Return wavevector = 2pi/lambda"""
    if wavelength is None:
        wavelength = energy2wavelength_a(energy_kev)
    return 2 * np.pi / wavelength


def calc_qmag(twotheta:float, **kwargs):
    """
    Calculate |Q| [A^-1] at a particular 2-theta (deg) for energy [keV]

    magQ [A^-1] = calqmag(twotheta, energy_kev=17.794)

    """

    if "energy_kev" in kwargs:
        energy_kev = kwargs["energy_kev"]
        wavelength_a = energy2wavelength_a(energy_kev)

    elif "wavelength_a" in kwargs:
        wavelength_a = kwargs["wavelength_a"]
        
    theta = twotheta * np.pi / 360  # theta in radians

    # Calculate |Q|
    # magq = 4pi sin(theta) / lambda
    magq = np.sin(theta) * 4 * np.pi / wavelength_a

    return magq

def qmag2ttheta(qmag: float, **kwargs) -> float:

    """Calculates the corresponding two theta value for a given Q vector magnitude and a particular X-ray energy. 

    Args:
        qmag: Magnitude of Qvector 
        energy_kev: energy of the xrays [keV]
        wavelength: Xray wavelenght [A]

    Returns:
        ttheta: Two Theta Value [Degrees]
    """
    
    if "energy_kev" in kwargs:
        energy_kev = kwargs["energy_kev"]
        wavelength_a = energy2wavelength_a(energy_kev)  # wavelength form photon energy
        
    elif "wavelength_a" in kwargs:
        wavelength_a = kwargs["wavelength_a"]
        
    else:
        raise ValueError("Supply either a energy (energy_kev) or wavelength (wavelength_a)")
        

    theta = np.arcsin(qmag * wavelength_a/(4*np.pi))

    theta = np.rad2deg(theta)
    ttheta = 2*theta

    return ttheta

def compute_phase_in_batches(q_vec, rj, batch_size=1000):
    """
    Compute phase = exp(-1j * dot(q_vec, rj.T)) in batches to save memory.

    Args:
        q_vec (ndarray): Array of shape (N, 3) containing wavevectors.
        rj (ndarray): Array of shape (M, 3) containing atomic positions.
        batch_size (int): Number of rows in q_vec to process at a time.

    Returns:
        ndarray: Phase matrix of shape (N, M) computed in batches.
    """
    N, M = q_vec.shape[0], rj.shape[0]
    phase = np.zeros((N, M), dtype=np.complex128)  # Allocate output array

    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)  # Handle last batch correctly
        q_batch = q_vec[i:batch_end]  # Extract batch
        phase[i:batch_end] = np.exp(-1j * np.einsum('ij,kj->ik', q_batch, rj))  # Efficient dot product

    return phase

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def compute_phase_batch(q_batch, rj):
    """Computes a batch of phase values."""
    return np.exp(-1j * np.einsum('ij,kj->ik', q_batch, rj))

def compute_phase_multithreaded(q_vec, rj, batch_size=1000, num_threads=4):
    """
    Compute phase = exp(-1j * dot(q_vec, rj.T)) using multithreading for large arrays.

    Args:
        q_vec (ndarray): (N, 3) wavevectors.
        rj (ndarray): (M, 3) atomic positions.
        batch_size (int): Number of rows in q_vec to process per batch.
        num_threads (int): Number of threads to use.

    Returns:
        ndarray: (N, M) phase matrix computed in parallel.
    """
    N = q_vec.shape[0]
    phase = np.zeros((N, rj.shape[0]), dtype=np.complex128)  # Allocate output

    # Define batch ranges
    batch_ranges = [(i, min(i + batch_size, N)) for i in range(0, N, batch_size)]

    # Run multithreading
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda b: compute_phase_batch(q_vec[b[0]:b[1]], rj), batch_ranges))

    # # Merge results into phase array
    # for (start, end), res in zip(batch_ranges, results):
    #     phase[start:end] = res  # Assign computed batches

    phase = np.concatenate(results)
    return phase

def compute_phase_parallel(q_vec, rj, batch_size=1000, n_jobs=4):
    """
    Compute phase = exp(-1j * dot(q_vec, rj.T)) using joblib for parallel processing.

    Args:
        q_vec (ndarray): (N, 3) wavevectors.
        rj (ndarray): (M, 3) atomic positions.
        batch_size (int): Number of rows in q_vec to process per batch.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        ndarray: (N, M) phase matrix computed in parallel.
    """
    N = q_vec.shape[0]

    # Define batch ranges
    batch_ranges = [(i, min(i + batch_size, N)) for i in range(0, N, batch_size)]

    # Run parallel processing using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_phase_batch)(q_vec[start:end], rj) for start, end in batch_ranges
    )

    # Merge results into a single array
    phase = np.concatenate(results)
    return phase

def calculate_atomic_formfactor(atom: str, qvec: np.ndarray, wavelength_a: float):

    """
    Calculates the atomic form factor for a given Q vector and wavelength.

    """
    theta = qmag2ttheta(np.linalg.norm(qvec, axis=-1), wavelength_a= wavelength_a)

    s = np.sin(theta ) / (wavelength_a)

    abc = af.formfactor[atom]
        
    a1 = abc[0]
    a2 = abc[1]
    a3 = abc[2]
    a4 = abc[3]
    b1 = abc[4]
    b2 = abc[5]
    b3 = abc[6]
    b4 = abc[7]
    c = abc[8]

    Z = abc[-1]

    inter = Z - 41.78214 * s**2  * ( (a1*np.exp(-b1*s**2)) + a2*np.exp(-b2*s**2) + a3*np.exp(-b3*s**2) +  a4*np.exp(-b4*s**2) ) +c
    return inter

def calculate_form_factor(real_lattice_vecs, q_vec, R_i, mask_Ri= None):
    
    """
    Input:
        real_lattice_vecs (np.ndarray): 3x3 array containing the real space lattice vectors
        q_vec (np.ndarray): The reciprocal lattice vectors to consider
        R_i (np.ndarray): the real space unit cell positions
        
        Optional:

            mask_Ri (np.ndarray): Mask Ris outside the region of interest (for ptycho scan)
        
    Returns: 
        f_q (np.ndarray): the Form factor of the crystal
    
    Following scattering from atoms in a crystal
    The form factor is Sum_{Unit Cells in Lattice} exp[-i(q.R_i)]
    R_i is the unit cell position
    """
    
    # Volume of unit cell
    V_cell = np.dot(real_lattice_vecs[0], np.cross(real_lattice_vecs[1], real_lattice_vecs[2])) 
    
    if mask_Ri is not None:
        R_i = R_i * mask_Ri[:,np.newaxis]
    
    #phase = np.exp(-1j * np.einsum('ij,kj->ik', q_vec, R_i))
    phase = compute_phase_parallel(q_vec, R_i, batch_size=10000, n_jobs=8)

    #phase = np.exp(-1j*np.dot(q_vec,R_i.T))

    scattering_strength_coeff = (np.sum(mask_Ri) / mask_Ri.shape[0]) # FIX ME!
    
    f_q = 2*pi * np.sum(phase, axis = -1) / V_cell * scattering_strength_coeff
    
    return f_q

from joblib import Parallel, delayed
def calculate_structure_factor(atoms, rj_atoms, q_vec, wavelength_a):
    """
    The structure factor at q_vec, followiung s_q = Sum_{atoms in unit cell} f_j * exp[-1j* q.r_j]

    Input: 
        atoms (list): list of the atom names
        rj_atoms (list or np.ndarray): the locations of the atoms in the unit cell, in the same order as atoms
        q_vec: the q vectors of the experiment
        wavelength_a: the wavelength in Angs
        
    Returns:
        np.ndarray: Structure factor for every q_vec
    """
    rj = np.array(rj_atoms)
    fjs = []
    for atom in atoms:
        fj = calculate_atomic_formfactor(atom, q_vec, wavelength_a)
        fjs.append(fj)
        
    fjs = np.array(fjs)
    phase = compute_phase_parallel(q_vec, rj, batch_size=10000, n_jobs=8)

    #phase = np.exp(-1j * np.einsum('ij,kj->ik', q_vec, rj))
    #phase = np.exp(-1j*np.dot(q_vec,rj.T))
 
    t = fjs.T*phase
    s_q = np.sum(t, axis = -1)
    return s_q

def calculate_scattering_amplitude(real_lattice_vecs, q_vec, R_i, atoms, rj_atoms, wavelength_a, mask_Ri = None):
    
    """
    Calculates the scattering amplitudes F(q) = S(q) [the structure factor] * f(q) [The form factor]
    
    Input: 
        real_lattice_vecs (np.ndarray): 3x3 array containing the real space lattice vectors
        q_vec (np.ndarray): The reciprocal lattice vectors to consider
        R_i (np.ndarray): the real space unit cell positions
        atoms (list): list of the atom names
        rj_atoms (list or np.ndarray): the locations of the atoms in the unit cell, in the same order as atoms
        wavelength_a: the wavelength in Angs

    Returns:
        np.ndarray: the scattering amplitude
    """
    form_factor = calculate_form_factor(real_lattice_vecs, q_vec, R_i, mask_Ri)
    structure_factor = calculate_structure_factor(atoms, rj_atoms, q_vec, wavelength_a)
    scattering_amp = form_factor*structure_factor
    
    return scattering_amp

def calculate_intensity(scat_amp):
    
    return np.abs(scat_amp)**2
    
def convergent_kins(wavelength, NA, focal_length, num_vectors=100):
    """
    Generate an array of incoming k-vectors (incident wave vectors)

    Parameters:
    - NA: Numerical aperture (NA)
    - focal_length: Focal length of the lens (mm)
    - num_vectors: Number of k-vectors to generate (default = 100)

    Returns:
    - Array of incoming k-vectors (shape: [num_vectors, 3])
    """
    # Calculate the maximum scattering angle from the numerical aperture
    theta_max = np.arcsin(NA)

    # Generate random directions within the cone defined by the NA
    phi = np.random.uniform(0, 2 * np.pi, num_vectors)  # Random azimuthal angle (0 to 2*pi)
    
    u = np.random.uniform(0,1, num_vectors)
    theta = np.acos(1-u*(1-np.cos(theta_max)))
    
    #theta = np.random.uniform(0, theta_max, num_vectors)  # Random polar angle (0 to theta_max)

    # Convert spherical coordinates to Cartesian coordinates for the k-vectors
    k_vectors = np.zeros((num_vectors, 3))
    k_vectors[:, 0] = np.sin(theta) * np.cos(phi)  # x component
    k_vectors[:, 1] = np.sin(theta) * np.sin(phi)  # y component
    k_vectors[:, 2] = np.cos(theta)  # z component

    # Normalize to have unit length (magnitude of k-vector should be 2*pi / wavelength)
    k_vectors /= np.linalg.norm(k_vectors, axis = 1, keepdims=True)
    k_magnitude = 2.0*pi / wavelength
    k_vectors *= k_magnitude

    # Converging effect: adjust directions to point towards the focal point
    # For simplicity, let's assume the focal point lies along the z-axis and the beam converges to (0, 0, focal_length)
    # # The beam is converging toward (0, 0, focal_length)
    # focal_point = np.array([0, 0, focal_length])

    # # Normalize each vector to point towards the focal point
    # for i in range(num_vectors):
    #     # Direction vector from the k-vector's point to the focal point
    #     direction_to_focus = focal_point - k_vectors[i]
    #     # Normalize this direction
    #     direction_to_focus /= np.linalg.norm(direction_to_focus)
    #     # Update k-vector direction (the normalized vector)
    #     k_vectors[i] = direction_to_focus * k_magnitude

    return k_vectors

def crystal_to_detector_pixels_vector(detector_distance, pixel_size, detector_size, wavelength):
    
    """
    Assumes the optical axis is along the z-direction
    Left handed coordinate system
    
    Args: 
        detector_distance (float): distance from the centre of the crystal to the detector
        pixel_size (tuple): (x,y) size of the crystal (um)
        detector_size (tuple): (x,y) size of the crystal (m)
        wavelength (float): wavelength in Angstroms
        
    Returns:
        The outgoing wavevectors to every detector pixel
    """
    
    pixel_size = np.array(pixel_size) * 1e-6
    detector_distance = np.array(detector_distance)
    
    # number of pixels on the detector in each dimension
    nx,ny = np.floor(detector_size / pixel_size)

    x = np.arange(nx) - nx/2 + 0.5
    y = np.arange(ny) - ny/2 + 0.5
    
    x*= pixel_size[0]
    y*= pixel_size[1]
    
    xx,yy = np.meshgrid(x,y)
    
    zz = np.full(xx.shape, detector_distance)
    
    pixel_vectors = np.stack([xx, yy, zz], axis=2).reshape(-1,3) # this would be the real space vector
    
    unit_vectors = pixel_vectors/np.linalg.norm(pixel_vectors, axis = 1)[:,np.newaxis]
    
    k = 2.0*pi /wavelength
    
    k_out = k*unit_vectors
        
    return k_out

def gen_qvectors_from_kins_kouts(kins, kouts):
    """
    Given two arrays kins and kouts, this function generates all possible Q vectors 
    along with the indices of their corresponding k_out and k_in.

    Args:
        kins (np.ndarray): The k_in's of the experiment (shape: N, 3)
        kouts (np.ndarray): The k_out's of the experiment (shape: M, 3)

    Returns:
        difference_vectors (np.ndarray): All Q-vectors (shape: N*M, 3)
        k_out_indices (np.ndarray): Indices of k_out corresponding to each Q-vector (shape: N*M,)
        k_in_indices (np.ndarray): Indices of k_in corresponding to each Q-vector (shape: N*M,)
    """

    # Compute the difference vectors for each combination
    difference_vectors = kouts[:, np.newaxis, :] - kins[np.newaxis, :, :]

    # Reshape the result to (N*M, 3)
    difference_vectors = difference_vectors.reshape(-1, 3)

    # Generate indices
    num_kouts, num_kins = len(kouts), len(kins)
    k_out_indices = np.repeat(np.arange(num_kouts), num_kins) # Repeat each k_out index num_kins times
    k_in_indices = np.tile(np.arange(num_kins), num_kouts)
    
    k_out = kouts[k_out_indices]
    k_in = kins[k_in_indices]
    
    return difference_vectors, k_out, k_in

def generate_detector_image(intensities, kouts, detector_size, pixel_size, distance):
    """
    Generate a 2D detector image based on scattering intensities and kouts, accounting for distance R.

    Args:
        intensities (np.ndarray): Array of corresponding scattering intensities (N,).
        indices: indices of the intensities on the detector 
        detector_size (tuple): Detector dimensions in meters (width, height).
        pixel_size (tuple): Pixel size in micrometers (width, height).
    Returns:
        np.ndarray: 2D detector image with intensities.
    """
    indices, intensities = reverse_kouts_to_pixels(kouts, intensities, detector_size, pixel_size, distance)
    
    # Convert pixel size to meters
    pixel_size = np.array(pixel_size) * 1e-6
    
    # Compute number of pixels in each dimension
    nx, ny = (detector_size / pixel_size).astype(int)

    detector_image = np.zeros((nx,ny))


    np.add.at(detector_image, (indices[:,0], indices[:,1]), intensities)
    
    ## Populate the image array with intensities
    #for i in range(len(intensities)):
    #    detector_image[indices[i,0], indices[i,1]] += intensities[i]

    return detector_image

def calc_ttheta_from_kout(kouts, kin_avg):

    """
    Calculates an estimate of the two theta value for a particular streak

    :param kouts: the array of the outgoing wavevectors

    returns: the two theta value in radians
    """
    

    optical_axis = kin_avg[0,:]
    optical_axis /= np.linalg.norm(optical_axis)

    norms = np.linalg.norm(kouts, axis = 1) *  np.linalg.norm(optical_axis)

    dots = np.dot(kouts, optical_axis[:,np.newaxis]) 

    cos_theta = dots / norms[:,np.newaxis]
    
    two_theta = np.arccos(cos_theta)

    return two_theta


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
    
    return coord
    
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
    
    ks = k*unit_vectors

    return ks
    
def reverse_kouts_to_pixels(kouts, intensiies, detector_size, pixel_size, detector_distance):
    """
    Reverse map k_out vectors to detector pixel indices.

    Args:
        kouts (np.ndarray): Array of k_out vectors (N, 3), normalized.
        detector_size (tuple): Detector dimensions in meters (width, height).
        pixel_size (tuple): Pixel size in micrometers (width, height).
        detector_distance (float): Distance from the crystal to the center of the detector in meters.

    Returns:
        np.ndarray: Array of pixel indices for k_out vectors.
    """
    
    # Convert pixel size to meters
    pixel_size = np.array(pixel_size) * 1e-6
    
    # Compute number of pixels in each dimension
    nx, ny = (detector_size / pixel_size).astype(int)

    # Reverse mapping to pixel indices
    x_pixels = (kouts[:, 0] / kouts[:, 2]) * detector_distance / pixel_size[0] + nx / 2
    y_pixels = (kouts[:, 1] / kouts[:, 2]) * detector_distance / pixel_size[1] + ny / 2

    # Convert to integer pixel indices using rounding
    x_pixel_indices = np.floor(x_pixels).astype(int)
    y_pixel_indices = np.floor(y_pixels).astype(int)

    # Verify indices are within bounds
    in_bounds_mask = (x_pixel_indices >= 0) & (x_pixel_indices < nx) & \
                (y_pixel_indices >= 0) & (y_pixel_indices < ny)

    
    x_pixel_indices = x_pixel_indices[in_bounds_mask]
    y_pixel_indices = y_pixel_indices[in_bounds_mask]

    intensiies = intensiies[in_bounds_mask]

    return np.vstack((y_pixel_indices, x_pixel_indices)).T, intensiies

from scipy.spatial.transform import Rotation as R
def generate_spherical_cap(kout, kin, max_angle_deg, num_samples):
    """
    Generate a set of rotated kout vectors forming a spherical cap.
    
    kout: The initial kout vector
    kin: The incident wavevector
    max_angle_deg: Maximum deflection angle from the central kout (defines the spherical cap size)
    num_samples: Number of rotations to generate points on the cap

    Returns:
    - rotated_kouts: An array of kout vectors spanning a spherical cap
    """
    max_angle_rad = np.radians(max_angle_deg)

    # Generate small-angle rotations in θ (polar) and φ (azimuthal)
    theta_vals = np.linspace(0, max_angle_rad, num_samples)  # Polar angles
    phi_vals = np.linspace(0, 2 * np.pi, num_samples)  # Azimuthal angles

    rotated_kouts = []
    
    for theta in theta_vals:
        for phi in phi_vals:
            # Euler rotation matrix for the spherical cap
            euler_angles = [theta, phi, 0]  # (ZXZ convention or other suitable convention)
            rot_matrix = R.from_euler('ZXZ', euler_angles).as_matrix()

            # Rotate kout
            kout_rotated = rot_matrix @ kout.T
            rotated_kouts.append(kout_rotated)

    return np.array(rotated_kouts)

from scipy.interpolate import RegularGridInterpolator
def extract_scattering_amplitude(rotated_kouts, kin, rlvs_high_res, scat_amp_full):
    """
    Interpolates scattering amplitude values from precomputed data.
    """
    q_vectors = rotated_kouts - kin  # Compute q = kout - kin
    sort_idx = np.lexsort((rlvs_high_res[:, 2], rlvs_high_res[:, 1], rlvs_high_res[:, 0]))
    rlvs_high_res = rlvs_high_res[sort_idx]
    # Interpolation function for efficient lookup
    interp_func = RegularGridInterpolator(
        (rlvs_high_res[:, 0], rlvs_high_res[:, 1], rlvs_high_res[:, 2]),
        scat_amp_full,
        method="linear",
        bounds_error=False,
        fill_value=0
    )

    return interp_func(q_vectors)

def compute_image_for_kin(kin, pf, crystal, detector, max_angle_deg, num_samples, rlvs_high_res, scat_amp_full):
    """ Compute detector image for one kin vector using a spherical cap. """
    kout = crystal.gvectors + kin  # Central kout
    rotated_kouts = generate_spherical_cap(kout, kin, max_angle_deg, num_samples)

    # Extract scattering amplitude from precomputed data
    scat_amp = pf * extract_scattering_amplitude(rotated_kouts, kin, rlvs_high_res, scat_amp_full)

    # Generate detector image
    image = generate_detector_image(np.abs(scat_amp), rotated_kouts, detector.size, detector.pixel_size, detector.distance)
    
    return image

# Rotation matrix from Euler angles
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

# Define the objective function
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

def compute_kout_from_G_kin(G_arr, kin_arr):
    """
    A functiont that computes k_out from the reciprocal lattice vectors, and k_in
    
    k_out = k_in + G

    Args:
        G_arr (np.ndarray): The reciprocal lattice vectors
        kin_arr (np.ndarray): The incoming wave vectors
    """
    
    
    k_out = G_arr[:, None, :] + kin_arr[None, :, :]
    
    k_out = k_out.reshape(-1,3)
    
    # Generate the indices for the kin vectors
    kin_indices = np.tile(np.arange(len(kin_arr)), len(G_arr))
    Garr_indices = np.repeat(np.arange(len(G_arr)), len(kin_arr))
    
    return k_out, kin_indices, Garr_indices


def apply_aberattions_to_kins(kins, amplitude_profile=None, phase_aberration=None):
    """
    Generate incoming k-vectors for a convergent beam, including pupil function effects.

    Parameters:
    - kins: the kin vector 
    - amplitude_profile: Function describing the amplitude distribution across the pupil.
                         Takes (kx_norm, ky_norm) as input and returns amplitude.
    - phase_aberration: Function describing the phase aberrations across the pupil.
                        Takes (kx_norm, ky_norm) as input and returns phase (in radians).

    Returns:
    - k_vectors: Array of shape (num_vectors, 3) with each row as a k-vector.
    - weights: Array of shape (num_vectors,) with amplitude and phase weights for each k-vector.
    """
    
    kx = kins[:,0]
    ky = kins[:,1]

    # Initialize weights (amplitude and phase)
    weights = np.ones(kins.shape[0], dtype=complex)

    # Apply amplitude profile if provided
    if amplitude_profile is not None:
        weights *= amplitude_profile(kx, ky)

    # Apply phase aberrations if provided
    if phase_aberration is not None:
        phase = phase_aberration(kx, ky)
        weights *= np.exp(1j * phase)  # Multiply by complex phase factor

    # Combine k-vectors and weights
    return kins, weights

# Phase Aberrations:
def defocus_aberration(kx, ky, defocus_coeff):
    """
    Defocus aberration: quadratic phase error.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - defocus_coeff: Coefficient controlling the strength of defocus.
    
    Returns:
    - Phase error (in radians).
    """
    return defocus_coeff * (kx**2 + ky**2)

def spherical_aberration(kx, ky, spherical_coeff):
    """
    Spherical aberration: quartic phase error.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - spherical_coeff: Coefficient controlling the strength of spherical aberration.
    
    Returns:
    - Phase error (in radians).
    """
    return spherical_coeff * (kx**2 + ky**2)**2

def coma_aberration(kx, ky, coma_coeff):
    """
    Coma aberration: linear in one direction, quadratic in the other.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - coma_coeff: Coefficient controlling the strength of coma.
    
    Returns:
    - Phase error (in radians).
    """
    return coma_coeff * kx * (kx**2 + ky**2)

def astigmatism_aberration(kx, ky, astigmatism_coeff):
    """
    Astigmatism aberration: quadratic phase error with asymmetry.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - astigmatism_coeff: Coefficient controlling the strength of astigmatism.
    
    Returns:
    - Phase error (in radians).
    """
    return astigmatism_coeff * ( kx * ky )

def random_error_profile(kx, ky, amplitude=0.1):
    """
    Random layer placement error.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - amplitude: Amplitude of the random error.
    
    Returns:
    - Random phase error (in radians).
    """
    return amplitude * np.random.normal(size=kx.shape)

def combined_aberrations(kx, ky, coefficients):
    """
    Combine multiple aberrations.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - coefficients: Dictionary of aberration coefficients.
    
    Returns:
    - Total phase error (in radians).
    """
    phase_error = 0.0
    phase_error += defocus_aberration(kx, ky, coefficients['defocus'])
    phase_error += spherical_aberration(kx, ky, coefficients['spherical'])
    phase_error += coma_aberration(kx, ky, coefficients['coma'])
    phase_error += astigmatism_aberration(kx, ky, coefficients['astigmatism'])
    return phase_error


#Amplitude Profiles
def uniform_amplitude(kx, ky):
    """
    Uniform amplitude profile: constant intensity across the lens.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    
    Returns:
    - Amplitude (constant value of 1).
    """
    return np.ones_like(kx)

def gaussian_amplitude(kx, ky, sigma=0.5):
    """
    Gaussian amplitude profile: smooth falloff in intensity.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - sigma: Width of the Gaussian profile.
    
    Returns:
    - Amplitude (Gaussian distribution).
    """
    return np.exp(-(kx**2 + ky**2) / (2 * sigma**2))

def top_hat_amplitude(kx, ky, radius=1.0):
    """
    Top-hat amplitude profile: sharp cutoff at the edges.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - radius: Radius of the lens aperture.
    
    Returns:
    - Amplitude (1 inside the aperture, 0 outside).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.where(r <= radius, 1.0, 0.0)


def apodized_amplitude(kx, ky, sigma=0.5):
    """
    Apodized amplitude profile: smooth tapering at the edges.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - sigma: Width of the apodization profile.
    
    Returns:
    - Amplitude (apodized distribution).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.exp(-(r**2) / (2 * sigma**2)) * (1 - r**2)

def ring_amplitude(kx, ky, radius=0.7, width=0.1):
    """
    Ring-shaped amplitude profile: intensity concentrated in a ring.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - radius: Radius of the ring.
    - width: Width of the ring.
    
    Returns:
    - Amplitude (ring-shaped distribution).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.exp(-((r - radius)**2) / (2 * width**2))

def absorption_amplitude(kx, ky, absorption_coeff=0.1):
    """
    Absorption-based amplitude profile: gradual decrease in intensity due to material absorption.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - absorption_coeff: Absorption coefficient.
    
    Returns:
    - Amplitude (absorption-based distribution).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.exp(-absorption_coeff * r)


def combined_amplitude(kx, ky, profiles):
    """
    Combine multiple amplitude profiles.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - profiles: List of amplitude profile functions.
    
    Returns:
    - Combined amplitude profile.
    """
    amplitude = 1.0
    for profile in profiles:
        amplitude *= profile(kx, ky)
    return amplitude