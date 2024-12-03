import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from math import sin, cos, pi
import itertools as it
from multiprocessing import Pool, cpu_count

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

def calculate_form_factor(real_lattice_vecs, q_vec, R_i):
    
    """
    Input:
        real_lattice_vecs (np.ndarray): 3x3 array containing the real space lattice vectors
        q_vec (np.ndarray): The reciprocal lattice vectors to consider
        R_i (np.ndarray): the real space unit cell positions
        
    Returns: 
        f_q (np.ndarray): the Form factor of the crystal
    
    Following scattering from atoms in a crystal
    The form factor is Sum_{Unit Cells in Lattice} exp[-i(q.R_i)]
    R_i is the unit cell position
    """
    
    # Volume of unit cell
    V_cell = np.dot(real_lattice_vecs[0], np.cross(real_lattice_vecs[1], real_lattice_vecs[2])) 
    print(q_vec.shape)
    print(R_i.shape)
    f_q = 2*pi * np.sum(np.exp(-1j*np.dot(q_vec,R_i.T)), axis = -1) / V_cell 
    
    return f_q

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
        
    print("atomic form factor done")
    
    fjs = np.array(fjs)
    
    print(fjs.shape)
    print(rj.shape)
    print("computing phase")
    phase = np.exp(-1j*np.dot(q_vec,rj.T))
    print("summing structure factor")
    
    s_q = np.sum(fjs*phase, axis = 1)
    
    return s_q


def calculate_scattering_amplitude(real_lattice_vecs, q_vec, R_i, atoms, rj_atoms, wavelength_a):
    
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
    print("computing form factor")
    form_factor = calculate_form_factor(real_lattice_vecs, q_vec, R_i)
    print("form factor done")
    print("computing structure factor")
    structure_factor = calculate_structure_factor(atoms, rj_atoms, q_vec, wavelength_a)
    print("structure_factor: done")
    scattering_amp = form_factor*structure_factor
    
    return structure_factor

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
    theta = np.random.uniform(0, theta_max, num_vectors)  # Random polar angle (0 to theta_max)

    # Convert spherical coordinates to Cartesian coordinates for the k-vectors
    k_vectors = np.zeros((num_vectors, 3))
    k_vectors[:, 0] = np.sin(theta) * np.cos(phi)  # x component
    k_vectors[:, 1] = np.sin(theta) * np.sin(phi)  # y component
    k_vectors[:, 2] = np.cos(theta)  # z component

    # Normalize to have unit length (magnitude of k-vector should be 2*pi / wavelength)
    k_magnitude = 2.0*pi / wavelength
    k_vectors *= k_magnitude

    # Converging effect: adjust directions to point towards the focal point
    # For simplicity, let's assume the focal point lies along the z-axis and the beam converges to (0, 0, focal_length)
    # The beam is converging toward (0, 0, focal_length)
    focal_point = np.array([0, 0, focal_length])

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
    
    print(f" ratio: {detector_distance/unit_vectors[:,2]}")
    
    return k_out

def gen_Qvectors_from_kins_kouts(kins, kouts):
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
    k_in_indices = np.repeat(np.arange(num_kins), num_kouts)
    
    k_out = kouts[k_out_indices]
    k_in = kins[k_in_indices]
    print(f"k_out shape is: {k_out.shape}")
    
    return difference_vectors, k_out, k_in


def generate_detector_image(intensities, indices, detector_size, pixel_size):
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
    
    # Convert pixel size to meters
    pixel_size = np.array(pixel_size) * 1e-6
    
    # Compute number of pixels in each dimension
    nx, ny = (detector_size / pixel_size).astype(int)
    
    detector_image = np.zeros((nx,ny))

    # Clip indices to ensure they lie within the detector bounds
    #x_pixel_indices = np.clip(x_pixel_indices, 0, nx - 1)
    #y_pixel_indices = np.clip(y_pixel_indices, 0, ny - 1)

    print(f"scat:amp: {intensities.shape}")
    print(f"detector_image: {detector_image.shape}")
    # Populate the image array with intensities
    
    
    for i in range(len(intensities)):
        detector_image[indices[i,0], indices[i,1]] += intensities[i]

    return detector_image


def reverse_kouts_to_pixels(kouts, detector_size, pixel_size, detector_distance):
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
    #kouts = kouts/np.linalg.norm(kouts, axis = 1)[:,np.newaxis]
    print(kouts)
    
    # Convert pixel size to meters
    pixel_size = np.array(pixel_size) * 1e-6
    
    # Compute number of pixels in each dimension
    nx, ny = (detector_size / pixel_size).astype(int)

    print(nx,ny)
    
    # Reverse mapping to pixel indices
    x_pixels = (kouts[:, 0] / kouts[:, 2]) * detector_distance / pixel_size[0] + nx / 2
    y_pixels = (kouts[:, 1] / kouts[:, 2]) * detector_distance / pixel_size[1] + ny / 2


    
    # Convert to integer pixel indices using rounding
    x_pixel_indices = np.floor(x_pixels).astype(int)
    y_pixel_indices = np.floor(y_pixels).astype(int)

    print(x_pixel_indices.max())
    print(x_pixels)
    
    # Verify indices are within bounds
    # Clip indices and log out-of-bounds
    out_of_bounds = (x_pixel_indices < 0) | (x_pixel_indices >= nx) | \
                    (y_pixel_indices < 0) | (y_pixel_indices >= ny)

    if np.any(out_of_bounds):
        print(f"Clipping {np.sum(out_of_bounds)} out-of-bounds indices.")
        x_pixel_indices = np.clip(x_pixel_indices, 0, nx - 1)
        y_pixel_indices = np.clip(y_pixel_indices, 0, ny - 1)
        
    return np.vstack((y_pixel_indices, x_pixel_indices)).T


