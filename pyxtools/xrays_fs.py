import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from math import sin, cos, pi
import itertools as it

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
    theta = qmag2ttheta(np.linalg.norm(qvec, axis=-1), wavelength_a)

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
    
    f_q = 2*pi * np.sum(np.exp(-1j*np.dot(q_vec,R_i)), axis = -1) / V_cell 
    
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
    fjs = np.array(fjs)
    
    phase = np.exp(-1j*np.dot(q_vec,rj.T))
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
    form_factor = calculate_form_factor(real_lattice_vecs, q_vec, R_i)
    structure_factor = calculate_structure_factor(atoms, rj_atoms, q_vec, wavelength_a)
    
    scattering_amp = form_factor*structure_factor
    
    return scattering_amp
    
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
    k_magnitude = 2 * np.pi / wavelength
    k_vectors *= k_magnitude

    # Converging effect: adjust directions to point towards the focal point
    # For simplicity, let's assume the focal point lies along the z-axis and the beam converges to (0, 0, focal_length)
    # The beam is converging toward (0, 0, focal_length)
    focal_point = np.array([0, 0, focal_length])

    # Normalize each vector to point towards the focal point
    for i in range(num_vectors):
        # Direction vector from the k-vector's point to the focal point
        direction_to_focus = focal_point - k_vectors[i]
        # Normalize this direction
        direction_to_focus /= np.linalg.norm(direction_to_focus)
        # Update k-vector direction (the normalized vector)
        k_vectors[i] = direction_to_focus * k_magnitude

    return k_vectors