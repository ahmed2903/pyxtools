import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from math import sin, cos, pi
import itertools as it

def calc_realspace_lattice_vectors(uc_size, uc_angles) -> np.ndarray:
    """
    Calculates the unit cell vectors for the crystal lattice. 

    input: 
        - uc_size: [a,b,c] A list of the size of the unit cell in Angstroms.
        - uc_angles: [alpha, beta, gamma] A list of the angles between the real space vectors in degrees. 

    output: A numpy array of the real space lattice vectors.
    """

    # Define primittive lattice vectors
    x = np.array([1,0,0])
    y= np.array([0,1,0])
    z = np.array([0,0,1])


    alpha = np.deg2rad(uc_angles[0])
    beta = np.deg2rad(uc_angles[1])
    gamma = np.deg2rad(uc_angles[2])

    a = uc_size[0]
    b = uc_size[1]
    c = uc_size[2]

    # Calculate lattice vector directions
    avec = a * x 
    bvec = b * x * cos(gamma)  + b * y * sin(gamma)
    cvec = c * cos(beta) * x + c * ((cos(alpha) - cos(beta)*cos(gamma))/ sin(gamma)) * y + z* c * np.sqrt( 1 - cos(beta)**2 - ((cos(alpha) - cos(beta)*cos(gamma))/ sin(gamma))**2)

    realspaceVecs = np.array([avec,bvec,cvec])

    return realspaceVecs


def calc_reciprocal_lattice_vectors(realspaceVecs: np.ndarray) -> np.ndarray:
    """
    Calculate the reciprocal lattice vectors from the real space vectors. 

    input: 
        - realspaceVecs: An array of the real space lattice vectors directions [A]. 

    output: A numpy array of the reciprocal space lattice vectors. 
    """

    avec = realspaceVecs[0]
    bvec = realspaceVecs[1]
    cvec = realspaceVecs[2]

    # Calculate Reciprocal lattice vectors
    a_star =  2*pi * np.cross(bvec, cvec) / (np.dot(avec, np.cross(bvec,cvec)))
    b_star =  2*pi * np.cross(cvec, avec) / (np.dot(bvec, np.cross(cvec,avec)))
    c_star =  2*pi * np.cross(avec, bvec) / (np.dot(cvec, np.cross(avec,bvec)))

    recpspaceVecs = np.array([a_star, b_star, c_star])

    return recpspaceVecs

def calculateLatticeSpacing(hkl, uc_size, uc_angles) -> float:
    
    """Calculates the interlattice spacing for a particular plane.

    Input: 
        - hkl: list of integers representing the Miller indices of the plane.
        - uc_size: [a,b,c] A list of the size of the unit cell in Angstroms.
        - uc_angles: [alpha, beta, gamma] A list of the angles between the real space vectors in degrees.

    Returns:
        The interlattice spacing, d_{hkl}
    """

    h = hkl[0]
    k = hkl[1]
    l = hkl[2]

    # convert degrees to radians
    alpha = np.deg2rad(uc_angles[0])
    beta = np.deg2rad(uc_angles[1])
    gamma = np.deg2rad(uc_angles[2])

    a = uc_size[0]
    b = uc_size[1]
    c = uc_size[2]

    inv = (1
        + 2 * cos(alpha)* cos(beta)* cos(gamma) 
        - cos(alpha)**2 - cos(beta)**2 
        - cos(gamma)**2) * (h**2 * sin(alpha)**2 / a**2 
        + k**2 * sin(beta)**2 /b**2 
        + l**2 * sin(gamma)**2 /c**2 
        + 2*h*k / (a*b) * ( cos(alpha)*cos(beta)- cos(gamma)) 
        + 2*l*k / (c*b) * ( cos(beta)*cos(gamma)- cos(alpha)) 
        + 2*h*l / (a*c) * ( cos(gamma)*cos(alpha) - cos(beta))
        )
    
    dhkl =  1/inv
    
    return dhkl

def generate_recip_lattice_points(recpspaceVecs: np.ndarray, max_hkl: int) -> np.ndarray:

    """_summary_
    Generates a set of reciprocal lattice points 

    Input: 
        - recpspaceVecs: A numpy array of the reciprocal space vectors of the system [A^-1]
        - max_hkl: The maximum Miller index value to generate

    Returns:
        H_hkl: A numpy array containing a set of reciprocal lattice points. 
    """
    h_range = range(-max_hkl, max_hkl + 1)
    k_range = range(-max_hkl, max_hkl + 1)
    l_range = range(-max_hkl, max_hkl + 1)
    H_hkl = []

    for h in h_range:
        for k in k_range:
            for l in l_range:
                if h!=0 and k!=0 and l!=0:
                    H = h * recpspaceVecs[0] + k * recpspaceVecs[1] + l * recpspaceVecs[2]
                    H_hkl.append(H)

    H_hkl = np.array(H_hkl)

    return H_hkl

def generate_realspace_lattice_points(N_ucs: int, realspaceVecs: np.ndarray) -> np.ndarray:
    """
    Generates a set of real space lattice points from the real space vectors

    Args:
        N_ucs (int): Number of unit cells 
        realspaceVecs (np.ndarray): Directions of the real space lattice vectors [A]

    Returns:
        np.ndarray: Array of the real sapce lattice points. 
    """

    h_range = range(-N_ucs, N_ucs + 1)
    k_range = range(-N_ucs, N_ucs + 1)
    l_range = range(-N_ucs, N_ucs + 1)

    R_n = []
    for h in h_range:
        for k in k_range:
            for l in l_range:
                if h != 0 and k != 0 and l != 0:
                    R = h * realspaceVecs[0] + k * realspaceVecs[1] + l * realspaceVecs[2]
                    R_n.append(R)

    return np.array(R_n)

def set_shape_array(arraysize, normals):
    """
    Create a shape function array representing the crystal's geometry.

    Args:
        arraysize (tuple): Dimensions of the 3D array representing the crystal grid.
        normals (list): List of facet normal vectors. Each facet is defined by a set of
                        start and end coordinates [x0, y0, z0, x1, y1, z1].

    Returns:
        None: The indices, and the shape array
    """
    # Initialize the shape array
    shape_array = np.zeros(arraysize, dtype=np.uint8)  # Binary array to represent shape (0 or 1)
    n_voxels = np.prod(arraysize)

    # Generate 3D grid indices for the shape array
    indices = np.transpose(np.unravel_index(np.arange(n_voxels), arraysize))

    # Process each surface defined by normals
    for surface in normals:
        start = np.array(surface[:3])  # Start coordinates
        end = np.array(surface[3:])   # End coordinates
        normal = (end - start)
        normal /= np.linalg.norm(normal)  # Normalize the normal vector

        # Determine which points lie "inside" the surface
        inside = np.dot(indices - start, normal) < 0
        shape_array[inside] = 1  # Mark voxels inside the shape as 1

    # Store the shape array
    shape_array = np.ascontiguousarray(shape_array, dtype=np.uint8)
    
    return indices, shape_array