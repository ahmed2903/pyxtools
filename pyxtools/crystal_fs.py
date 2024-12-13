from turtle import shapetransform
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from math import sin, cos, pi, sqrt
import itertools as it

from pyxtools.general_fs import pad

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
    l_range = range(-max_hkl, max_hkl + 1)
    k_range = range(-max_hkl, max_hkl + 1)
    
    H_hkl = []

    for h in h_range:
        for k in k_range:
            for l in l_range:
                if not (h == 0 and k == 0 and l == 0):
                #if h!=0 and k!=0 and l!=0:
                    H = h * recpspaceVecs[0] + k * recpspaceVecs[1] + l * recpspaceVecs[2]
                    H_hkl.append(H)
    
    H_hkl = np.array(H_hkl)
    
    return H_hkl

def generate_recip_lattice_points_hkl(recpspaceVecs: np.ndarray, hkl:tuple, hkl_range, gridpoints) -> np.ndarray:

    """_summary_
    Generates a set of reciprocal lattice points 

    Input: 
        - recpspaceVecs: A numpy array of the reciprocal space vectors of the system [A^-1]
        - hkl: The  Miller index value of the reflection of interest to generate

    Returns:
        H_hkl: A numpy array containing a set of reciprocal lattice points around the desired reflection . 
    """
    
    hkl_freq = (2*hkl_range) / gridpoints 
        
    h_range = np.arange(-hkl_range, hkl_range, hkl_freq) * np.linalg.norm(recpspaceVecs[0])
    k_range = np.arange(-hkl_range, hkl_range, hkl_freq) * np.linalg.norm(recpspaceVecs[1])
    l_range = np.arange(-hkl_range, hkl_range, hkl_freq) * np.linalg.norm(recpspaceVecs[2])
    
    print("hrange is: ")
    print(h_range)
    
    h,k,l = hkl
    Gvec = h*recpspaceVecs[0] + k*recpspaceVecs[1] + l*recpspaceVecs[2]
    
    H_hkl = []

    for ih in h_range:
        for ik in k_range:
            for il in l_range:
                delta_q = ih * recpspaceVecs[0] + ik * recpspaceVecs[1] + il * recpspaceVecs[2]
                H_hkl.append(Gvec+delta_q)
    
    H_hkl = np.array(H_hkl)
    
    return H_hkl

def generate_recip_lattice_points2(recpspaceVecs: np.ndarray, max_hkl: int, ravel = False) -> np.ndarray:

    """_summary_
    Generates a set of reciprocal lattice points 

    Input: 
        - recpspaceVecs: A numpy array of the reciprocal space vectors of the system [A^-1]
        - max_hkl: The maximum Miller index value to generate

    Returns:
        H_hkl: A numpy array containing a set of reciprocal lattice points. 
    """
    # Define h, k, l ranges
    h_range = np.linspace(-max_hkl, max_hkl, 2*max_hkl+1) * np.linalg.norm(recpspaceVecs[0])
    k_range = np.linspace(-max_hkl, max_hkl, 2*max_hkl+1) * np.linalg.norm(recpspaceVecs[1])
    l_range = np.linspace(-max_hkl, max_hkl, 2*max_hkl+1) * np.linalg.norm(recpspaceVecs[2])
    
    H_hkl = []
    
    h_grid, k_grid, l_grid = np.meshgrid(h_range, k_range, l_range, indexing="ij")
    

    if ravel:
        # Stack into a single array of shape (grid_points**3, 3)
        q_vectors = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=-1)
    else: 
        q_vectors = np.stack((h_grid, k_grid,l_grid), axis=-1)

    return q_vectors



    
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


def gen_RLS_from_maxhkl(max_hkl, grid_points, a, b, c, ravel=False):
    """
    Generate reciprocal space q-vectors for a general lattice.

    Parameters:
        max_hkl (int): Maximum value of h, k, l indices.
        grid_points (int): Number of grid points along each axis.
        a, b, c (np.ndarray):(reciprocal-space lattice vectors).

    Returns:
        q_vectors (numpy.ndarray): Array of shape (grid_points**3, 3)
                                   containing the q-vectors.
    """
    # Reciprocal lattice basis vectors
    b1 = np.linalg.norm(a)
    b2 = np.linalg.norm(b)
    b3 = np.linalg.norm(c)
        
    # Define h, k, l ranges
    h = np.linspace(-max_hkl, max_hkl, grid_points) 
    k = np.linspace(-max_hkl, max_hkl, grid_points) 
    l = np.linspace(-max_hkl, max_hkl, grid_points) 

    # Generate 3D grid of q-space
    h_grid, k_grid, l_grid = np.meshgrid(h, k, l, indexing="ij")
    
    qx = h_grid * b1 
    qy = k_grid * b2
    qz = l_grid * b3

    if ravel:
        # Stack into a single array of shape (grid_points**3, 3)
        q_vectors = np.stack((qx.ravel(), qy.ravel(), qz.ravel()), axis=-1)
    else: 
        q_vectors = np.stack((qx, qy, qz), axis=-1)

    return q_vectors
def gen_RLS_from_maxhkl_maskOrigin(max_hkl, grid_points, a, b, c, threshold=0.1, ravel=False):
    """
    Generate reciprocal space q-vectors for a general lattice and remove the 0,0,0 order and surrounding cube.

    Parameters:
        max_hkl (int): Maximum value of h, k, l indices.
        grid_points (int): Number of grid points along each axis.
        a, b, c (np.ndarray): Reciprocal-space lattice vectors.
        threshold (float): Distance threshold to exclude points around (0, 0, 0).
        ravel (bool): Whether to return the q-vectors as a flat array.

    Returns:
        q_vectors (numpy.ndarray): Array of q-vectors with the 0,0,0 region removed.
    """
    # Reciprocal lattice basis vector magnitudes
    b1 = np.linalg.norm(a)
    b2 = np.linalg.norm(b)
    b3 = np.linalg.norm(c)
        
    # Define h, k, l ranges
    h = np.linspace(-max_hkl, max_hkl, grid_points) 
    k = np.linspace(-max_hkl, max_hkl, grid_points) 
    l = np.linspace(-max_hkl, max_hkl, grid_points) 

    # Generate 3D grid of q-space
    h_grid, k_grid, l_grid = np.meshgrid(h, k, l, indexing="ij")
    
    # Calculate qx, qy, qz components
    qx = h_grid * b1 
    qy = k_grid * b2
    qz = l_grid * b3

    # Combine components into a single array
    q_vectors = np.stack((qx, qy, qz), axis=-1)
    
    # Compute the distance of each point from the origin
    distances = np.sqrt(qx**2 + qy**2 + qz**2)

    # Mask out the region near the origin (including 0,0,0)
    mask = distances > threshold

    if ravel:
        # Stack into a single array of shape (grid_points**3, 3)
        q_vectors = q_vectors[mask]
        
    else: 
        q_vectors[~mask] = np.nan 

    return q_vectors

def gen_rlvs_for_one_hkl(hkl:tuple, recip_vecs, grid_points, range_scale, ravel = True):
    
    h,k,l = hkl
    
    G_vec = h*recip_vecs[0] +  k*recip_vecs[1] + l*recip_vecs[2]
    
    # Reciprocal lattice basis vector magnitudes
    b1 = np.linalg.norm(recip_vecs[0])
    b2 = np.linalg.norm(recip_vecs[1])
    b3 = np.linalg.norm(recip_vecs[2])
        
    # Define h, k, l ranges
    h = np.linspace(-1, 1, grid_points) * b1 * range_scale
    k = np.linspace(-1, 1, grid_points) * b2 * range_scale 
    l = np.linspace(-1, 1, grid_points) * b3 * range_scale

    print(h * b1)
    # Generate 3D grid of q-space
    h_grid, k_grid, l_grid = np.meshgrid(h, k, l, indexing="ij")
    
    # Calculate qx, qy, qz components
    qx = h_grid 
    qy = k_grid 
    qz = l_grid 
    
    # Combine components into a single array
    q_vectors = G_vec + np.stack((qx, qy, qz), axis=-1)
    
    if ravel:
        q_vectors = q_vectors.reshape(-1,3)
    
    
    print("qvector at genertion")
    print(q_vectors.shape)
    return q_vectors, (h,k,l)
    

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
    
        
        shape_array[tuple(indices[inside].T)] = 1
        #flat_shape_array[inside] = 1  # Mark voxels inside the shape as 1s

    # Store the shape array
    shape_array = np.ascontiguousarray(shape_array, dtype=np.uint8)
    
    shape_array = pad(shape_array, 2,2,2,2,2,2)
    
    return indices, shape_array

def cuboid_normals(arraysize):
    
    X=arraysize[0]
    Y=arraysize[2]
    Z=arraysize[2]
    dX = 30
    X2=X/2 - dX
    Y2=Y/2 - dX
    Z2=Z/2 - dX
    s3 = sqrt(3)
    
    normals = [
    [dX,Y2,Z2,dX-1,Y2,Z2],\
    [X-dX,Y2,Z2,X,Y2,Z2],\
    [X2/2+X/2,s3*Y2/2+Y/2,Z2,X2/1.9+X/2,s3*Y2/1.9+Y/2,Z2],\
    [X2/2+X/2,-s3*Y2/2+Y/2,Z2,X2/1.9+X/2,-s3*Y2/1.9+Y/2,Z2],\
    [-X2/2+X/2,s3*Y2/2+Y/2,Z2,-X2/1.9+X/2,s3*Y2/1.9+Y/2,Z2],\
    [-X2/2+X/2,-s3*Y2/2+Y/2,Z2,-X2/1.9+X/2,-s3*Y2/1.9+Y/2,Z2],\
    [X2,Y2,dX,X2,Y2,dX-1],\
    [X2,Y2,Z-dX,X2,Y2,Z]
    ]
    
    return normals



def compute_shape_transform(shape_array, grid_spacing):
    
    shapetransform = np.fft.fftshift(np.fft.fftn(shape_array))
    
    # Create the reciprocal space grid
    nx, ny, nz = shape_array.shape
    dx, dy, dz = grid_spacing

    # Define the reciprocal grid extents
    qx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi
    qy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi
    qz = np.fft.fftshift(np.fft.fftfreq(nz, d=dz)) * 2 * np.pi
    
    qx, qy, qz = np.meshgrid(qx, qy, qz, indexing='ij')

    return shapetransform, (qx, qy, qz)


