
import numpy as np 
from math import cos, sin, pi, fabs
import scipy.ndimage
import math

def X_Rot(x, right_handed = True):
    """
    Right handed mu rotation about x axis
    """
    x = np.deg2rad(x)
    if right_handed == False:
            x = -x 

    return np.array([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)] ]) 

def Y_Rot(y, right_handed = True):

    
    if right_handed == False:
        y = -y
    
    y = np.deg2rad(y)

    return np.array([
        [cos(y), 0, sin(y)],
        [0, 1, 0],
        [-sin(y), 0, cos(y)]
    ])

def Z_Rot(z, right_handed = True): 

    z = np.deg2rad(z)
    if right_handed == False:
        z = -z

    return np.array([
        [cos(z), -sin(z), 0],
        [sin(z), cos(z), 0],
        [0, 0, 1]
    ])

    
def StanRotMat(chi, mu, eta, phi):        
    
    #Standard System
    mu_rot = X_Rot(mu, right_handed = True)
    eta_rot = Z_Rot(eta, right_handed = False)
    chi_rot = Y_Rot(chi, right_handed = True)
    phi_rot = Z_Rot(phi, right_handed = False)
    
    rotmat = np.dot(mu_rot, np.dot(eta_rot, np.dot(chi_rot, phi_rot)))

    return rotmat

def InvStanRotMat(chi, mu, eta, phi):

    #Standard System
    mu_rot = X_Rot(mu, right_handed = True)
    eta_rot = Z_Rot(eta, right_handed = False)
    chi_rot = Y_Rot(chi, right_handed = True)
    phi_rot = Z_Rot(phi, right_handed = False)
    
    rotmat = np.dot(phi_rot, np.dot(chi_rot, np.dot(eta_rot, mu_rot)))

    return rotmat

def CalcQ(delta):
    
    k_i = [0.0,1.0,0.0]
    
    delta_rot = Z_Rot(delta, right_handed = False)
    
    k_f = np.dot(delta_rot, k_i) 
    Q_vec = k_f - k_i
    
    return Q_vec
    
    
def rotation_matrix(axis, num):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    theta = 2*np.pi/num
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def CalculateCone(vector, theta, num_vectors):
    
    """
    Takes a central vector and return a cone on vectors around that formed by theta degrees. 

    Returns:
        np.ndarray: An array of vectors on the cone 
    """

    theta = np.deg2rad(theta)
    vector = vector / np.linalg.norm(vector)

    # Define a different vector 
    not_vector = np.array([1,1,0])

    # Calculate Normal vector 
    normal = np.cross(vector,not_vector)
    normal = normal/np.linalg.norm(normal)
    normal_2 = np.cross(vector, normal)

    # Calculate first vector
    vector_2 = vector * np.cos(theta) + normal * np.sin(theta)

    #Rotation Matrix
    rotmat = rotation_matrix(vector,num_vectors)

    # Generate the series of vectors
    vectors = []
    for i in range(num_vectors):
        if i == 0:
            vectors.append(vector_2)	
        else:
            new_vector = rotmat.dot(vectors[-1])
            new_vector /= np.linalg.norm(new_vector)
            vectors.append(new_vector)


    return vectors

def compute_detector_distance_from_NA(NA, pupil_size):

    Ld2 = pupil_size / 2

    theta_d2 = np.arcsin(NA)
    
    distance = Ld2 / np.tan( theta_d2 )

    return distance


def calculate_NA(focal_length, height):

    return np.sin(np.arctan(height/(2*focal_length)))