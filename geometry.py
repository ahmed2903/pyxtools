
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

def mag(A):
    """
    Returns the magnitude of vector A
    If A has 2 dimensions, returns an array of magnitudes

    E.G.
     mag([1,1,1]) = 1.732
     mag(array([[1,1,1],[2,2,2]]) = [1.732, 3.464]
    """
    A = np.asarray(A, dtype=float)
    return np.sqrt(np.sum(A ** 2, axis=len(A.shape) - 1))

def cart2sph(xyz, deg=False):
    """
    Convert coordinates in cartesian to coordinates in spherical
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    ISO convention used.
        theta = angle from Z-axis to X-axis
          phi = angle from X-axis to component in XY plane
    :param xyz: [n*3] array of [x,y,z] coordinates
    :param deg: if True, returns angles in degrees
    :return: [r, theta, phi]
    """
    xyz = np.asarray(xyz).reshape(-1, 3)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    r = mag(xyz)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # theta = np.arctan2(xyz[:,2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    if deg:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
    return np.vstack((r, theta, phi)).T


def sph2cart(r_th_ph, deg=False):
    """
    Convert coordinates in spherical to coordinates in cartesian
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
        radius = sphere radius
        theta = angle from Z-axis to X-axis
          phi = angle from X-axis to component in XY plane
    :param r_th_ph: [[radius, theta, phi], ]
    :param deg: if True, converts theta, phi from degrees
    :return: [x,y,z]
    """
    r, theta, phi = np.transpose(r_th_ph)
    if deg:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.vstack((x, y, z)).T

def rot3D(A, alpha=0., beta=0., gamma=0.):
    r"""Rotate 3D vector A by euler angles
        A = rot3D(A,alpha=0.,beta=0.,gamma=0.)
       where alpha = angle from X axis to Y axis (Yaw)
             beta  = angle from Z axis to X axis (Pitch)
             gamma = angle from Y axis to Z axis (Roll)
       angles in degrees
       In a right-handed coordinate system.
           Z
          /|\
           |
           |________\Y
           \        /
            \
            _\/X
    """

    A = np.asarray(A, dtype=float).reshape((-1, 3))

    # Convert to radians
    alpha = alpha * np.pi / 180.
    beta = beta * np.pi / 180.
    gamma = gamma * np.pi / 180.

    # Define 3D rotation matrix
    Rx = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0., np.sin(gamma), np.cos(gamma)]])
    Ry = np.array([[np.cos(beta), 0., np.sin(beta)], [0., 1., 0.], [-np.sin(beta), 0., np.cos(beta)]])
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0.], [np.sin(alpha), np.cos(alpha), 0.], [0., 0., 1.]])
    R = np.dot(np.dot(Rx, Ry), Rz)

    # Rotate coordinates
    return np.dot(R, A.T).T



def plane_intersection(line_point, line_direction, plane_point, plane_normal):
    """
    Calculate the point at which a line intersects a plane
    :param line_point: [x,y],z] some coordinate on line
    :param line_direction: [dx,dy],dz] the direction of line
    :param plane_point:  [x,y],z] some coordinate on the plane
    :param plane_normal: [dx,dy],dz] the normal vector of the plane
    :return: [x,y],z]
    """

    line_point = np.asarray(line_point)
    plane_point = np.asarray(plane_point)
    line_direction = np.asarray(line_direction) / np.sqrt(np.sum(np.square(line_direction)))
    plane_normal = np.asarray(plane_normal) / np.sqrt(np.sum(np.square(plane_normal)))

    u1 = np.dot(plane_normal, plane_point - line_point)
    u2 = np.dot(plane_normal, line_direction)

    if u2 == 0:
        print('Plane is parallel to line')
        return None
    u = u1 / u2
    intersect = line_point + u*line_direction
    return intersect