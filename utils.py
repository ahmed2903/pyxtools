import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas
from math import sin, cos, pi


hplanck = 6.62606868E-34  # Js  Plank consant
c = 299792458  # m/s   Speed of light
Ang = 1e-10  # m Angstrom
echarge = 1.6021733E-19  # C  electron charge
emass = 9.109e-31  # kg Electron rest mass
r0 = 2.8179403227e-15  # m classical electron radius = e^2/(4pi*e0*me*c^2)
Cu = 8.048  # Cu-Ka emission energy, keV
Mo = 17.4808  # Mo-Ka emission energy, keV

def indexQ(Qvec, rlv):
    """_summary_

    Args:
        Qvec (vector): Q vector (n,3)
        rlv (vector): Reciprocal lattice vectors (3,3) (astar, bstar, cstar)
    """

    HKL = np.round(np.dot(Qvec, np.linalg.inv(rlv))).astype(int)
    return HKL

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
    
    return Rz @ Ry @ Rx



# Define the objective function
def objective_function(params, kouts, G_initial, ttheta):
    # Rotate G using Euler angles
    alpha, beta, gamma = params
    R = rotation_matrix(alpha, beta, gamma)
    G = R @ G_initial
    C = G/2
    
    k_mag = 2*np.pi/exp.wavelength 
    R_circle = k_mag * np.cos(ttheta)
    Gmag = 4*np.pi / exp.wavelength * np.sin(ttheta/2)

    
        
    # Update k_in
    kin = kouts - G

    print(np.mean(np.linalg.norm(kin, axis = 1)))
    # Combine kout and kin in one array for the circle condition
    ks = np.vstack((kin, kouts))
    
    # Compute deviations for k_out (circle condition)
    f_circle = np.mean(np.abs(np.linalg.norm(kin - C, axis=1) - R_circle) / R_circle)

    # Compute error in the 2theta angle 
    dots = np.sum( (kin/np.linalg.norm(kin, axis=1)[:,np.newaxis] ) * (kouts/np.linalg.norm(kouts, axis=1)[:,np.newaxis]), axis = 1)
    f_angle = np.sum( np.abs( (dots - np.cos(ttheta)) / np.cos(ttheta) ) ) # np.std(dots)**2) #

    # Compute error in magnitude of k_in
    f_kmag = np.sum ( np.abs( np.linalg.norm(kin, axis = 1) - k_mag ) / k_mag )

    # compute error in the magnitude of G 
    f_gmag = (np.linalg.norm(G) - Gmag)**2 / Gmag**2

    print(f"error in angle: {f_angle}, radius: {f_circle}, kmag: {f_kmag}")

    w_angle = 1
    w_circle = 0
    w_kmag = 0
    
    return (w_angle*f_angle + w_circle*f_circle + w_kmag * f_kmag)/(w_circle+w_angle+ w_kmag) #+ 5*f_kmag  + f_gmag

def optimise_kin(G_init, ttheta, kouts, wavelength):

    # Initial guess for optimization
    initial_guess = [0, 0, 0]

    # Perform optimization
    result = minimize(objective_function, initial_guess, args=(kouts, G_init, ttheta),
                  method='BFGS', options={'disp': True, 
                                        "gtol" : 1e-6, 
                                        #'return_all': True
                                        })

    # Optimized orientation
    optimal_angles = result.x

    opt_mat = rotation_matrix(*optimal_angles)
        
    G_opt =  opt_mat @ G_init
    
    kin_opt = (kouts-G_opt)
    print(optimal_angles)
    kin_opt /= np.linalg.norm(kin_opt, axis = 1)[:,np.newaxis]
    kin_opt *= 2*np.pi/wavelength

    return kin_opt