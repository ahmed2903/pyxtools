import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas
from math import sin, cos, pi
from scipy.optimize import minimize

hplanck = 6.62606868E-34  # Js  Plank consant
c = 299792458  # m/s   Speed of light
Ang = 1e-10  # m Angstrom
echarge = 1.6021733E-19  # C  electron charge
emass = 9.109e-31  # kg Electron rest mass
r0 = 2.8179403227e-15  # m classical electron radius = e^2/(4pi*e0*me*c^2)
Cu = 8.048  # Cu-Ka emission energy, keV
Mo = 17.4808  # Mo-Ka emission energy, keV

import time
import functools

def time_it(func):
    """Decorator to measure execution time of a function."""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Compute execution time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result  # Return original function result
    return wrapper

def get_meta_data(func):
    """Decorator to save the meta_data elements in the function"""
    meta_dat = []
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)  # Execute the function
        scalars = [arg for arg in args if not isinstance(arg, (list, tuple, dict))]
        meta_dat.append(scalars)
        return result  # Return original function result
        # Attach the metadata list to the wrapper so it's accessible
    wrapper.meta_data = meta_dat
    return wrapper

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

