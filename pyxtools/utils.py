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

    HKL = np.dot(Qvec, np.linalg.inv(rlv)).astype(int)
    return HKL