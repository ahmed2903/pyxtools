import numpy as np



def get_pupil(cl, rl, ch, rh):
    ''' 
    This function makes a rectangular pupil
    rl, cl : lower row and column indices
    rh, ch : upper row and column indices
    '''
    Pupil = np.zeros((n,m))
    Pupil[rl:rh, cl:ch] = 1
    return Pupil