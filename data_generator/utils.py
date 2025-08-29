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

def create_2dgrid(Nr, Nc, dr, dc):
    """
    Creates a 2D coordinate grid with specified dimensions and spacing.
    
    Parameters:
    Nr : int
        Number of rows in the grid
    Nc : int
        Number of columns in the grid  
    dr : float
        Spacing between grid points in the row direction (y-direction)
    dc : float
        Spacing between grid points in the column direction (x-direction)
        
    Returns:
    tuple of numpy.ndarray
        x, y : 2D coordinate arrays of shape (Nr, Nc)
               x contains x-coordinates, y contains y-coordinates
               Grid is centered at origin (0, 0)
    """
    
    
    grid_x = np.arange(-Nc//2, Nc//2, 1) * dc  
    grid_y = np.arange(-Nr//2, Nr//2, 1) * dr  
    
    # Create 2D coordinate meshgrids
    # x varies along columns (horizontal direction)
    # y varies along rows (vertical direction)
    # Both arrays have shape (Nr, Nc)
    x, y = np.meshgrid(grid_x, grid_y)
    
    return x, y


def make_kin_grid(Npupil_row, Npupil_col, dkx, dky):
    """
    Creates a 2D grid of incident k-vectors (wave vectors) for pupil function sampling.
    
    Parameters:
    -----------
    Npupil_row : (int) Number of rows in the pupil grid
    Npupil_col : (int) Number of columns in the pupil grid
    dkx : (float) Sampling interval in kx direction (spatial frequency spacing)
    dky : (float) Sampling interval in ky direction (spatial frequency spacing)
        
    Returns:
    --------
    kins : (numpy.ndarray) Array of shape (Npupil_row*Npupil_col, 2) containing [kx, ky] coordinates
    for each point in the pupil grid, flattened into a list of vector pairs
    """   
    kx_in, ky_in = create_2dgrid(Npupil_row, Nppuil_col, dky, dkx)
    
    # Flatten 2D grids into 1D arrays
    # Each array contains coordinates for all grid points
    kx = kx_in.flatten() 
    ky = ky_in.flatten() 
    
    # Stack coordinates into array of [kx, ky] pairs
    # Shape: (Nr*Nc, 2) where each row is [kx_i, ky_i]
    kins = np.vstack((kx, ky)).T
    
    return kins

def gseq_square(arraysize):
    '''
    Generates a sequence of indices in NXN matrix, where N is equal to the arraysize,
    from centre to the edges. This is to configure the updating sequence of reconstruction
    in Fourier Ptychographic Imaging

    Parameters:
    array size: (int)size of the source array in one dimension

    Returns:
    seq: (numpy array) sequences
    
    '''
    n = ( arraysize + 1)//2
    #arraysize = 2*n - 1
    sequence = np.zeros((2, arraysize**2))
    sequence[0, 0] = n
    sequence[1, 0] = n
    dx = 1
    dy = -1
    stepx = 1
    stepy = -1
    direction = 1
    counter = 0
    for k in range(1, arraysize**2):
        counter = counter + 1
        if (direction == 1):
            sequence[0, k] = sequence[0, k-1] + dx
            sequence[1, k] = sequence[1, k-1]
            if (counter == np.abs(stepx)):
                counter = 0
                direction = direction*(-1)
                dx = dx*(-1)
                stepx = stepx * (-1)
                if (stepx>0):
                    stepx = stepx + 1
                else:
                    stepx = stepx - 1


        
        else:
            sequence[0, k] = sequence[0, k-1]
            sequence[1, k] = sequence[1, k-1] + dy
            if (counter==abs(stepy)):
                counter = 0
                direction = direction*(-1)
                dy = dy * (-1)
                stepy = stepy *(-1)
                if (stepy > 0):
                    stepy = stepy + 1
                else:
                    stepy = stepy - 1

    seq = (sequence[0, :] - 1)*arraysize + sequence[1, :]
    #seqf[0, 0:arraysize**2] = seq
    return seq