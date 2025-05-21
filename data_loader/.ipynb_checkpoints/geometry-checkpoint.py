
import numpy as np

def estimate_pupil_size(array, mask_val, pixel_size, pupil_roi=None, crop=True):
    """
    Computes the average length of a box with jagged edges in x and y directions.

    Args:
        array (2D ndarray): A 2D array where the box is represented by non-zero values.
        mask_val (float): value of the mask
        pixel_size (float): size of the detector pixel
        pupil_roi (list): a 4 element list containing the roi of the pupil [xs, xe, ys, ye]

    Returns:
        tuple: (average_x_length, average_y_length)
    """
    if pupil_roi is not None: 
        array = array[pupil_roi[0]:pupil_roi[1], pupil_roi[2]:pupil_roi[3]]

    min_lx = array.shape[0] /3
    min_ly = array.shape[1] /3
    
    nx , ny = np.where(array>mask_val)

    x_lengths = []
    for x in np.unique(nx):
        y_indices = np.where(array[x,:] > mask_val)[0]
        if len(y_indices) > min_lx:
            x_lengths.append(y_indices[-1] - y_indices[0] + 1)

    
    y_lengths = []
    for y in np.unique(ny):
        x_indices = np.where(array[:,y] > mask_val)[0]
        if len(x_indices) > min_ly:
            y_lengths.append(x_indices[-1] - x_indices[0] + 1)

    x_lengths = np.array(x_lengths) 
    y_lengths = np.array(y_lengths)

    avg_x = np.max(x_lengths) * pixel_size
    avg_y = np.max(y_lengths) * pixel_size

    return avg_x, avg_y
    
def estimate_detector_distance_from_NA(NA, pupil_size):

    Ld2 = pupil_size / 2

    theta_d2 = np.arcsin(NA)
    
    distance = Ld2 / np.tan( theta_d2 )

    return distance


def calculate_NA(focal_length, height):

    return np.sin(np.arctan(height/(2*focal_length)))
