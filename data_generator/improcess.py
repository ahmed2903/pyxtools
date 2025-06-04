''' Image Processing toolbox'''

import numpy as np
from skimage.measure import block_reduce


def DS_img(img, target_shape):
    '''
    Downsampling the image to the target shape '''
    blocksize = (int(img.shape[0]//target_shape[0]), int(img.shape[1]//target_shape[1]))
    ds_img = block_reduce(img, block_size = blocksize, func = np.mean)
    downsampled_img = ds_img[:target_shape[0], :target_shape[1]]
    return downsampled_img

def prep_img(img, target_size=1024):
    img =  img.convert("L")
    # Resize the image
    img = img.resize((target_size, target_size))
    img = np.array(img)
    img = img/img.max()
    return img

def gseq_square(arraysize):
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