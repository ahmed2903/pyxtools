''' Image Processing toolbox'''

import numpy as np
from skimage.measure import block_reduce


def DS_img(img, target_shape):
    '''
    Downsampling the image to the target shape
    
    Parameters--
    img : image to be downsampled
    target_shape : (tuple) shape to which image is downsampled

    Returns--
    downsampled_img : downsampled image
    
    '''
    blocksize = (int(img.shape[0]//target_shape[0]), int(img.shape[1]//target_shape[1]))
    ds_img = block_reduce(img, block_size = blocksize, func = np.mean)
    downsampled_img = ds_img[:target_shape[0], :target_shape[1]]
    return downsampled_img

def preprocess_image(img, target_size=(1024, 1024)):
    """
    Preprocesses an input image for analysis by converting to grayscale,
    resizing, and normalizing pixel values.
    
    Parameters--
    img : PIL.Image
        Input image object (can be RGB, RGBA, or any PIL-supported format)
    target_size : tuple of int, optional
        Target dimensions (width, height) for resizing. Default is (1024, 1024)
    
    Returns--
    img: numpy.ndarray
        Preprocessed grayscale image as normalized float array with values [0, 1]
    """
    
    # Convert image to grayscale (luminance mode)
    # This reduces RGB/RGBA images to single channel based on perceived brightness
    img = img.convert("L")
    
    # Resize the image to target dimensions
    # Uses PIL's default resampling method (usually LANCZOS for downsampling)
    img = img.resize((target_size[0], target_size[1]))
    
    # Convert PIL Image to numpy array for numerical operations
    # Results in 2D array with pixel values typically in range [0, 255]
    img = np.array(img)
    
    # Normalize pixel values to range [0, 1] by dividing by maximum value
    # This ensures consistent scaling regardless of original bit depth
    img = img / img.max()
    
    return img

