import numpy as np 
import matplotlib.pyplot as plt
import h5py
import os
from PIL import Image
import inspect

from .signals import *


class field:
    '''
    Class for loading the input fields
    '''
    def __init__(self, array_size, pixel_size):
        """
        Initialize the field generator parameters

        Parameters:
        array_size (tuple): (Nr, Nc) Number of pixels in rows X columns
        Defines the size of the computational window
        Nr and Nc needs to be integers
        pixel_size (tuple): (pixel_row, pixel_col) Pixel Size
        """
        self.array_size = array_size
        self.pixel_size = pixel_size
          

    def get_kwargs(self, obj, kwargs):
        '''
        Extracts valid keyword arguments for a given object's function signature

        Parameters:
        obj : callable
        The target function or method whose signature will be inspected
        
        kwargs : dict
        Dictionary of keyword arguments (e.g. user-provided settings)

        Returns:
        obj_args : dict
        Filtered dictionary containing only the keyword arguments
        that match the parameter names in `obj`'s signature.
        '''
        obj_sigs = []
        obj_args = {}
        for arg in inspect.signature(obj).parameters.values():
            if not arg.default is inspect._empty:
                obj_sigs.append(arg.name)
        for key, value in kwargs.items():
            if key in obj_sigs:
                obj_args[key] = value
        return obj_args
    

    def _get_signal(self, name, **kwargs):
        """
        Generate a signal using the specified name of the function.
        
        Args:
            name (str): Name of the function to use for making the signal
            **kwargs: Function-specific arguments
            
        Returns:
            signal_array
        """
        
        # Get the function
        func = name 
        
        # Filter kwargs to only include valid arguments for this function
        valid_kwargs = self.get_kwargs(func, kwargs)

        # Call the function with filtered arguments
        signal_array = func(self.array_size, self.pixel_size, **valid_kwargs)
        return signal_array

    def make_signal(self, name, amp_or_pha, **kwargs):
        '''
        Generates either an amplitude-only and phase-only signal
    
        This method is directly used to create an amplitude or phase object

        Parameters:
        name (str):
            The name of signal to generate. 
            
        amp_or_pha(str):
            Indicates whether to generate an amplitude or phase signal.
            Must be either "amplitude" or "phase".
        
        **kwargs(dict):
            Additional keyword arguments passed directly to `_get_signal`, which
            includes signal parameters particular to the signal.
        '''

        if amp_or_pha == 'amplitude':
            self.amp = self._get_signal(name, **kwargs) 

        elif amp_or_pha == 'phase':
            self.phase = self._get_signal(name, **kwargs)

    def make_complex_field(self, amplitude_name, phase_name, amplitude_kwargs, phase_kwargs):
        """
        Generate a complex field with separate amplitude and phase patterns.
        
        Args:
        amplitude_name (str): Name of function for amplitude
        phase_name (str): Name of function for phase
        amplitude_kwargs (dict): Arguments for amplitude function
        phase_kwargs (dict): Arguments for phase function
        
        Returns:
        array: complex_field, where complex_field is amplitude * exp(1j * phase)
        """
       
        print("Loading amplitude...")
        self.amp = self._get_signal(amplitude_name, amp_or_pha = "amplitude", **amplitude_kwargs)  # <- Fixed: use **
        print("Amplitude shape:", self.amp.shape if self.amp is not None else "None")
            
        print("Loading phase...")
        self.phase = self._get_signal(phase_name, amp_or_pha = "phase", **phase_kwargs)  # <- Fixed: use **
        print("Phase shape:", self.phase.shape if self.phase is not None else "None")
        
        # Create complex field
        self.complex_field = self.amp * np.exp(1j * self.phase)
        
    
    def load_signal(self, amp_or_pha, path):

        ext = path.split('.')[-1]

        if ext == 'jpeg' or ext == 'png' or ext == "bmp" or ext == "jpg":
            img = Image.open(path)

        elif ext == "npy":
            img = np.load(path)
        
        else:
            raise ValueError("File Type Not Supported")

        if amp_or_pha == 'amplitude':
            self.amp = prep_img(img,  self.array_size) # FIX ME for the numpy array as well Pillow images

        elif amp_or_pha == 'phase':
            self.phase = prep_img(img,  self.array_size)
    
    def load_complex_field(self, amp_path=None, phase_path=None):
        """
        Load a complex field from amplitude and phase image files.
        
        Args:
            amp_path (str, optional): Path to amplitude image file. If None, creates uniform amplitude (ones).
            phase_path (str, optional): Path to phase image file. If None, creates zero phase.
        
        Returns:
            array: Complex field combining amplitude and phase
        """

        self.load_signal(amp_or_pha="amplitude", path= amp_path)
        self.load_signal(amp_or_pha="phase", path = phase_path)
        
        self.complex_field = self.amp * np.exp(1j * self.phase)

    def compute_complex_field(self):

        self.complex_field = self.amp * np.exp(1j * self.phase)



    
    
    ### Plotting functions
    def plot_complex_field(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        
        # amplitude pattern
        im1 = axes[0].imshow(self.amp,  cmap='magma')
        axes[0].set_title('Amplitude')
        plt.colorbar(im1, ax=axes[0], shrink=0.5)
        
        # Phase pattern
        im2 = axes[1].imshow(self.phase,  cmap='twilight')
        axes[1].set_title('Phase')
        plt.colorbar(im2, ax=axes[1], shrink=0.5)
        plt.tight_layout()



    def plot_signal(self, amp_or_pha):

        if amp_or_pha == 'amplitude':
            plt.imshow(self.amp, cmap = 'jet')
            plt.colorbar()

        elif amp_or_pha == 'phase':
            plt.imshow(self.phase, cmap = 'jet')
            plt.colorbar()

        elif amp_or_pha == 'complex':
            self.plot_complex_field()
        
