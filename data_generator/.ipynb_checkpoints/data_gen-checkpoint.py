
''' Generates the Fourier Ptychography Data '''

import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import imageio
from PIL import Image
from skimage.registration import phase_cross_correlation
import matplotlib.animation as animation
import pylab as pl
from IPython import display
import time
from scipy.stats import poisson

from .objects import *
from .utils import *
from .improcess import *

class generate_data:
    ''' 
    Class for generating the Fourier Ptychography data. The data 
    comprises of a set of images recorded at the image plane when 
    the overall object is illuminated by a tilt series of plane waves
    in a Conventional Transmission X-ray Microscope (CTXM).
    '''

    def __init__(self,
                 wavelength: float,
                 arraysize : int,
                 pt_source_gap : float,
                 source_object_distance : float,
                 comp_window: int,
                 det_psize: float,
                 recon_psize: float,
                 NA: float 
                ):
        
        self.numexp_params = {
            'wavelength' : wavelength, # Wavelength of the source
            'arraysize': arraysize,    # Number of point sources in the Source Array
            'pt_source_gap': pt_source_gap,  # (in mm) gap between two point sources
            'source_object_distance' : source_object_distance, # (in mm) distance between the source and the object
            'comp_window': comp_window,
            'det_psize' : det_psize,
            'recon_psize' : recon_psize,
            'NA' : NA 
        }

    def define_object(self, comp_window ):
        '''
        Importing the objects
        comp_window: size of the window over which object is defined
        '''
        # number of rows and columns
        self.Nrow = comp_window
        self.Ncol = comp_window
        self.object = get_vortex(comp_window, topo_charge = 1)

    def load_object(self, amp_path, phase_path, comp_window):
        img1 = Image.open(amp_path)
        img2 = Image.open(phase_path)
        amp = 0.5 + 0.5*prep_img(img1, comp_window)
        phi = prep_img(img2, comp_window)
        self.object = amp*np.exp(1j*phi)
        
        self.Nrow = comp_window
        self.Ncol = comp_window
        

    def define_source(self, arraysize, pt_source_gap, source_object_distance, wavelength, recon_psize):
        ''' This function will give a plane where a number of 
        sources are arranged as an array. This array of source 
        is the source of illumination, where each source element 
        illuminates the object one after another. 
        ---------------------------------------------------------
        The source parameters are:
        wavelength : Wavelength of the source
        arraysize : Number of point sources in the Source Array
        pt_source_gap :  gap between two sources
        source_object_distance : distance between the source and the object    
        '''
        
        # Initialize arrays
        self.xlocation = np.zeros(arraysize**2)
        self.ylocation = np.zeros(arraysize**2)
        
        # Generate LED positions
        for i in range(arraysize):  
            start_idx = arraysize * (i)
            end_idx = arraysize + arraysize * (i)
            
            # Create x positions (same for each row)
            self.xlocation[start_idx:end_idx] = np.arange(-(arraysize-1)/2, (arraysize-1)/2 + 1) * pt_source_gap
            
            # Create y positions (constant for each row)
            self.ylocation[start_idx:end_idx] = ((arraysize-1)/2 - (i)) * pt_source_gap
        
        self.theta_xz = np.arctan(self.xlocation / source_object_distance)
        self.theta_yz = np.arctan(self.ylocation / source_object_distance)
        
        kx_relative = -np.sin(np.arctan(self.xlocation / source_object_distance))
        ky_relative = -np.sin(np.arctan(self.ylocation / source_object_distance))

        self.dkx = 2*np.pi/(recon_psize*self.Ncol)
        self.dky = 2*np.pi/(recon_psize*self.Nrow)

        self.theta_xz = 

        # Wavevector magnitude
        self.k0 = 2*np.pi/wavelength

        # Wave-vector corresponding to each point source
        self.kx = kx_relative * self.k0
        self.ky = ky_relative * self.k0


    def generate_coherent_imgs(self, det_psize, recon_psize, arraysize):
        '''
        Generating the diffraction limited images of the object corresponding
        to the tilt direction of the plane wave. Here the tilt of the illuminating 
        beam is realized by the shift of the centre of the Object's Fourier Transform.
        The resolution limit of these images is determined by the pupil aperture
        or the NA of the imaging lens
        '''
        self.objectFT =  np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.object)))
        self.pixBW = int(det_psize//recon_psize)
        self.Npupil_row = int(self.Nrow//self.pixBW)
        self.Npupil_col = int(self.Ncol//self.pixBW)
        
        self.imSeqLowRes = np.zeros((arraysize**2, self.Npupil_row, self.Npupil_col))
        self.imSeqLowRes = self.imSeqLowRes.astype(np.float32)
        
        for tt in range(arraysize**2):
            kxc = np.round((self.Ncol+1)/2 + self.kx[tt]/self.dkx)
            kyc = np.round((self.Nrow+1)/2 + self.ky[tt]/self.dky)
            kyl = round(kyc- self.Npupil_row//2)
            kyh = round(kyc+ self.Npupil_row//2)
            kxl = round(kxc- self.Npupil_col//2)
            kxh = round(kxc+ self.Npupil_col//2)
            imSeqLowFT = (self.Npupil_col/self.Ncol)**2 * self.objectFT[kyl:kyh, kxl:kxh]
            self.imSeqLowRes[tt] = np.abs( np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT))) )

    def generate_coherent_imgs_RSmodel(self, arraysize, recon_psize):
        '''
        Generating the diffraction limited images of the object corresponding
        to the tilt direction of the plane wave. Here the tilted plane wave illuminates
        the object.

        arraysize: size of the source arrays, as each image of the object is corresponding 
        to one particular source
        recon_psize : Pixel size at the object plane (This is what we aim to recover 
        in Fourier Ptychography)
        '''
        
        # creating meshgrid for the object plane
        grid_x = np.arange(-self.Nrow//2, self.Nrow//2, 1)*recon_psize
        x, y = np.meshgrid(grid_x, grid_x)
        
        self.imSewLowRes_data = np.zeros((arraysize**2, self.Npupil_row, self.Npupil_col)).astype(np.float32)
        
        rl = self.Nrow//2 - self.Npupil_row//2
        rh = self.Nrow//2 + self.Npupil_row//2
        cl = self.Ncol//2 - self.Npupil_col//2
        ch = self.Ncol//2 + self.Npupil_col//2
        
        for tt in range(arraysize**2):
            this_obj = self.object *  np.exp(1j*(self.kx[tt]*x+ self.ky[tt]*y))
            this_obj_FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(this_obj)))
            this_obj_FT_filt = this_obj_FT[rl : rh, cl : ch]
            this_im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(this_obj_FT_filt)))
            self.imSewLowRes_data[tt] = np.abs(this_im)   

   


    # -------------------Plotting---------------------------------------


    def plot_object(self):
        ''' Plot amplitude and the Phase of the object'''
        
        amp = np.abs(self.object)
        phase = np.angle(self.object)

        fig, axes = plt.subplots(1,2, figsize= (12,5))
        object_amp = axes[0].imshow(amp, cmap='jet')
        axes[0].set_title(f'Object Amplitude')
        axes[0].axis('on')  
        plt.colorbar(object_amp, ax=axes[0], fraction=0.046, pad=0.04)
        
        object_phase= axes[1].imshow(phase, cmap='jet')
        axes[1].set_title(f'Object Phase')
        axes[1].axis('on')
        plt.colorbar(object_phase, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

    def plot_source(self):
        '''  
        Plot Source array positions
        '''
        plt.figure(figsize=(8, 8))
        plt.plot(self.xlocation, self.ylocation, '*')
        plt.title('LED array positions')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('x position (mm)')
        plt.ylabel('y position (mm)')
        plt.show()    

    def plot_coherent_imgs(self, arraysize, idx):
        ''' Plotting the set of coherent images'''

        #idx = arraysize**2//2
        coherent_img = self.imSeqLowRes[idx]

        plt.figure(figsize= (8,8))
        plt.imshow(coherent_img, cmap='gray')
        plt.title(f'Coherent Image')
        plt.axis('on')  
        plt.colorbar()
        plt.show()
         
        
       
        
        
