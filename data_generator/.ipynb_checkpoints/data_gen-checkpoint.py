
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
                 NA: float,
                 ground_truth_object :np.ndarray,
                ):
        
        # self.numexp_params = {
        #     'wavelength' : wavelength, # Wavelength of the source
        #     'arraysize': arraysize,    # Number of point sources in the Source Array
        #     'pt_source_gap': pt_source_gap,  # (in mm) gap between two point sources
        #     'source_object_distance' : source_object_distance, # (in mm) distance between the source and the object
        #     'comp_window': comp_window,
        #     'det_psize' : det_psize,
        #     'recon_psize' : recon_psize,
        #     'NA' : NA 
        # }

        self.wavelength = wavelength
        self.arraysize = arraysize
        self.pt_source_gap = pt_source_gap  # (in mm) gap between two point sources
        self.source_object_distance = source_object_distance # (in mm) distance between the source and the object
        self.comp_window = comp_window
        self.det_psize = det_psize
        self.recon_psize = recon_psize
        self.NA = NA
        self.object = ground_truth_object
        
       
        self.Nrow = comp_window
        self.Ncol = comp_window 
        self.objectLength = comp_window * recon_psize

    # def define_object(self, comp_window, recon_psize ):
    #     '''
    #     Importing the objects

    #     Parameters--
    #     comp_window: size of the window over which object is defined

    #     Returns--
    #     object
    #     '''
    #     # number of rows and columns
    #     self.Nrow = comp_window
    #     self.Ncol = comp_window
        
    #     self.objectLength = comp_window*recon_psize
    #     self.object = get_vortex(comp_window, topo_charge = 1)

    # def load_object_with_aperture(self, amp_path, phase_path, comp_window ):
    #     '''
    #     Loading the amplitude and phases of the objects from Images

    #     Parameters--
    #     comp_window: size of the window over which object is defined

    #     Returns--
    #     object: numpy ndarray (complex-field)
    #     with aperture
    #     '''
    #     # number of rows and columns
    #     img1 = Image.open(amp_path)
    #     img2 = Image.open(phase_path)

    #     # number of rows and columns
    #     self.Nrow = comp_window
    #     self.Ncol = comp_window
        
    #     amp = 0.5 + 0.5*preprocess_image(img1, (self.Nrow, self.Ncol))
    #     phi = preprocess_image( img2, (self.Nrow, self.Ncol) )

    #     aperture = circ2d( pixel_size = 1,array_size=comp_window, radius=comp_window//3, center_x=0.0, center_y=0.0)
                               
    #     self.object = amp*np.exp(1j*phi)* aperture

    # def load_object(self, amp_path, phase_path, comp_window):
        
    #     img1 = Image.open(amp_path)
    #     img2 = Image.open(phase_path)

    #     # number of rows and columns
    #     self.Nrow = comp_window
    #     self.Ncol = comp_window
        
    #     amp = 0.5 + 0.5*preprocess_image(img1, (self.Nrow, self.Ncol))
    #     phi = preprocess_image( img2, (self.Nrow, self.Ncol) )
                               
    #     self.object = amp*np.exp(1j*phi)
        

    def define_source(self):
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

        Returns--
        
        '''
        
        # Initialize arrays
        self.xlocation = np.zeros(self.arraysize**2)
        self.ylocation = np.zeros(self.arraysize**2)
        
        # Generate LED positions
        for i in range(self.arraysize):  
            start_idx = self.arraysize * (i)
            end_idx = self.arraysize + self.arraysize * (i)
            
            # Create x positions (same for each row)
            self.xlocation[start_idx:end_idx] = np.arange(-(self.arraysize-1)/2, (self.arraysize-1)/2 + 1) * self.pt_source_gap
            
            # Create y positions (constant for each row)
            self.ylocation[start_idx:end_idx] = ((self.arraysize-1)/2 - (i)) * self.pt_source_gap
        
        self.theta_xz = np.arctan(self.xlocation / self.source_object_distance)
        self.theta_yz = np.arctan(self.ylocation / self.source_object_distance)
        
        kx_relative = -np.sin(np.arctan(self.xlocation / self.source_object_distance))
        ky_relative = -np.sin(np.arctan(self.ylocation / self.source_object_distance))

        self.dkx = 2*np.pi/(self.recon_psize*self.Ncol)
        self.dky = 2*np.pi/(self.recon_psize*self.Nrow)

        # Wavevector magnitude
        self.k0 = 2*np.pi/self.wavelength

        # Wave-vector corresponding to each point source
        self.kx = kx_relative * self.k0
        self.ky = ky_relative * self.k0
        
        # Wave vector co-ordiate - this decides where the Fourier transform of the
        # coherent image would center
        self.kout = np.vstack((self.kx, self.ky)).T


    def _get_pupil(self, pupil_path = None):
        '''
        This method either generates the pupil with the given dimension
        or load the aberrated pupil function 

        parameters--
        (Npupil_row, Npupil_col) : Shape of the pupil
        pupil_path : (str) numpy file name

        Returns--
        Pupil : Lens Pupil
        '''
        # Aberrated Free Pupil Function
        if pupil_path is None:
            pupil = np.ones((self.Npupil_row, self.Npupil_col))
        else:
            pupil = np.load(pupil_path)
            pupil = DS_img(pupil, (self.Npupil_row, self.Npupil_col))

        return pupil

    def _make_kin_grid(self):
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
        kx_in, ky_in = create_2dgrid( self.Npupil_row, self.Npupil_col, self.dky, self.dkx )

        # Flatten 2D grids into 1D arrays
        # Each array contains coordinates for all grid points
        kx_in = kx_in.flatten() 
        ky_in = ky_in.flatten() 
        
        # Stack coordinates into array of [kx, ky] pairs
        # Shape: (Nr*Nc, 2) where each row is [kx_i, ky_i]
        kins = np.vstack((kx_in, ky_in)).T
        
        return kins


    def generate_coherent_imgs(self, pupil_path = None):
        '''
        Generating the diffraction limited images of the object corresponding
        to the tilt direction of the plane wave. Here the tilt of the illuminating 
        beam is realized by the shift of the centre of the Object's Fourier Transform.
        The resolution limit of these images is determined by the pupil aperture
        or the NA of the imaging lens
        '''
        self.objectFT =  np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.object)))
        self.pixBW = int(self.det_psize//self.recon_psize)
        self.Npupil_row = int(self.Nrow//self.pixBW)
        self.Npupil_col = int(self.Ncol//self.pixBW)
        
        # Defining kin vector for pupil
        self.kin_vec = self._make_kin_grid()

        # Get pupil phase and define Pupil Function
        self.pupil_phase = self._get_pupil(pupil_path)
        self.pupil = np.exp(1j*self.pupil_phase)
        
        #Number of coherent images
        self.N_coherent = self.arraysize**2
        # Initializing the sequence of coherent images
        self.imSeqLowRes = np.zeros((self.N_coherent, self.Npupil_row, self.Npupil_col))
        self.imSeqLowRes = self.imSeqLowRes.astype(np.float32)
        
        for tt in range(self.N_coherent):
            kxc = np.round((self.Ncol+1)/2 + self.kx[tt]/self.dkx)
            kyc = np.round((self.Nrow+1)/2 + self.ky[tt]/self.dky)
            kyl = round(kyc- self.Npupil_row//2)
            kyh = round(kyc+ self.Npupil_row//2)
            kxl = round(kxc- self.Npupil_col//2)
            kxh = round(kxc+ self.Npupil_col//2)
            imSeqLowFT = (self.Npupil_col/self.Ncol)**2 * self.objectFT[kyl:kyh, kxl:kxh] * self.pupil
            self.imSeqLowRes[tt] = np.abs( np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT))) )

    def generate_coherent_imgs_BL(self, DS_fac = 2, pupil_path = None):
        '''
        Generating the diffraction limited images of the object corresponding
        to the tilt direction of the plane wave. Here the tilt of the illuminating 
        beam is realized by the shift of the centre of the Object's Fourier Transform.
        The resolution limit of these images is determined by the pupil aperture
        or the NA of the imaging lens

        Here the object Spectrum is BandLimited

        DS_fac (int): down sampling factor
                        ratio of the Pupil aperture and object's band corresponding to
                        the downsampled version of the object
        '''
        self.objectFT_pre =  np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.object)))
        
        self.pixBW = int(self.det_psize//self.recon_psize)
        self.Npupil_row = int(self.Nrow//self.pixBW)
        self.Npupil_col = int(self.Ncol//self.pixBW)
        
        # Defining kin vector for pupil
        self.kin_vec = self._make_kin_grid()

        # Get pupil phase and define Pupil Function
        self.pupil_phase = self._get_pupil(pupil_path)
        self.pupil = np.exp(1j*self.pupil_phase)

        #Band limited Object
        self.objectFT = np.zeros_like(self.objectFT_pre)

        self.Nds_row =  int( self.Npupil_row // DS_fac )
        self.Nds_col =  int( self.Npupil_col // DS_fac )

        self.lr_psize = self.det_psize * DS_fac
        rl = self.Nrow//2 - self.Nds_row//2
        rh = self.Nrow//2 + self.Nds_row//2
        cl = self.Ncol//2 - self.Nds_col//2
        ch = self.Ncol//2 + self.Nds_col//2
        self.objectFT[rl : rh, cl : ch] = self.objectFT_pre[rl : rh, cl : ch]
        
        
        #Number of coherent images
        self.N_coherent = self.arraysize**2
        # Initializing the sequence of coherent images
        self.imSeqLowRes = np.zeros((self.N_coherent, self.Nds_row, self.Nds_col))
        self.imSeqLowRes = self.imSeqLowRes.astype(np.float32)
        self.imSeqLowFT = np.zeros((self.N_coherent, self.Nds_row, self.Nds_col)).astype(dtype = complex)
        
        for tt in range(self.N_coherent):
            kxc = np.round((self.Ncol+1)/2 + self.kx[tt]/self.dkx)
            kyc = np.round((self.Nrow+1)/2 + self.ky[tt]/self.dky)
            kyl = round(kyc- self.Npupil_row//2)
            kyh = round(kyc+ self.Npupil_row//2)
            kxl = round(kxc- self.Npupil_col//2)
            kxh = round(kxc+ self.Npupil_col//2)
            imSeqLowFT = (self.Npupil_col/self.Ncol)**2 * self.objectFT[kyl:kyh, kxl:kxh] * self.pupil
            self.imSeqLowFT[tt] = self._cutout_zeros( imSeqLowFT )
            self.imSeqLowRes[tt] = np.abs( np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.imSeqLowFT[tt]))) )

    def _cutout_zeros(self, arr):
        mask_it = arr != 0
        row_mask, col_mask = np.where(mask_it)
        row_min, row_max = row_mask.min(), row_mask.max()
        col_min, col_max = col_mask.min(), col_mask.max()
        return arr[row_min: row_max + 1, col_min : col_max + 1]

    def generate_coherent_imgs_RSmodel(self, arraysize, recon_psize):
        '''
        Generating the diffraction limited images of the object corresponding
        to the tilt direction of the plane wave. Here the tilted plane wave illuminates
        the object.

        Parameters-
        arraysize: size of the source arrays, as each image of the object is corresponding 
        to one particular source
        recon_psize : Pixel size at the object plane (This is what we aim to recover 
        in Fourier Ptychography)

        Returns-
        imSeqLowRes_data: diffraction limited coherent images (resolution limited by the
        pupil of the imaging lens)
        
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
            this_obj_FT_filt = this_obj_FT[rl : rh, cl : ch] * self.pupil
            this_im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(this_obj_FT_filt)))
            self.imSeqLowRes_data[tt] = np.abs(this_im)   

    def downsample_coherent_imgs(self, det_psize, DS_fac = 2):
        '''
        This method further downsamples the coherent images. Downsampling the coherent images
        in Conventional Transmission microscope is analogous to the scanning the object at the 
        larger step size in the scanning transmission microscope.

        Parameters:
        DS_fac: (int) downsampling factor 

        Returns:
        Nds_row : rows in downsampled coherent image
        Nds_col : columns in downsampled coherent image
        DS_imSeqLowRes : downsampled coherent image sequence of shape (arraysize^2, Nds_row, Nds_col)
        '''
        self.Nds_row =  self.Npupil_row // DS_fac 
        self.Nds_col =  self.Npupil_col // DS_fac 

        self.lr_psize = det_psize * DS_fac

        DS_imSeqLowRes = []

        for tt in range(self.N_coherent):
            this_DS_imSeqLowRes = DS_img(self.imSeqLowRes[tt], (self.Nds_row, self.Nds_col))
            DS_imSeqLowRes.append(this_DS_imSeqLowRes)
            
        self.DS_imSeqLowRes = np.array(DS_imSeqLowRes)
        print(f"coherent images have been downsampled to the shape {self.N_coherent,self.Nds_row, self.Nds_col }")


    def make_sequenced_images(self, arraysize, img_seq):
        '''
        
        Reorders coherent images and their corresponding k-vectors according to a 
        spiral/radial sequence from center to edge of a square array.
        
        This method rearranges the coherent image sequence to follow a specific 
        reconstruction order that typically starts from the center of the illumination 
        pattern and spirals outward to the edges. This ordering is often used in 
        Fourier ptychography where the reconstruction quality benefits from processing 
        center (low spatial frequency) images before edge (high spatial frequency) images.
        
        Parameters:
        -----------
        arraysize : int
            Side length of the square illumination array (e.g., if arraysize=5, 
            there are 5x5=25 total illumination angles/images)
            
        Side Effects:
        -------------
        Creates/modifies:
            self.DSimseq : numpy.ndarray
                Reordered sequence of downsampled coherent images
            self.kout_seq : numpy.ndarray  
                Reordered sequence of corresponding k-vectors (illumination directions)
        
        Dependencies:
        -------------
        Requires:
            - gseq_square(arraysize) function to generate the sequence indices
            - self.DS_imSeqLowRes : original coherent image sequence
            - self.kout : original k-vector sequence
            - self.N_coherent : total number of coherent images

        '''
        sequence_idx =  gseq_square(arraysize) 

        if len(sequence_idx) != self.N_coherent:
            raise ValueError(f"Sequence length {len(sequence_idx)} doesn't match N_coherent {self.N_coherent}")
    
        if np.any(sequence_idx >= (self.N_coherent+1)) or np.any(sequence_idx < 0):
            raise ValueError("Sequence indices out of bounds")
        
        self.DSimseq = np.empty_like(img_seq)
        self.kout_seq = np.empty_like(self.kout)
        
        for tt in range(self.N_coherent):
            seq_iter = int(sequence_idx[tt]) - 1
            self.DSimseq[tt] = img_seq[seq_iter]
            self.kout_seq[tt]= self.kout[seq_iter]
            

    

    # -------------------Plotting---------------------------------------


    def plot_object(self):
        ''' Plot amplitude and the Phase of the object'''
        
        amp = np.abs(self.object)
        phase = np.angle(self.object)

        fig, axes = plt.subplots(1,2, figsize= (12,5))
        object_amp = axes[0].imshow(amp)
        axes[0].set_title(f'Object Amplitude')
        axes[0].axis('on')  
        plt.colorbar(object_amp, ax=axes[0], fraction=0.046, pad=0.04)
        
        object_phase= axes[1].imshow(phase)
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
        plt.imshow(coherent_img, cmap='magma')
        plt.title(f'Coherent Image')
        plt.axis('on')  
        plt.colorbar()
        plt.show()
         
        
       
        
        
