import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
import h5py
import os
from .utils_pr import *
from .plotting import plot_images_side_by_side, update_live_plot, initialize_live_plot
from scipy.ndimage import fourier_shift
from PIL import Image
#from ..data_fs import * #downsample_array, upsample_images, pad_to_double

from skimage.restoration import unwrap_phase
from .utils_zernike import *


class difference_map:

    def __init__(self, images, pupil_func: str, kout_vec, ks_pupil, 
                 lr_psize, object_est=None, object_estFT=None,
                 num_jobs=4):

        '''
        images : coherent Images
        pupil_func : pupil function
        '''
        
        self.images = images # Coherent images
        self.num_images = self.images.shape[0]
        
        self.pupil_func = pupil_func
        self.kout_vec = kout_vec
        self.ks_pupil = ks_pupil
        self.lr_psize = lr_psize
        
        self.object_est = object_est   #has shape different from the pupil_function
        self.object_estFT = object_estFT  
        
        self.losses = []
        self.iters_passed = 0

        self.zoom_factor = 1
        self.num_jobs = num_jobs

        self.Npupil_rows, self.Npupil_cols = self.pupil_func.shape
        self.Ncoherent_imgs, self.Nr1, self.Nc1 = self.images.shape


    @time_it
    def prepare(self, **kwargs):
        
        print("Preparing")
        
        if 'zoom_factor' in kwargs:
            self.zoom_factor = kwargs['zoom_factor']

        if 'extend' in kwargs:
            extend = kwargs['extend']
        else:
            extend = None
            
        self._prep_images()
        
        self.kout_vec = np.array(self.kout_vec)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.ks_pupil, self.lr_psize, extend = extend)

        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        omegas = calc_obj_freq_bandwidth(self.lr_psize)
        self.omega_obj_x, self.omega_obj_y = omegas

        self._load_pupil()
        
        self._initiate_recons_images()

    def _prep_images(self):
        
        self.images = np.abs(np.array(self.images))

        if self.zoom_factor > 1:
            self.images = upsample_images(self.images, self.zoom_factor, n_jobs = self.num_jobs)
            self.lr_psize = self.lr_psize / self.zoom_factor 
            
        self.images = np.array(self.images)
        
    def _initiate_recons_images(self):
        if self.object_est is None and self.object_estFT is None: 
            self.object_est = np.ones_like(self.images[0]).astype(complex)
            self.object_estFT = np.ones_like(self.images[0]).astype(complex)
        
        elif self.object_est is not None:
            self.object_estFT = fftshift(fft2(self.object_est))
            
        elif self.object_estFT is not None:
            self.object_est = ifft2(ifftshift(self.object_estFT))
            
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.object_est.shape
        
       
    def _load_pupil(self):

        '''
        takes a complex-valued generalized pupil function
        '''
        
        dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        dims = make_dims_even(dims)

        full_array = np.zeros(dims).astype(complex)
        
        if isinstance(self.pupil_func, str):
            pupil_func = np.load(self.pupil_func)
        elif isinstance(self.pupil_func, np.ndarray):
            pupil_func = self.pupil_func
        else:
            pupil_func = np.ones(dims)
        
        # Get the scaling factors for each dimension
        scale_x = dims[0] / pupil_func.shape[0] / 2
        scale_y = dims[1] / pupil_func.shape[1] / 2 

        # Scale the pupil phase array to match the required pupil dimensions
        scaled_pupil_func = zoom(pupil_func, (scale_x, scale_y))

        # Calculate center indices
        N, M = dims[0]//2, dims[1]//2
        
        start_x, start_y = (dims[0] - N) // 2, (dims[1] - M) // 2
        end_x, end_y = start_x + N, start_y + M
    
        # Set central region to ones
        full_array[start_x:end_x, start_y:end_y] = scaled_pupil_func
        
        self.pupil_func = full_array

    def _upsample_coherent_images(self):
        Nx, Ny = self.pupil_func.shape
        upsamp_img = np.zeros((self.Ncoherent_imgs, Nx, Ny))
        
        for k in range(self.Ncoherent_imgs):
            image = self.images[k]
            image_FT = self._fft2(image)
            new_image_FT = np.zeros((Nx, Ny), dtype=complex)
            new_image_FT[Nx//2 - self.Nr1//2 : Nx//2 + self.Nr1//2, Ny//2 -self.Nc1//2: Ny//2 + self.Nc1//2] = image_FT
            upsamp_img[k] = self._ifft2(new_image_FT)
    
        return upsamp_img

    def iterate(self, iterations, live_plot = True):

        
        #zero_padding the object guess FT to have the same shape as of pupil_func
        self.object_estFT_up = pad_array_flexible(self.object_estFT, target_shape = self.pupil_func.shape) 
        self.object_est_up = np.fft.ifft2(self.object_estFT_up)
        #upsample the coherent images to have the same shape as of pupil
        self.up_imSeqLowRes = self._upsample_coherent_images()   

        if live_plot:
            fig, axes, img_amp, img_phase, img_pupil_amp, img_pupil_phase = self._initialize_live_plot()

        for it in range(iterations):
            self.object_est_up = np.fft.ifft2(self.object_estFT_up)
            
            # update pupil
            self.pupil_func = self._pupil_update()
            
            #update object FT upsampled
            self.object_estFT_up = self._objectFT_update()
            
            PSI_update = self.difference_map_engine()
            self.PSI_FT = PSI_update

            if live_plot and it % 5 == 0:
                self._update_live_plot(img_amp, img_phase, img_pupil_amp, img_pupil_phase, fig, axes)


    def _pupil_update(self):
        '''
        Updating the pupil function
        '''
        denom = np.zeros_like(self.pupil_func)
        numer = np.zeros_like(self.pupil_func)
        PSI_FT = []
        objectFT = []
        for i, (image, kx_iter, ky_iter) in enumerate(tqdm(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1]), 
                                                               desc="Processing", total=len(self.images), unit="images")):
            this_objectFT = self._shift_objectFT(kx_iter, ky_iter)
            denom += np.abs(this_objectFT)**2
            this_PSI = self._get_exitFT(self.pupil_func, this_objectFT)
            numer += np.conjugate(this_objectFT) * this_PSI
            
            objectFT.append(this_objectFT)
            PSI_FT.append(this_PSI)
            
        pupil_func_update = numer/(denom+10**-15)   # use self.pupil_func
        self.PSI_FT = np.array(PSI_FT)
        self.objectFT_shifted = np.array(objectFT)
        return pupil_func_update

    def _shift_objectFT(self, kx_iter, ky_iter):
        '''
        This method shifts the object's FT estimate to the (kx_cidx, ky_cidx) location
        '''
        object_patch = np.zeros_like(self.pupil_func)
        
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)

        rl = self.Npupil_rows//2 - self.Nr1//2
        rh = self.Npupil_rows//2 + self.Nr1//2
        cl = self.Npupil_cols//2 - self.Nc1//2
        ch = self.Npupil_cols//2 + self.Nc1//2

        object_patch[kx_lidx:kx_hidx, ky_lidx:ky_hidx] = self.object_estFT_up[rl:rh, cl:ch]  
        return object_patch

    def _get_exitFT(self, pupil, objectFT):
        return pupil*objectFT

    ###--------
    def _objectFT_update(self):
        Nx, Ny = self.pupil_func.shape
        denom = np.zeros_like(self.pupil_func)
        numer = np.zeros_like(self.pupil_func)
        for i, (image, kx_iter, ky_iter) in enumerate(tqdm(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1]), 
                                                               desc="Processing", total=len(self.images), unit="images")):
            this_pupil = self._get_pupil_patch_centred(kx_iter, ky_iter)
            denom += np.abs(this_pupil)**2
            this_PSI = self._get_exitFT(this_pupil, self.object_estFT_up)
            numer += np.conjugate(this_pupil) * this_PSI #self.get_exitFT_centred(this_pupil) 
        objectFT_update = numer/(denom + 10**-15)
        return objectFT_update

    def _get_pupil_patch_centred(self, kx_iter, ky_iter):
        '''
        This method get Pupil patch to the centre
        '''
        pupil_func_patch = np.zeros_like(self.pupil_func).astype(complex) 
        
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)

        rl = self.Npupil_rows//2 - self.Nr1//2
        rh = self.Npupil_rows//2 + self.Nr1//2
        cl = self.Npupil_cols//2 - self.Nc1//2
        ch = self.Npupil_cols//2 + self.Nc1//2

        pupil_func_patch[rl : rh, cl:ch] = self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]

        return pupil_func_patch


    def get_exitFT_centred(self, pupil):
        '''
        This method initializes the exit field in Fourier domain
        objectFT: (128, 128)
        '''
        rl = self.Npupil_rows//2 - self.Nr1//2
        rh = self.Npupil_rows//2 + self.Nr1//2
        cl = self.Npupil_cols//2 - self.Nc1//2
        ch = self.Npupil_cols//2 + self.Nc1//2

        # Bring the exit patch to the centre
        exit_FT = np.zeros_like(self.pupil_func)
        exit_FT[rl:rh, cl:ch] =  pupil * self.object_estFT_up
        self.exitFT_centred = exit_FT
        return exit_FT

    def difference_map_engine(self):
        '''
        Write description of the algorithm
        '''
    
        PSI_n = np.zeros_like(self.PSI_FT).astype(complex)
        for i, (image, kx_iter, ky_iter) in enumerate(tqdm(zip(self.up_imSeqLowRes, self.kout_vec[:, 0], self.kout_vec[:, 1]), 
                                                               desc="Processing", total=len(self.images), unit="images")):
            this_objectFT = self._shift_objectFT(kx_iter, ky_iter)
            Ps_psi = self._get_exitFT(self.pupil_func, this_objectFT)
            Rs_psi = 2*Ps_psi - self.PSI_FT[i]
            Pm_rpsi = self.project_data(image, Rs_psi)
            PSI_n[i] = self.PSI_FT[i] + Pm_rpsi - Ps_psi 
        return PSI_n

    def project_data(self, image, arr_FT):
        '''
        measurement projection
        '''
        psi =  self._ifft2(arr_FT) 
        psi_new = image*np.exp(1j*np.angle(psi))
        PSI_PM  =  self._fft2(psi_new)  
        return PSI_PM

    def _fft2(self, arr):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))

    def _ifft2(self, arrFT):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arrFT)))

# Plotting
    def _initialize_live_plot(self):
        """
        Initializes the live plot with two subplots: one for amplitude and one for phase.
        
        Returns:
            fig, ax: Matplotlib figure and axes.
            img_amp, img_phase: Image objects for real-time updates.
        """
        
        # Initialize empty images
        fig, axes = plt.subplots(2, 2, figsize=(5, 8))
                
        # Initialize the plots with the initial image
        img_amp = axes[0,0].imshow( np.abs(self.object_est_up), cmap='viridis')
        axes[0,0].set_title("Object Amplitude")
        cbar_amp = plt.colorbar(img_amp, ax=axes[0,0])
        
        img_phase = axes[0,1].imshow(np.angle(self.object_est_up), cmap='viridis')
        axes[0,1].set_title("Object Phase")
        img_phase.set_clim(-np.pi, np.pi)
        cbar_phase = plt.colorbar(img_phase, ax=axes[0,1])
        
        img_pupil_amp = axes[1,0].imshow(np.abs(self.pupil_func), cmap='viridis')
        axes[1,0].set_title("Fourier Amplitude")
        cbar_fourier = plt.colorbar(img_pupil_amp, ax=axes[1,0], shrink = 0.5)
        
        img_pupil_phase = axes[1,1].imshow(np.angle(self.pupil_func), cmap='viridis')
        axes[1,0].set_title("pupil phase")
        img_pupil_phase.set_clim(-np.pi, np.pi)  # Set proper phase limits
        cbar_fourier = plt.colorbar(img_pupil_phase, ax=axes[1,1], shrink = 0.5)
        
        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show()
        
        return fig, axes, img_amp, img_phase, img_pupil_amp, img_pupil_phase
        
    def _update_live_plot(self, img_amp, img_phase, img_pupil_amp, img_pupil_phase, fig, axes):
        """
        Updates the live plot with new amplitude and phase images.
        
        Args:
            img_amp: Matplotlib image object for amplitude.
            img_phase: Matplotlib image object for phase.
            hr_obj_image: The complex object image to be plotted.
        """
        amplitude_obj = np.abs(self.object_est_up)
        phase_obj = np.angle(self.object_est_up)
        
        amplitude_pupil = np.abs(self.pupil_func)
        pupil_phi = np.angle(self.pupil_func)
        
        img_amp.set_data(amplitude_obj)  # Normalize for visibility
        img_phase.set_data(phase_obj)
        img_pupil_amp.set_data(amplitude_pupil)
        img_pupil_phase.set_data(pupil_phi)
        
        
        # amp_mean = np.mean(amplitude_obj)
        # vmin = max(amp_mean + 2 * amp_mean, 0)
        # vmax = amp_mean + 10 * amp_mean
        # img_amp.set_clim(vmin, vmax)
        
        # ft_mean = np.mean(amplitude_ft)
        # vmin = ft_mean + 2 * ft_mean
        # vmax = ft_mean + 10 * ft_mean
        # fourier_amp.set_clim(vmin, vmax)
            
        clear_output(wait=True)
        display(fig)
        fig.canvas.flush_events()

    
            

    

    
        




    
        
        
        







    