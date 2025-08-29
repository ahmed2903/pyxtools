import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
from functools import partial
import jax.tree_util as jtu

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
import h5py

from .utils_pr import *


class classical_optimization:
    
    def __init__(self, images, pupil_func, k_out, k_in, lr_psize, object_est, object_est_FT, object_GT ):
        '''
        Initializing the class attributes
        coh_imgs : (numpy array of shape (Num_coh, )
        '''
    
        self.images = images
        self.pupil_func = pupil_func
        self.pupil_kins = k_in
        self.k_out = k_out
        self.lr_psize = lr_psize
        self.object_est = object_est
        self.object_est_FT = object_est_FT
        self.object_GT = object_GT
        
        self.Num_coh = images.shape[0]

        self.losses_tv = []
        self.losses_data_fidelity = []
        self.iters_passed = 0
    
    def _FT2(self, arr):
        '''
        Computating 2D FFT using jax 

        Parameter:
        arr (numpy ndarray): array whose 2D Fourier transform is to be calculated

        Returns:
        arr_FT(numpy ndarray): complex-valued 2D Fourier transform of arr
        '''
        arr_FT = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(arr)))
        return arr_FT
    
    def _IFT2(self, arr_FT):
        '''
        Computating 2D IFFT using jax 

        Parameter:
        arr_FT (numpy ndarray): array whose 2D Inverse Fourier transform is to be calculated

        Returns:
        arr(numpy ndarray): complex-valued 2D Inverse Fourier transform of arr_FT
        '''
        arr = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(arr_FT)))
        return arr
    
    def _prep_images(self):
        """
        Converts the input images to a NumPy array and ensures they are stored as a PyTorch tensor.
        
        This method:
        - Converts `self.images` into a NumPy array if it is not already one.
        - Extracts image dimensions and assigns them to `self.image_dims`.
        - Ensures `self.images` is a PyTorch tensor of type float64 and moves it to the specified device.
        
        """
        self.images = jnp.array(self.images)
        self.image_dims = self.images[0].shape
        self.nx_lr, self.ny_lr = self.image_dims
    
    def prepare(self, extend_pupil = None):
    
        """
        Prepares the input parameters that are given to the optimization routine
        
        This method:
        - Prepares images and converts kout vectors to NumPy arrays.
        - Computes spatial frequency bounds and step sizes.
        - Determines the pupil dimensions, ensuring they are even.
        - Calculates the object frequency bandwidth.
        
        Args:
            double_pupil (str, optional): Mode of pupil extension. "by_bandwidth", "double", None. Default is None
        """
        self._prep_images()
        self.kin_vec = jnp.array(self.pupil_kins)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.pupil_kins, lr_psize = self.lr_psize, extend = extend_pupil)
        
        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        self.pupil_dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        self.pupil_dims = make_dims_even(self.pupil_dims)
        self.omega_obj_x, self.omega_obj_y  = calc_obj_freq_bandwidth(self.lr_psize)  
    
      
    def get_total_error(self, obj_est):
        # List to store all the data mismatch terms corresponding to each image
        data_mismatch_i = []
        obj_est_FT = self._FT2(obj_est)
        for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.k_out[:, 0], self.k_out[:, 1])): 
            pupil_patch = self._update_spectrum(kx_iter, ky_iter)
            data_fidelity_i = self._LSE(obj_est_FT, pupil_patch, image)
            data_mismatch_i.append(data_fidelity_i)
            
        return jnp.sum(jnp.array(data_mismatch_i))
    
    def _update_spectrum(self, kx_iter, ky_iter):
        """
        Fourier domain update
        """
        self.prepare()
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)
        
        pupil_func_patch = self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]
        return pupil_func_patch
    
    def _LSE(self, obj_ft, pupil, image):
        """
        Least Squares Error calculation 
        
        Args:
            obj_ft: Object in Fourier domain
            pupil: Pupil function 
            image: Measured image 
        
        Returns:
            error_i: Frobenius norm of the data mismatch 
        """
        
        exit_FT = obj_ft * pupil
        
        # Transform back to spatial domain
        exit = self._IFT2(exit_FT)
        
        # Calculate data mismatch (difference between measured and calculated intensities)
        data_mismatch = jnp.abs(image - jnp.abs(exit))
        
        # Return Frobenius norm using jAX        
        return jnp.linalg.norm(data_mismatch, ord='fro')
    
    def optimize_object(self, iterations_num, learning_rate = 1e-2, live_plot = False, save_gif = False):
        """
        Optimize obj_est using Adam optimizer to minimize total error
        
        Args:
            initial_obj_est: Initial guess for the object estimate
            iterations_num: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            object_est, object_est_FT: Optimized object estimate and its Fourier transform 
            loss_history: History of loss values with Optimization
            
        """
        if live_plot:
            fig, ax, img_amp, img_phase, fourier_amp, loss_im, loss_tv0, axes = self._initialize_live_plot()
             
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)

        params = self.object_est # Object in the real space is a parameter
        opt_state = self.optimizer.init(params)

        #step_size = []
        for iter_loop in range(iterations_num):
            self.iters_passed += 1
            params, opt_state, loss_tv, loss_data_error = self.MGD_train_step(params, opt_state)
            self.losses_tv.append(loss_tv)
            self.losses_data_fidelity.append(loss_data_error)
        
            self.obj_est = params
            self.object_est_FT = self._FT2(self.object_est)

            if live_plot:
                self._update_live_plot(img_amp, img_phase, fourier_amp, loss_im, loss_tv0, fig, iter_loop, axes)


    
    ## Functions for Optimization Stuff
    @partial(jit, static_argnums =0)
    def train_step(self, params, opt_state):
        '''
        This Method only optimizes for the Error loss Function
        '''
        loss, grads = value_and_grad(self.get_total_error)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    def _Total_Variation(self, img):
        ux = img - jnp.roll(img, 1, axis=0)
        uy = img - jnp.roll(img, 1, axis=1)
        ds = jnp.sqrt(jnp.abs(ux)**2 + jnp.abs(uy)**2 + 1e-20)
        return jnp.sum(ds)

    def tree_norm(self, tree):
        return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jtu.tree_leaves(tree)))

    # Normalize a pytree
    def normalize_tree(self, tree, eps = 1e-12):
        norm = self.tree_norm(tree)
        return jtu.tree_map(lambda x: x / (norm + eps), tree)

    def MGD_train_step(self, params, opt_state):
        '''
        Optimization with MGD
        '''
        loss_err, grad1 = value_and_grad(self.get_total_error)(params)
        loss_TV, grad2 = value_and_grad(self._Total_Variation)(params)        
        grad1_norm = self.normalize_tree(grad1)
        grad2_norm = self.normalize_tree(grad2)

        # Average the normalized gradients (element-wise)
        grad_MGD = jtu.tree_map(lambda x, y: (x + y) / 2, grad1_norm, grad2_norm)
        
        updates, opt_state = self.optimizer.update(grad_MGD, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_TV, loss_err

    def iterate(self, iterations_num, learning_rate = 1e-2, live_plot = False, save_gif = False):
        """
        Optimize obj_est using Adam optimizer to minimize total error
        
        Args:
            initial_obj_est: Initial guess for the object estimate
            iterations_num: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            object_est, object_est_FT: Optimized object estimate and its Fourier transform 
            loss_history: History of loss values with Optimization
            
        """
        if live_plot:
            fig, ax, img_amp, img_phase, fourier_amp, loss_im, loss_tv0, axes = self._initialize_live_plot()
             
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)

        params = self.object_est # Object in the real space is a parameter
        opt_state = self.optimizer.init(params)

        #step_size = []
        for iter_loop in range(iterations_num):
            self.iters_passed += 1
            params, opt_state, loss_data_error = self.train_step(params, opt_state)
            self.losses_tv.append(0)
            self.losses_data_fidelity.append(loss_data_error)
        
            self.obj_est = params
            self.object_est_FT = self._FT2(self.object_est)

            if live_plot:
                self._update_live_plot(img_amp, img_phase, fourier_amp, loss_im, loss_tv0, fig, iter_loop, axes)


#########################################################################
    
    def plot_Object_Recovery(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Reconstructed Objects')
    
        im1 = ax1.imshow(jnp.abs(self.obj_est))
        ax1.set_title('Amplitude')
        fig.colorbar(im1, ax=ax1)
    
        im2 = ax2.imshow(jnp.angle(self.obj_est))
        ax2.set_title('Phase')
        fig.colorbar(im2, ax=ax2)
    
        plt.tight_layout()
        plt.show()
                               
    
    def _initialize_live_plot(self):
        """
        Initializes the live plot with two subplots: one for amplitude and one for phase.
        
        Returns:
            fig, ax: Matplotlib figure and axes.
            img_amp, img_phase: Image objects for real-time updates.
        """
        
        # Initialize empty images
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                
        # Initialize the plots with the initial image
        img_amp = axes[0,0].imshow(np.abs(self.object_est), vmin =.2, vmax = 1, cmap='viridis')
        axes[0,0].set_title("Object Amplitude")
        cbar_amp = plt.colorbar(img_amp, ax=axes[0,0])
        
        img_phase = axes[0,1].imshow(np.angle(self.object_est), cmap='viridis', vmin = -0.1, vmax = 0.5)
        axes[0,1].set_title("Object Phase")
        #img_phase.set_clim(-np.pi, np.pi)
        cbar_phase = plt.colorbar(img_phase, ax=axes[0,1])
        
        
        fourier_amp = axes[1,0].imshow(np.abs(self.object_est_FT), cmap='viridis')
        axes[1,0].set_title("Fourier Amplitude")
        cbar_fourier = plt.colorbar(fourier_amp, ax=axes[1,0])
        
        loss_im, = axes[1,1].plot([],[], 'b-', label = 'TV loss')
        loss_tv0, = axes[1,1].plot([], [], 'r-', label = 'Ground Truth TV')
        axes[1,1].set_xlabel("iteration")
        axes[1,1].set_ylabel("TV")
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show()
        
        return fig, axes, img_amp, img_phase, fourier_amp, loss_im, loss_tv0, axes

    def _update_live_plot(self, img_amp, img_phase, fourier_amp, loss_im, loss_tv0, fig, it, axes):
        """
        Updates the live plot with new amplitude and phase images.
        
        Args:
            img_amp: Matplotlib image object for amplitude.
            img_phase: Matplotlib image object for phase.
            hr_obj_image: The complex object image to be plotted.
        """
        amplitude_obj = np.abs(self.object_est)
        phase_obj = np.angle(self.object_est)
        amplitude_ft = np.abs(self.object_est_FT)
        
        img_amp.set_data(amplitude_obj)  # Normalize for visibility
        img_phase.set_data(phase_obj)
        fourier_amp.set_data(amplitude_ft)

        x_data = range(self.iters_passed)
        loss_im.set_xdata(x_data)
        loss_im.set_ydata(self.losses_data_fidelity)

        loss_tv0.set_xdata(x_data)
        loss_tv0.set_ydata(self._Total_Variation(self.object_GT))
        
        axes[1,1].set_title(f"Iteration: {it}, loss: {self.losses_tv[it]:.2f}", fontsize=12)
        axes[1,1].relim()
        axes[1,1].autoscale_view()
        if it>1:
            axes[1,1].set_yscale('log')
        
        amp_mean = np.mean(amplitude_obj)
        vmin = max(amp_mean + 2 * amp_mean, 0)
        vmax = amp_mean + 10 * amp_mean
        img_amp.set_clim(vmin, vmax)
        
        ft_mean = np.mean(amplitude_ft)
        vmin = ft_mean + 2 * ft_mean
        vmax = ft_mean + 10 * ft_mean
        fourier_amp.set_clim(vmin, vmax)
            
        clear_output(wait=True)
        display(fig)
        fig.canvas.flush_events()
        
        
#class optimization_mgd(classical_optimization):
        

        
        
        
