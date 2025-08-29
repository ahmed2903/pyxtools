import torch
import torch.fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
import h5py

from .utils_pr import *



class optimization_MGD:
    
    def __init__(self, images, pupil_func, k_out, k_in, lr_psize, object_est_FT, object_est, object_GT):
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
        self.Num_coh = images.shape[0]
        self.object_GT = object_GT


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
        self.kin_vec = np.array(self.pupil_kins)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.pupil_kins, lr_psize = self.lr_psize, extend = extend_pupil)
        
        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        self.pupil_dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        self.pupil_dims = make_dims_even(self.pupil_dims)
        self.omega_obj_x, self.omega_obj_y  = calc_obj_freq_bandwidth(self.lr_psize)  
        self._load_pupil()
    

    def _prep_images(self):
        """
        Converts the input images to a NumPy array and ensures they are stored as a PyTorch tensor.

        This method:
        - Converts `self.images` into a NumPy array if it is not already one.
        - Extracts image dimensions and assigns them to `self.image_dims`.
        - Ensures `self.images` is a PyTorch tensor of type float64 and moves it to the specified device.

        """
        self.images = np.array(self.images)
        self.image_dims = self.images[0].shape
        self.nx_lr, self.ny_lr = self.image_dims
        
        # Ensure target_image is a tensor and has the same size as reconstructed_image
        if not isinstance(self.images, torch.Tensor):
            self.images = torch.tensor(self.images, dtype=torch.float64)  
            
    #!FIX the FPM_simulation code accordingly
    def _load_pupil(self):
        '''
        This method loads the pupil function guess
        '''
        
        dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        dims = make_dims_even(dims)

        full_array = np.zeros(dims)
        
        if isinstance(self.pupil_func, str):
            phase = np.load(self.pupil_func)
        elif isinstance(self.pupil_func, np.ndarray):
            phase = self.pupil_func #Fix me
        else:
            phase = np.zeros(dims)
        
        # Get the scaling factors for each dimension
        scale_x = dims[0] / phase.shape[0] / 2
        scale_y = dims[1] / phase.shape[1] / 2 

        # Scale the pupil phase array to match the required pupil dimensions
        scaled_pupil_phase = zoom(phase, (scale_x, scale_y))

        # Calculate center indices
        N, M = dims[0]//2, dims[1]//2
        
        start_x, start_y = (dims[0] - N) // 2, (dims[1] - M) // 2
        end_x, end_y = start_x + N, start_y + M
    
        # Set central region to ones
        full_array[start_x:end_x, start_y:end_y] = scaled_pupil_phase
        
        self.pupil_func = np.exp(1j*full_array)
        
        
    def get_total_error(self, obj_est):
        # List to store all the data mismatch terms corresponding to each image
        data_mismatch_i = []
        
        # Ensure obj_est is a torch tensor
        if not isinstance(obj_est, torch.Tensor):
            obj_est = torch.tensor(obj_est, dtype=torch.complex64)
    
        for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.k_out[:, 0], self.k_out[:, 1])):
            
            pupil_patch = self._update_spectrum(kx_iter, ky_iter)
            obj_est_FT = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(obj_est)))
            data_fidelity_i = self._LSE(obj_est_FT, pupil_patch, image)
            data_mismatch_i.append(data_fidelity_i)
            
         # Convert list to tensor and sum
        data_mismatch_i = torch.stack(data_mismatch_i)
        total_data_mismatch = torch.sum(data_mismatch_i)
        return total_data_mismatch

    def _update_spectrum(self, kx_iter, ky_iter):
        
        """
        Fourier domain update
        """
        
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
        Least Squares Error calculation using PyTorch operations
        
        Args:
            obj_ft: Object in Fourier domain (torch.Tensor)
            pupil: Pupil function (torch.Tensor) 
            image: Measured image (torch.Tensor)
        
        Returns:
            error_i: Frobenius norm of the data mismatch (torch.Tensor)
        """
        
        
        # Ensure all inputs are torch tensors
        if not isinstance(obj_ft, torch.Tensor):
            obj_ft = torch.tensor(obj_ft, dtype=torch.complex64)
        if not isinstance(pupil, torch.Tensor):
            pupil = torch.tensor(pupil, dtype=torch.complex64)
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        
        exit_FT = obj_ft * pupil
        
        # Transform back to spatial domain
        exit = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(exit_FT)))
        
        # Calculate data mismatch (difference between measured and calculated intensities)
        data_mismatch = torch.abs(image - torch.abs(exit))
        
        # Calculate Frobenius norm using PyTorch
        error_i = torch.linalg.norm(data_mismatch, ord='fro')
        
        return error_i
        
    # REVISE the comment

    def tv_penalty(self, image):
        """
        Compute the Total Variation (TV) regularization term.

        This method computes the Total Variation regularization for a given image. The TV regularization term 
        encourages smoothness by penalizing large gradients between adjacent pixels. It is often used in image 
        reconstruction to avoid overfitting and promote image smoothness.

        Args:
            image (torch.Tensor): A tensor representing the image or batch of images, with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The TV regularization term for each image in the batch (shape: (batch_size,)).
        """
        # Compute gradients in x and y directions
        gradient_x = torch.abs(torch.roll(image, shifts=-1, dims=1) - image)
        #gradient_x = gradient_x[:, :-1]  # Remove the last column 
        gradient_y = torch.abs(torch.roll(image, shifts=-1, dims=0) - image)
        #gradient_y = gradient_y[:-1, :]  # Remove the last row 
    
        # Compute the TV regularization term for each image
        tv = torch.sum(torch.sqrt(gradient_x**2 + gradient_y**2))
        
        return tv


    def _normalize_gradients(self,gradients):
        """Normalize gradients to have unit L2 norm (to balance their contributions)."""
        
        non_none_grads = [g for g in gradients if g is not None]
        if not non_none_grads:
            return gradients

        
        # Compute the total norm of all gradients (treat as one vector)
        total_norm = torch.sqrt(sum(torch.norm(g, p=2) ** 2 for g in non_none_grads))
        
        # Normalize each gradient (avoid division by zero)
        eps = 1e-8
        normalized_grads = [g / (total_norm + eps) if g is not None else None 
                            for g in gradients]

            
        return normalized_grads
    

    def iterate(self, iterations_num, learning_rate = 1e-2, live_plot = False, save_gif = False):
        """
        Optimize obj_est using Adam optimizer to minimize total error and TV using MGD
        
        Args:
            initial_obj_est: Initial guess for the object estimate
            iterations_num: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            object_est, object_est_FT: Optimized object estimate and its Fourier transform 
            loss_history: History of loss values with Optimization
            
        """
        amp = torch.tensor(np.abs(self.object_est), dtype=torch.float64) 
        phase = torch.tensor(np.angle(self.object_est), dtype=torch.float64) 

        self.object_amp = torch.tensor(amp, dtype=torch.float64, requires_grad=True)
        self.object_phase = torch.tensor(phase, dtype=torch.float64, requires_grad=True)
        
               
        pupil_amp = torch.tensor(np.abs(self.pupil_func), dtype = torch.float64)
        pupil_phase = torch.tensor(np.angle(self.pupil_func), dtype = torch.float64)
        
        self.pupil_amp = torch.tensor(pupil_amp, dtype=torch.float64, requires_grad=True)
        self.pupil_phase = torch.tensor(pupil_phase, dtype=torch.float64, requires_grad=True)
        
        self.optimizer = torch.optim.Adam([self.object_amp, self.object_phase, self.pupil_amp, self.pupil_phase], lr = learning_rate)

        loss_history = []
             
        
        for iter_loop in range(iterations_num):
            print(f" iteration {iter_loop}")
            self.optimizer.zero_grad()
            
            self.complex_object = self.object_amp * torch.exp(1j*self.object_phase)
            
            mse_loss = self.get_total_error(self.complex_object)
            tv_loss = self.tv_penalty(self.complex_object)
            
            tv_grads = torch.autograd.grad(tv_loss, [self.object_amp, self.object_phase], retain_graph = True, allow_unused = True)
            mse_grads = torch.autograd.grad(mse_loss, [self.object_amp, self.object_phase], retain_graph = True)

            # Normalise Gradients
            tv_grads_norm = self._normalize_gradients(tv_grads)
            mse_grads_norm = self._normalize_gradients(mse_grads)

            # Combine gradients
            combined_grads = []
            for tv_grad, mse_grad in zip(tv_grads_norm, mse_grads_norm):
                if tv_grad is not None and mse_grad is not None:
                    combined_grads.append((tv_grad + mse_grad) / 2)
                elif tv_grad is not None:
                    combined_object_grads.append(tv_grad)
                elif mse_grad is not None:
                    combined_grads.append(mse_grad)
                else:
                    combined_grads.append(None)

            

            
            params = [self.object_amp, self.object_phase]
            for param, grad in zip(params, combined_grads):
                param.grad = grad

            self.optimizer.step()

            # Store loss history
            loss_history.append({
                'iteration': iter_loop,
                'mse_loss': mse_loss.item(),
                'tv_loss': tv_loss.item()
            })
            
    
        # Create final complex object
        self.final_object = self.object_amp * torch.exp(1j * self.object_phase)
        self.final_object_ft = self._FT2(self.final_object)
        self.losses_trend = loss_history
                    

    def _FT2(self, arr):
        '''
        Computing 2D FFT using PyTorch 
        Parameter:
        arr (torch.Tensor): tensor whose 2D Fourier transform is to be calculated
        Returns:
        arr_FT (torch.Tensor): complex-valued 2D Fourier transform of arr
        '''
        arr_FT = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(arr)))
        return arr_FT

    def _IFT2(self, arr_FT):
        '''
        Computing 2D IFFT using PyTorch 
        Parameter:
        arr_FT (torch.Tensor): tensor whose 2D Inverse Fourier transform is to be calculated
        Returns:
        arr (torch.Tensor): complex-valued 2D Inverse Fourier transform of arr_FT
        '''
        arr = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(arr_FT)))
        return arr

    
             

   
class optim_OP(optimization_MGD):

    def get_total_error(self):

        obj_est_FT = self._FT2(self.complex_object)
        num_images = len(self.images)
        data_mismatch_tensor = torch.zeros(num_images, dtype=torch.float32, device=self.complex_object.device)
    
        for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.k_out[:, 0], self.k_out[:, 1])):
            
            pupil_patch = self._update_spectrum(kx_iter, ky_iter)
            data_mismatch_tensor[i] = self._LSE(obj_est_FT, pupil_patch, image)
            
        total_data_mismatch = torch.sum(data_mismatch_tensor)
        return total_data_mismatch
    
    def iterate(self, iterations_num, learning_rate = 1e-2, live_plot = False, save_gif = False):
        """
        Update object and Pupil estimates in every iteration
        - using Adam optimizer to minimize total error and TV with MGD
        
        Args:
            iterations_num: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            object_est, object_est_FT: Optimized object estimate and its Fourier transform 
            loss_history: History of loss values with Optimization
            
    
            
        """
        amp = torch.tensor(np.abs(self.object_est), dtype=torch.float64) 
        phase = torch.tensor(np.angle(self.object_est), dtype=torch.float64) 

        self.object_amp = torch.tensor(amp, dtype=torch.float64, requires_grad=True)
        self.object_phase = torch.tensor(phase, dtype=torch.float64, requires_grad=True)

        
        pupil_amp = torch.tensor(np.abs(self.pupil_func), dtype = torch.float64)
        pupil_phase = torch.tensor(np.angle(self.pupil_func), dtype = torch.float64)
        
        self.pupil_amp = torch.tensor(pupil_amp, dtype=torch.float64, requires_grad=True)
        self.pupil_phase = torch.tensor(pupil_phase, dtype=torch.float64, requires_grad=True)
        
        self.optimizer = torch.optim.Adam([self.object_amp, self.object_phase, self.pupil_amp, self.pupil_phase], lr = learning_rate)

        loss_history = []
             
        
        for iter_loop in range(iterations_num):
            print(f" iteration {iter_loop}")
            self.optimizer.zero_grad()
            
            self.complex_object = self.object_amp * torch.exp(1j*self.object_phase)
            self.pupil_func = self.pupil_amp * torch.exp(1j*self.pupil_phase)
            
            mse_loss = self.get_total_error(self.complex_object)
            tv_loss = self.tv_penalty(self.complex_object)
            
            tv_grads = torch.autograd.grad(tv_loss, [self.object_amp, self.object_phase, self.pupil_amp, self.pupil_phase], retain_graph = True, allow_unused = True)
            mse_grads = torch.autograd.grad(mse_loss, [self.object_amp, self.object_phase, self.pupil_amp, self.pupil_phase], retain_graph = True)

            # Normalise Gradients
            tv_grads_norm = self._normalize_gradients(tv_grads)
            mse_grads_norm = self._normalize_gradients(mse_grads)

            # Combine gradients
            # combined_grads = []
            combined_object_grads = []
            for tv_grad, mse_grad in zip(tv_grads_norm, mse_grads_norm[:2]):
                if tv_grad is not None and mse_grad is not None:
                    combined_object_grads.append((tv_grad + mse_grad) / 2)
                elif tv_grad is not None:
                    combined_object_grads.append(tv_grad)
                elif mse_grad is not None:
                    combined_object_grads.append(mse_grad)
                else:
                    combined_object_grads.append(None)

            pupil_grads = mse_grads_norm[2:]

            all_grads = combined_object_grads + pupil_grads

            
            params = [self.object_amp, self.object_phase, self.pupil_amp, self.pupil_phase]
            for param, grad in zip(params, all_grads):
                param.grad = grad

            self.optimizer.step()

            # Store loss history
            loss_history.append({
                'iteration': iter_loop,
                'mse_loss': mse_loss.item(),
                'tv_loss': tv_loss.item()
            })
            
    
        # Create final complex object
        self.final_object = self.object_amp * torch.exp(1j * self.object_phase)
        self.final_object_ft = self._FT2(self.final_object)
        self.losses_trend = loss_history
        self.final_pupil = self.pupil_amp * torch.exp(1j * self.pupil_phase)

class optim_OAP(optimization_MGD):
    """
        Update object and Pupil estimates in alternate fashion
        - using Adam optimizer to minimize total error and TV with MGD
        
        Args:
            iterations_num: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            object_est, object_est_FT: Optimized object estimate and its Fourier transform 
            loss_history: History of loss values with Optimization
            
        """

    def get_total_error(self):

        obj_est_FT = self._FT2(self.object_est)
        num_images = len(self.images)
        data_mismatch_tensor = torch.zeros(num_images, dtype=torch.float32, device=self.object_est.device)
        
        for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.k_out[:, 0], self.k_out[:, 1])):
            
            pupil_patch = self._update_spectrum(kx_iter, ky_iter)
            data_mismatch_tensor[i] = self._LSE(obj_est_FT, pupil_patch, image)
            
        total_data_mismatch = torch.sum(data_mismatch_tensor)
        return total_data_mismatch
    
    def iterate(self, iterations_num, learning_rate=1e-2, live_plot=False, save_gif=False, alter_iter = 2):
        
        amp = torch.tensor(np.abs(self.object_est), dtype=torch.float64) 
        phase = torch.tensor(np.angle(self.object_est), dtype=torch.float64) 
        
        self.object_amp = torch.tensor(amp, dtype=torch.float64, requires_grad=True)
        self.object_phase = torch.tensor(phase, dtype=torch.float64, requires_grad=True)
    
        pupil_amp = torch.tensor(np.abs(self.pupil_func), dtype=torch.float64)
        pupil_phase = torch.tensor(np.angle(self.pupil_func), dtype=torch.float64)
        
        self.pupil_amp = torch.tensor(pupil_amp, dtype=torch.float64, requires_grad=True)
        self.pupil_phase = torch.tensor(pupil_phase, dtype=torch.float64, requires_grad=True)

        if live_plot:
            fig, axes, img_amp, img_phase, img_pupil_amp, img_pupil_phase = self._initialize_live_plot()
        
        self.optimizer = torch.optim.Adam([self.object_amp, self.object_phase, self.pupil_amp, self.pupil_phase], lr=learning_rate)
        
        loss_history = []
             
        for iter_loop in range(iterations_num):
            self.optimizer.zero_grad()
            
            if iter_loop % alter_iter == 0: 
                self._update_pupil()
                print(f"Iteration {iter_loop}: Updated object")
            else:  
                self._update_object()
                print(f"Iteration {iter_loop}: Updated pupil")
        
            # Store loss history
            loss_history.append({
                'iteration': iter_loop,
                'mse_loss': self.mse_loss.item(),
                'tv_loss': self.tv_loss.item() 
            })
            if live_plot and iter_loop % 5 == 0:
                self._update_live_plot(img_amp, img_phase, img_pupil_amp, img_pupil_phase, fig, axes)
            
        # Create final complex object
        self.final_object = self.object_amp * torch.exp(1j * self.object_phase)
        self.final_object_ft = self._FT2(self.final_object)
        self.losses_trend = loss_history
        self.final_pupil = self.pupil_amp * torch.exp(1j * self.pupil_phase)


    def _update_pupil(self):
        """Update only pupil parameters"""
        # Update complex representations
        self.object_est = self.object_amp * torch.exp(1j*self.object_phase)
        self.pupil_func = self.pupil_amp * torch.exp(1j*self.pupil_phase)
        
        self.mse_loss = self.get_total_error()
        self.tv_loss = self.tv_penalty(self.object_est)

        
        # Calculate gradients only for pupil parameters
        mse_grads = torch.autograd.grad(self.mse_loss, [self.pupil_amp, self.pupil_phase], retain_graph=True)
        
        # Set gradients for pupil parameters, None for object parameters
        self.object_amp.grad = None
        self.object_phase.grad = None
        self.pupil_amp.grad = mse_grads[0]
        self.pupil_phase.grad = mse_grads[1]
    
        self.optimizer.step()
        


    def _update_object(self):
        """Update only object parameters"""
        # Update complex representations
        self.object_est = self.object_amp * torch.exp(1j*self.object_phase)
        self.pupil_func = self.pupil_amp * torch.exp(1j*self.pupil_phase)
        
        self.mse_loss = self.get_total_error()
        self.tv_loss = self.tv_penalty(self.object_est)
        
        # Calculate gradients for object parameters
        tv_grads = torch.autograd.grad(self.tv_loss, [self.object_amp, self.object_phase], retain_graph=True, allow_unused=True)
        mse_grads = torch.autograd.grad(self.mse_loss, [self.object_amp, self.object_phase], retain_graph=True)
    
        # Normalize gradients
        tv_grads_norm = self._normalize_gradients(tv_grads)
        mse_grads_norm = self._normalize_gradients(mse_grads)
    
        # Combine gradients
        combined_object_grads = []
        for tv_grad, mse_grad in zip(tv_grads_norm, mse_grads_norm):
            if tv_grad is not None and mse_grad is not None:
                combined_object_grads.append((tv_grad + mse_grad) / 2)
            elif tv_grad is not None:
                combined_object_grads.append(tv_grad)
            elif mse_grad is not None:
                combined_object_grads.append(mse_grad)
            else:
                combined_object_grads.append(None)
    
        # Set gradients for object parameters, None for pupil parameters
        self.object_amp.grad = combined_object_grads[0]
        self.object_phase.grad = combined_object_grads[1]
        self.pupil_amp.grad = None
        self.pupil_phase.grad = None
    
        self.optimizer.step()




    ### Plotting
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
        img_amp = axes[0,0].imshow( self.object_amp.detach().numpy(), cmap='viridis')
        axes[0,0].set_title("Object Amplitude")
        cbar_amp = plt.colorbar(img_amp, ax=axes[0,0])

        img_phase = axes[0,1].imshow(self.object_phase.detach().numpy(), cmap='viridis')
        axes[0,1].set_title("Object Phase")
        img_phase.set_clim(-np.pi, np.pi)
        cbar_phase = plt.colorbar(img_phase, ax=axes[0,1])
        
        img_pupil_amp = axes[1,0].imshow(self.pupil_amp.detach().numpy(), cmap='viridis')
        axes[1,0].set_title("Fourier Amplitude")
        cbar_fourier = plt.colorbar(img_pupil_amp, ax=axes[1,0])
        
        img_pupil_phase = axes[1,1].imshow(self.pupil_phase.detach().numpy(), cmap='viridis')
        axes[1,0].set_title("pupil phase")
        img_pupil_phase.set_clim(-np.pi, np.pi)  # Set proper phase limits
        cbar_fourier = plt.colorbar(img_pupil_phase, ax=axes[1,1])

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
        amplitude_obj = self.object_amp.detach().numpy()
        phase_obj = self.object_phase.detach().numpy()
        
        amplitude_pupil = self.pupil_amp.detach().numpy()
        pupil_phi = self.pupil_phase.detach().numpy()
        
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
        