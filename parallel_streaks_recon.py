# Ahmed H. Mokhtar
#
# Given a Bragg Signal from a convergent beam data
# This script does the reconstruction from multiple parallel streaks 

 

import numpy as np
import data_loader.cbd_loader as db
import data_loader.utils as ut
from phase_retrieval import epry_class
from phase_retrieval import utils_pr

############# USER INPUT #################

# experimental params
class Exp:
    det_psize = 75 
    energy = 17.3 #keV
    wavelength = ut.energy2wavelength_a(energy)
    det_npixel = ( 4362, 4148)
    crys_distance = 130e3
    step_size = 100 # Angstroms
    number_of_steps = 101 # steps
    beam_spot_size = 100 # Angstroms

    lens1 = {'focal_length': 1.2e-3,
                'height'   : 40.2e-6
            } 
    lens2 = {'focal_length': 1.1e-3,
                'height'   : 38.6e-6
            }
    
exp = Exp()
exp.centre_pixel = np.array((142+2170,28+2090))

# data load initialisation
gold = db.load_data(
    directory = '/asap3/petra3/gpfs/p11/2023/data/11018188/raw/scan_frames/',
    scan_num = 399,
    det_psize = exp.det_psize, 
    det_distance = exp.crys_distance,
    centre_pixel = exp.centre_pixel, 
    wavelength = exp.wavelength,
    slow_axis = 0, # 0 for x, 1 for y
    beamtime = 'new',
    fast_axis_steps=101,
)

# pupil
gold.add_roi(roi_name = "pupil" , roi = [2190,2295,2110,2230])
pupil_kout_mask = 80
pupil_NA_mask = 1e3
streak_mask = 1
mask_region_pupil = [10,100,4,10]

# streak
gold.add_roi(roi_name = "streak", roi = [3685,3795,2725,2895]) 
streak_mask = .5
est_ttheta = 34.329

# streak offsets
offsets = range(-6,6)

iterations = 500

############# Preparing #################

# Pupil things
gold.prepare_roi("pupil", mask_val = pupil_kout_mask)
gold.mask_region_detector("pupil", mask_region_pupil, 'zeros')
gold.estimate_pupil_size(mask_val=pupil_NA_mask)
gold.add_lens("back", exp.lens1['focal_length'], exp.lens1['height'])
gold.add_lens("front", exp.lens2['focal_length'], exp.lens2['height'])
gold.estimate_detector_distance_from_NA()
gold.compute_kins("pupil", est_ttheta=0)

# Streak
gold.prepare_roi("streak", mask_val = streak_mask)
ttheta_rad = np.deg2rad(est_ttheta) 
gold.compute_kins("streak", est_ttheta = ttheta_rad)
gold.prepare_coherent_images("streak", 
                             order_imgs=True
                            )
gold.average_frames_roi("streak")

########### Reconstruction #################

from numpy.fft import fftshift, ifftshift, fft2, ifft2

#  initialisation 
image_ft = fftshift(fft2(gold.averaged_coherent_images['streak']) ) #*np.exp(1j*coh_images[0])))

init_fourier = image_ft 
init_object = ifft2(ifftshift(init_fourier))

N = len(offsets)

full_arr = np.zeros((N,gold.coherent_imgs['streak'].shape[0], gold.coherent_imgs['streak'].shape[1]), dtype=complex)


for i, offset in enumerate(offsets):
    
    mask = gold.select_single_pixel_streak("streak", width=1, offset=offset)
    
    imgs = gold.coherent_imgs['streak'][mask]
    kins = gold.kins['streak'][mask]
    
    epry = epry_class.EPRy_lr(imgs, 
            pupil_func = None, 
            hr_fourier_image = init_fourier,
            kout_vec = kins,
            ks_pupil = kins,
            lr_psize = exp.step_size,
            alpha = .2, 
            beta = .9)
    
    epry.prepare(extend = 'double')
    epry.iterate(iterations=iterations)
    
    full_arr[i,:,:] += epry.hr_obj_image
    

np.save(full_arr, "3d_recon.npy")