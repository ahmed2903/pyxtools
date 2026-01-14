import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import rotate

# class random_weighted_norm:
#     def __init__(self, img_size = 64, pos_var = 0.1, w_max = 2, r = None):
#         x,y = np.meshgrid(*[np.linspace(-1,1, img_size) for _ in [0,1]])
#         self.xy = np.stack([x,y])
#         self.pos_var = pos_var
#         self.w_max = w_max
#         self.r = (0.3,0.6) if r is None else r
          
#     def __call__(self,p=1):
#         w = np.random.uniform(1., self.w_max, size=(2,))
#         r = np.random.uniform(*self.r)
#         m = np.random.normal(0, self.pos_var, size=(2,1,1))
#         return self.weighted_norm(m, w, p, r)
    
#     def weighted_norm(self, m, w, p, r):
#         return 1. * (np.linalg.norm((self.xy - m) * w[:,None,None], axis=0, ord=p) < r)
    

# def get_data(batch_size, img_size, noise_lvl):
#     x_recon = np.zeros((batch_size, 1, img_size, img_size))
#     x = np.zeros((batch_size, 1, img_size, img_size))
    
#     rwn = random_weighted_norm(img_size=img_size)

#     # set real images and inputs
#     for i in range(batch_size):
    
#         x[i,0,...] = rwn(1)
#         x_recon[i, 0, ...] = x[i,0,...] + np.random.normal(0, noise_lvl, size=x[i,0,...].shape)
    
#     return x_recon, torch.Tensor(x)



# ---------------------------------------------------------
# Primitive shape generator
# ---------------------------------------------------------

class RandomWeightedNorm:
    def __init__(self, img_size=64, pos_var=0.3, w_max=3.0, r=(0.2, 0.6)):
        x, y = np.meshgrid(
            *[np.linspace(-1, 1, img_size) for _ in range(2)],
            indexing="ij"
        )
        self.xy = np.stack([x, y])
        self.pos_var = pos_var
        self.w_max = w_max
        self.r = r

    def __call__(self, p):
        w = np.random.uniform(1.0, self.w_max, size=(2,))
        r = np.random.uniform(*self.r)
        m = np.random.normal(0, self.pos_var, size=(2, 1, 1))
        sharpness = np.random.uniform(10, 40)
        return self.soft_weighted_norm(m, w, p, r, sharpness)

    def soft_weighted_norm(self, m, w, p, r, sharpness):
        d = np.linalg.norm(
            (self.xy - m) * w[:, None, None],
            axis=0,
            ord=p
        )
        return 1.0 / (1.0 + np.exp(sharpness * (d - r)))


# ---------------------------------------------------------
# Shape composition
# ---------------------------------------------------------

def generate_shapes(img_size):
    rwn = RandomWeightedNorm(img_size=img_size)

    n_obj = np.random.randint(1, 5)
    img = np.zeros((img_size, img_size))

    # for _ in range(n_obj):
    p = np.random.choice([1, 2, 4, 8, 16, 32, np.inf])
    img += rwn(p)

    img = np.clip(img, 0, 1)

    return img


# ---------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------

def elastic_deform(img, alpha=5, sigma=4):
    shape = img.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha

    x, y = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        indexing="ij"
    )

    coords = np.array([x + dx, y + dy])
    return map_coordinates(img, coords, order=1)


def random_affine(img):
    angle = np.random.uniform(0, 360)
    return rotate(img, angle, reshape=False, order=1)


# ---------------------------------------------------------
# Intensity + background
# ---------------------------------------------------------

def apply_intensity(img):
    
    amp = np.random.uniform(0.5, 2.0)
    img = img * amp

    if np.random.rand() < 0.3:
        bg = gaussian_filter(np.random.randn(*img.shape), sigma=10)
        img += bg * np.random.uniform(0.05, 0.2)

    return img


# ---------------------------------------------------------
# Noise models
# ---------------------------------------------------------

def apply_noise(img, noise_lvl):
    noise = np.random.poisson(img)
    # if np.random.rand() < 0.4:
    #     # Gaussian
    #     # noise = np.random.normal(0, noise_lvl, img.shape)    
        
    # elif np.random.rand() < 0.7:
    #     # Laplacian
    #     noise = np.random.laplace(0, noise_lvl, img.shape)
    # else:
    #     # Correlated Gaussian
    #     noise = gaussian_filter(
    #         np.random.normal(0, noise_lvl, img.shape),
    #         sigma=np.random.uniform(0.5, 2.0)
    #     )

    # spatially varying noise
    sigma_map = np.random.uniform(0.7, 1.3, size=img.shape)
    return img + noise 


# ---------------------------------------------------------
# Main batch generator
# ---------------------------------------------------------

def get_data(batch_size, img_size=64):
    
    x_clean = np.zeros((batch_size, 1, img_size, img_size))
    x_noisy = np.zeros_like(x_clean)

    for i in range(batch_size):
        img = generate_shapes(img_size)

        if np.random.rand() < 0.7:
            img = random_affine(img)

        if np.random.rand() < 0.7:
            img = elastic_deform(img)

        noise_lvl = np.abs(np.random.normal(0,.2,1))
        
        img = np.abs(apply_intensity(img))
        
        noisy = apply_noise(img, noise_lvl)

        x_clean[i, 0] = img
        x_noisy[i, 0] = noisy

    return torch.tensor(x_noisy, dtype=torch.float32), \
           torch.tensor(x_clean, dtype=torch.float32)
