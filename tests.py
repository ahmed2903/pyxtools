import numpy as np
import matplotlib.pyplot as plt


from xrays_fs import reverse_kins_to_pixels, convergent_kins, apply_aberattions_to_kins, gaussian_amplitude, defocus_aberration, generate_detector_image

kins = convergent_kins(.7, .2, 0.5, int(1e5))
kins, weights = apply_aberattions_to_kins(kins=kins, amplitude_profile=gaussian_amplitude, phase_aberration=defocus_aberration)
coords = reverse_kins_to_pixels(kins, 75, 1e6, (1000,1000))

image = generate_detector_image(np.abs(weights)**2, kins, (.1,.1), (75,75),.11)


plt.figure()
plt.imshow(image)
plt.show()


image = generate_detector_image(np.angle(weights), kins, (.1,.1), (75,75),.11)
plt.figure()
plt.imshow(image)
plt.show()