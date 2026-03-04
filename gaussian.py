import numpy as np
import skimage as ski
import os
from skimage.color import rgb2gray
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

img = [ski.io.imread("./toy_problem/" + image) for image in os.listdir("./toy_problem")]
img = [rgb2gray(image) for image in img]


ndimg_x = ndi.gaussian_filter1d(img[1], 4, order=1, axis=1)
ndimg_y = ndi.gaussian_filter1d(img[1], 4, order=1, axis=0)
ndimg_z = ndi.gaussian_filter1d(img[0:9], 4, order=1, axis=0)

print(len(ndimg_z))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(ndimg_x)
ax[0].set_title("x")
ax[1].imshow(ndimg_y)
ax[1].set_title("y")
ax[2].imshow(ndimg_z[4])
ax[2].set_title("z")
plt.show()
