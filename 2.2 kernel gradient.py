import numpy as np
import skimage as ski
import os
from skimage.color import rgb2gray
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

img = [ski.io.imread("./toy_problem/" + image) for image in os.listdir("./toy_problem")]
img = [rgb2gray(image) for image in img]

ndimg_x = ndi.convolve(img[1], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
ndimg_y = ndi.convolve(img[1], np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).T)
ndimg_z = ndi.convolve(img[0:2], np.array([-2, 0, 2]), axes=(0))

ndimg_z2 = ndi.convolve(img[0:2],
               [[[-1, -1, -1], [-1, -2, -1], [-1, -1, -1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[1, 1, 1], [1, 2, 1], [1, 1, 1]]], axes=(0, 1, 2))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(ndimg_x)
ax[0].set_title("x")
ax[1].imshow(ndimg_y)
ax[1].set_title("y")
ax[2].imshow(ndimg_z[0])
ax[2].set_title("z")
plt.show()
