import numpy as np
import skimage as ski
import os
from skimage.color import rgb2gray

import matplotlib.pyplot as plt


img = [ski.io.imread("./toy_problem/" + image) for image in os.listdir("./toy_problem")]
img = [rgb2gray(image) for image in img]

gradients = np.zeros([3,64, 255, 255])
for frame in range(1,2):#len(gradients)):
    for y in range(1, 255):
        for x in range(1, 255):
            gradients[0][frame][y][x] = (img[frame][y][x] - img[frame][y][x - 1]) / 2 + 0.5
            gradients[1][frame][y][x] = (img[frame][y][x] - img[frame][y - 1][x]) / 2 + 0.5
            gradients[2][frame][y][x] = (img[frame][y][x] - img[frame - 1][y][x]) / 2 + 0.5

print(gradients[1])

# gradients_x = np.zeros([64, 255, 255])
# for frame in range(1,2):#len(gradients)):
#     for x in range(1,255):
#         for y in range(1,255):
#             gradients_x[frame][x][y] = gradients[frame][x][y][0]/2+0.5
#

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(gradients[0][1])
ax[0].set_title("x")
ax[1].imshow(gradients[1][1])
ax[1].set_title("y")
ax[2].imshow(gradients[2][1])
ax[2].set_title("z")
plt.show()

# print(len(img[0][]))