import os
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # Or 'Qt5Agg' if you have PyQt installed
from matplotlib import animation
from skimage.color import rgb2gray


image_list = [mpimg.imread("./toy_plot/" + img) for img in os.listdir('toy_plot')]
# image_list = [rgb2gray(img) for img in image_list]


# Assuming 'image_list' is your list of NumPy arrays or images
fig, ax = plt.subplots()

# # Remove axes for a cleaner look
ax.set_axis_off()

frames = []
for img in image_list:
    im = ax.imshow(img, animated=True)
    frames.append([im])

# Create the animation
ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)#, repeat_delay=1000)
writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save("LK.gif", writer=writer)
plt.show()