import numpy as np
import skimage as ski
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def make_gradients(video):
    gradients = np.zeros([3, 64, 256, 256])
    for frame_v in range(0, 63):#len(gradients)):
        for y_g in range(0, 255):
            for x_g in range(0, 255):
                gradients[0][frame_v][y_g][x_g] = (video[frame_v][y_g][x_g + 1] - video[frame_v][y_g][x_g])
                gradients[1][frame_v][y_g][x_g] = (video[frame_v][y_g + 1][x_g] - video[frame_v][y_g][x_g])
                gradients[2][frame_v][y_g][x_g] = (video[frame_v + 1][y_g][x_g] - video[frame_v][y_g][x_g])
    return gradients

def lk_voxel(gradients_v, x, y, z, n):
    # ignore boundary pixels
    voxel = gradients_v[:, max(z - n, 0):min(z + n, 63), max(y - n, 0):min(y + n, 254), max(x - n, 0):min(x + n, 254)]
    A= np.array([voxel[0].flatten(), voxel[1].flatten()]).T
    b=-np.array(voxel[2].flatten())
    return  np.linalg.lstsq(A, b.T)


def plot_vectors(gradients, stride, n):
    if not os.path.exists("./toy_plot"):
        os.mkdir("./toy_plot")
    for frame in range(0,63):
        plot = {"x":[],"y":[],"U":[],"V": []}
        print(frame)
        for y in range(0, 255):
            for x in range(0, 255):
                if ((x %stride == 0 and y %stride == 0)
                        # remove small gradients
                        and not (gradients[0][frame][y][x]**2 + gradients[1][frame][y][x]**2)**(1/2) < 0.005):
                    sol = lk_voxel(gradients, x, y, frame, n)
                    plot["x"].append(x)
                    #y is inverted for some reason
                    plot["y"].append(254-y)
                    plot["U"].append(sol[0][0])
                    plot["V"].append(sol[0][1])

        # remove outliers
        outliers = np.where((plot["U"] > np.mean(plot["U"]) + 3 * np.std(plot["U"])) | (plot["U"] < np.mean(plot["U"]) - 3 * np.std(plot["U"]))
                            | (plot["V"] > np.mean(plot["V"]) + 3 * np.std(plot["V"])) | (plot["V"] < np.mean(plot["V"]) - 3 * np.std(plot["V"])))
        plot = {key:np.delete(plot[key], outliers) for key in plot.keys()}

        plt.quiver(plot["x"], plot["y"], plot["U"], plot["V"], color="blue", scale = 20, width = 0.0008)
        if frame < 10: frameno = "0" + str(frame)
        else: frameno = str(frame)
        plt.title('Lukas-Kanade' + " frame " + str(frame) + " n = " + str(n) + " stride = " + str(stride))
        plt.savefig('./toy_plot/frame' + frameno + ".png", dpi=500)
        # plt.show()
        plt.clf()

vid = [ski.io.imread("./toy_problem/" + image) for image in os.listdir("./toy_problem")]
vid = [rgb2gray(image) for image in vid]

vid_gradients = make_gradients(vid)
plot_vectors(vid_gradients, 4, 5)
