import numpy as np
import skimage as ski
import os
import matplotlib
from matplotlib import image as mpimg
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from matplotlib import animation


def make_gradients_simple(video):
    gradients = np.zeros([3, dimensions[0], dimensions[1], dimensions[2]])
    for frame_v in range(0, dimensions[0]):  # len(gradients)):
        for y_g in range(0, dimensions[1]):
            for x_g in range(0, dimensions[2]):
                gradients[0][frame_v][y_g][x_g] = (video[frame_v][y_g][x_g + 1] - video[frame_v][y_g][x_g])
                gradients[1][frame_v][y_g][x_g] = (video[frame_v][y_g + 1][x_g] - video[frame_v][y_g][x_g])
                gradients[2][frame_v][y_g][x_g] = (video[frame_v + 1][y_g][x_g] - video[frame_v][y_g][x_g])
    return gradients


def make_gradients_kernel(video):
    gradients = np.zeros([3, dimensions[0], dimensions[1], dimensions[2]])
    gradients[0] = [ndi.convolve(video[i], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) for i in range(1, dimensions[0])]
    gradients[1] = [ndi.convolve(video[i], np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).T) for i in range(1, dimensions[0])]
    gradients[2] = [ndi.convolve(video[i - 1:i + 1],
                                 [[[-1, -1, -1], [-1, -2, -1], [-1, -1, -1]],
                                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                  [[1, 1, 1], [1, 2, 1], [1, 1, 1]]], axes=(0, 1, 2))[0] for i in range(1, dimensions[0])]
    return gradients


def make_gradients_gaussian(video):
    gradients = np.zeros([3, dimensions[0], dimensions[1], dimensions[2]])
    gradients[0] = [ndi.gaussian_filter1d(video[i], 4, order=1, axis=1) for i in range(0,dimensions[0])]
    gradients[1] = [ndi.gaussian_filter1d(video[i], 4, order=1, axis=0) for i in range(0,dimensions[0])]
    gradients[2] = ndi.gaussian_filter1d(video, 4, order=1, axis=0)# for i in range(0,64)]

    return gradients


def lk_voxel(gradients_v, x, y, z, n):
    # ignore boundary pixels
    voxel = gradients_v[:, max(z - n, 0):min(z + n, dimensions[0]), max(y - n, 0):min(y + n, dimensions[1]), max(x - n, 0):min(x + n, dimensions[2])]
    A = np.array([voxel[0].flatten(), voxel[1].flatten()]).T
    b = -np.array(voxel[2].flatten())
    return np.linalg.lstsq(A, b.T)




def plot_vectors(gradients, n, stride, gradient_type):
    path = "./edgar_n" + str(n) + "_stride" + str(stride) + "_gradient_" + gradient_type
    if not os.path.exists(path):
        os.mkdir(path)
    gradient_len = [(x**2 + y**2)**(1/2) for x,y in zip(gradients[0].flatten(), gradients[1].flatten())]
    print(max(gradient_len))
    cutoff =  np.percentile(gradient_len, 90)
    print(cutoff)

    for frame in range(n, dimensions[0] - n):
        plot = {"x": [], "y": [], "U": [], "V": []}
        print(frame)
        for y in range(n, dimensions[1] - n):
            for x in range(n, dimensions[2] - n):
                if ((x % stride == 0 and y % stride == 0)
                        # remove small gradients
                        and not (gradients[0][frame][y][x] ** 2 + gradients[1][frame][y][x] ** 2) ** (1 / 2) < cutoff
                ):
                    sol = lk_voxel(gradients, x, y, frame, n)
                    plot["x"].append(x)
                    plot["y"].append(y)
                    plot["U"].append(sol[0][0])
                    plot["V"].append(sol[0][1])

        # remove outliers
        # sdevs = 7
        # outliers = np.where((plot["U"] > np.mean(plot["U"]) + sdevs * np.std(plot["U"])) | (
        #         plot["U"] < np.mean(plot["U"]) - sdevs * np.std(plot["U"]))
        #                     | (plot["V"] > np.mean(plot["V"]) + sdevs * np.std(plot["V"])) | (
        #                             plot["V"] < np.mean(plot["V"]) - sdevs * np.std(plot["V"])))
        range_U = [np.percentile(plot["U"],1), np.percentile(plot["U"],99)]
        range_V = [np.percentile(plot["V"],1), np.percentile(plot["V"],99)]
        outliers = np.where((plot["U"] > range_U[1]) | (plot["U"] < range_U[0]) | (plot["V"] > range_V[1]) | (plot["V"] < range_V[0]))
        plot = {key: np.delete(plot[key], outliers) for key in plot.keys()}

        plt.imshow(vid[frame])
        plt.quiver(plot["x"], plot["y"], plot["U"], plot["V"], color="red", scale=20, width=0.0005)
        if frame < 10:
            frameno = "0" + str(frame)
        else:
            frameno = str(frame)
        plt.title('Lukas-Kanade' + " frame " + str(frame) + " n = " + str(n) + " stride = " + str(stride) + " gradient = " + gradient_type)
        plt.savefig(path + "/frame" + frameno + ".png", dpi=500)
        plt.clf()
    plt.close()
    animate(path)

def animate(path):
    image_list = [mpimg.imread(path +"/" +  img) for img in os.listdir(path)]
    fig, ax = plt.subplots()
    ax.set_axis_off()
    frames = [[ax.imshow(img, animated=True)] for img in image_list]

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save(path + ".gif", writer=writer)
    plt.close()

def LK(gradient, n, stride):
    if gradient == "gauss":
        vid_gradients = make_gradients_gaussian(vid)
    elif gradient == "kernel":
        vid_gradients = make_gradients_kernel(vid)
    else:
        gradient = "simple"
        vid_gradients = make_gradients_simple(vid)

    plot_vectors(vid_gradients, n, stride, gradient)

vid = np.array([rgb2gray(ski.io.imread("./edgar/" + image)) for image in os.listdir("./edgar")])
dimensions = vid.shape
print(dimensions)
LK("gauss", 2,4)

