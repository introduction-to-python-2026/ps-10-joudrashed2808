from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image(path):
    img = Image.open(path).convert("L")
    img = np.array(img, dtype=np.float32)
    return img
    
def edge_detection(image):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    gx = convolve2d(image, sobel_x, mode="same", boundary="symm")
    gy = convolve2d(image, sobel_y, mode="same", boundary="symm")

    edges = np.sqrt(gx**2 + gy**2)

    return edges
