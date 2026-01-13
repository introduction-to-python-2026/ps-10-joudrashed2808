from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    return img

def edge_detection(img):
    gray = img.mean(axis=2)

    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
