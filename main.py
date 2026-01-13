from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import numpy as np

img = load_image("image.jpg")

clean_img = median(img, ball(3))

edgeMAG = edge_detection(clean_img)

threshold = edgeMAG.mean()
edge_binary = edgeMAG > threshold

edge_binary = (edge_binary * 255).astype(np.uint8)
Image.fromarray(edge_binary).save("my_edges.png")

