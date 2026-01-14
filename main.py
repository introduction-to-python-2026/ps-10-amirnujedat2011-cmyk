import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import numpy as np

image = load_image('original_image.jpg') 

clean_image = median(image, ball(3))

edge_mag = edge_detection(clean_image)

threshold = 100 
edge_binary = edge_mag > threshold

edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save('my_edges.png')

plt.imshow(edge_binary, cmap='gray')
plt.show()
