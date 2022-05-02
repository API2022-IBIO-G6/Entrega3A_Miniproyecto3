import glob
import os

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

train = [f for f in glob.glob(os.path.join('data_mp3\\DB\\train', '*.jpg'))]
test = [f for f in glob.glob(os.path.join('data_mp3\\DB\\test', '*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data_mp3\\DB\\Val', '*.jpg'))]
image =  plt.imread(train[0])
image_grey = image[:,:,0]


fd, hog_image = hog(image_grey, orientations=15, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
print(fd) # fd es un vector de caracteristicas
print(hog_image) # hog_image es una imagen

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image_grey, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

