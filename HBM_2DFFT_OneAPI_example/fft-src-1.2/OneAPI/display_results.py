# Create a plot of the input and output images created by this example

from matplotlib import pyplot as plt
import numpy as np
import math

# Select colour map
plt.rc('image',cmap='inferno')

# Load images
image = open("image_in.raw","r")
a = np.fromfile(image, dtype=np.uint16)
a = a.reshape(1024,1024)
image2 = open("image_out.raw","r")
b = np.fromfile(image2, dtype=np.uint16)
b = b.reshape(1024,1024)

# Create plots
plt.rcParams["figure.figsize"] = (15,7)
plt.subplot(1,2,1)
plt.title('Imput Image')
plt.imshow(a)
plt.subplot(1,2,2)
plt.title('Transformed Image')
plt.imshow(b)

# Show plot
plt.show()
