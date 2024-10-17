import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv.imread(os.path.join(".", "assets", "cat2.jpg"))
# Matrix used to increment or decrement the brightness of an image.
matrix = np.ones(img.shape, dtype=np.uint8) * 50

brighter = cv.add(img, matrix)
darker = cv.subtract(img, matrix)

plt.figure(figsize=[18,5])
plt.subplot(141)
plt.imshow(img[:,:,::-1])
plt.title("Original")
plt.show()

plt.close()
