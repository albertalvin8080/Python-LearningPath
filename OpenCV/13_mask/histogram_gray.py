import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

img = cv.imread(os.path.join(".", "assets", "cat.jpg"))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
"""
>> histSize -> quantity of Bins (interval of pixels intensities).
>> ranges -> 0 to 256 because that's the limit for pixel values.
"""
hist = cv.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

plt.ioff()
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()

# -----------------------------------------------------------------------

blank = np.zeros(gray.shape[:2], dtype=np.uint8)
circle = cv.circle(blank.copy(), (blank.shape[1]//2, blank.shape[0]//2), 200, 255, cv.FILLED)

masked = cv.bitwise_and(gray, gray, mask=circle)
hist_mask = cv.calcHist([gray], [0], masked, histSize=[256], ranges=[0, 256])

cv.imshow("mask", masked)

plt.ioff()
plt.figure()
plt.title("Mask Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()

plt.close()
cv.waitKey()
cv.destroyAllWindows()