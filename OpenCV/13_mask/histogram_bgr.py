import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

img = cv.imread(os.path.join(".", "assets", "cat2.jpg"))

circle_mask = None
blank = np.zeros(img.shape[:2], dtype=np.uint8)
# Uncomment this to use the mask
# circle_mask = cv.circle(blank.copy(), (blank.shape[1] // 2, blank.shape[0] // 2), 150, 255, cv.FILLED)
# cv.imshow("mask", circle_mask)

# masked = cv.bitwise_and(img, img, mask=circle_mask)

cv.imshow("img", img)

plt.ioff()
plt.figure()
plt.title("Colored Images Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")

colors = ("b", "g", "r")
for i, c in enumerate(colors):
    """
    >> histSize -> quantity of Bins (interval of pixels intensities).
    >> ranges -> 0 to 256 because that's the limit for pixel values.
    """
    hist = cv.calcHist([img], channels=[i], mask=circle_mask, histSize=[256], ranges=[0, 256])
    plt.plot(hist, color=c)
    plt.xlim([0, 256])

plt.show()

plt.close()
cv.waitKey()
cv.destroyAllWindows()
