import cv2 as cv
import numpy as np
import os

img = cv.imread(os.path.join(".", "assets", "fox.jpg"))
# These three are 2x2 grayscale images with the intensities of each color.
# The higher the brightness of the pixel, more of that color is present in the pixel of the original image.
# Each array has the value of the corresponding color from the original image.
b_img, g_img, r_img = cv.split(img)

cv.imshow("b", b_img)
cv.imshow("g", g_img)
cv.imshow("r", r_img)

# combining them in the right order back to the original image
merge_img = cv.merge([b_img, g_img, r_img])
cv.imshow("merge", merge_img)

# ----------------------------------------------------------------

# showing only the respective colors
blank = np.zeros(img.shape[:2], dtype=np.uint8)
cv.imshow("blueish", cv.merge([b_img, blank, blank]))
cv.imshow("greenish", cv.merge([blank, g_img, blank]))
cv.imshow("redish", cv.merge([blank, blank, r_img]))

cv.waitKey(0)
cv.destroyAllWindows()