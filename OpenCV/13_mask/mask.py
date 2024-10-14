import cv2 as cv
import numpy as np
import os

img = cv.imread((os.path.join(".", "assets", "cat2.jpg")))

blank = np.zeros(img.shape[:2], dtype=np.uint8)

# circle and rectangle are the masks
circle = cv.circle(blank.copy(), (blank.shape[1]//2-100, blank.shape[0]//2), 100, 255, cv.FILLED)
rectangle = cv.rectangle(blank.copy(), (100, 100), (300, 300), 255, cv.FILLED)

masked = cv.bitwise_and(img, img, mask=circle)
masked2 = cv.bitwise_and(img, img, mask=rectangle)

cv.imshow("circle", circle)
cv.imshow("masked", masked)
cv.imshow("masked2", masked2)

cv.waitKey(0)
cv.destroyAllWindows()