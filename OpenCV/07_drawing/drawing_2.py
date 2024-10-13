import cv2 as cv
import numpy as np
import os

img = np.zeros((500, 500, 3), dtype=np.uint8)

# same effect when using thickness=cv.FILLED
# img[0:250, 0:500] = 255, 0, 0
# cv.rectangle(img, (0, 0), (250, 250), (255, 0, 0), thickness=cv.FILLED)
cv.rectangle(
    img,
    (0, 0),
    # x, y
    (img.shape[0] // 2, img.shape[1]),
    (255, 0, 0),
    thickness=cv.FILLED,
)
print(img.shape)

# single blue pixel
img[250, 250, 0] = 255
# img.itemset((250, 250, 0), 255) # removed from NumPy 2.0

cv.imshow("img", img)

cv.waitKey(0)
cv.destroyAllWindows()
