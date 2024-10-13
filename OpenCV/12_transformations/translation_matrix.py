import cv2 as cv
import numpy as np
import os
from albert_utils import transformations

img = cv.imread(os.path.join(".", "assets", "fox.jpg"))
# img = operations.translate(img, x=100, y=100)
img = transformations.translate(img, x=200, y=-100)

cv.imshow("img", img)

cv.waitKey(0)
cv.destroyAllWindows()
