import cv2 as cv
import numpy as np
import os
from albert_utils import transformations

img = cv.imread(os.path.join(".", "assets", "fox.jpg"))
cv.imshow("img", img)

# positive angle -> anti-clockwise
# negative angle -> clockwise
other = transformations.rotate(img, 45)
cv.imshow("other", other)

other = transformations.rotate(img, 90)
cv.imshow("other2", other)

other = transformations.rotate(img, -90)
cv.imshow("other3", other)

cv.waitKey(0)
cv.destroyAllWindows()
