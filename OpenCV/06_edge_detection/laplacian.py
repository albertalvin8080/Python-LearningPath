import cv2 as cv
import numpy as np
import os

img = cv.imread(os.path.join(".", "assets", "cat2.jpg"))

"""
>> ddepth=cv.CV_64F:
The Laplacian operator computes second-order derivatives, and the resulting pixel values can be 
positive or negative (indicating the direction of intensity changes).
Using cv.CV_64F allows you to capture these precise values, especially when they are outside the 
usual 8-bit range (0 to 255).
"""
# >> ddepth -> The ddepth parameter determines the data type of the result matrix. 
laplaced = cv.Laplacian(img, ddepth=cv.CV_64F, ksize=1)
laplaced = np.uint8(np.absolute(laplaced))
laplaced2 = cv.Laplacian(img, ddepth=cv.CV_64F, ksize=3)
laplaced2 = np.uint8(np.absolute(laplaced2))

cv.imshow("laplaced", laplaced)
cv.imshow("laplaced2", laplaced2)

cv.waitKey()
cv.destroyAllWindows()
