import os
import cv2 as cv
import numpy as np

img_path = os.path.join(".", "assets", "cat2.jpg")
img = cv.imread(img_path)

"""
>> sobel_x_or_y: This result combines edges detected separately in the x and y directions, 
using a bitwise OR. It highlights edges in both directions but does not consider their 
combined magnitude or diagonal edges.
>> sobel_xy: This result shows the gradient computed in both x and y directions at once, 
effectively detecting diagonal edges. It captures the true change in both directions, 
yielding a more mathematically correct combination of x and y gradients.
"""
sobel_x = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0)
sobel_y = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1)
sobel_x_or_y = cv.bitwise_or(sobel_x, sobel_y)
sobel_xy = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=1)

cv.imshow("x", sobel_x)
cv.imshow("y", sobel_y)
cv.imshow("x_or_y", sobel_x_or_y)
cv.imshow("xy", sobel_xy)

cv.waitKey()
cv.destroyAllWindows()