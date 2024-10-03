import os
import cv2
import numpy as np

img_path = os.path.join(".", "assets", "cat.jpg")
img = cv2.imread(img_path)

"""
>> threshold1 -> This is the lower threshold for the hysteresis procedure 
in the Canny algorithm. It is the minimum gradient value (intensity change) 
that is considered a potential edge.
>> threshold2 -> This is the upper threshold for the hysteresis procedure.
Any pixel with a gradient value greater than or equal to this threshold 
is definitely considered an edge.
>> hysteresis ensures that:
* Strong edges are always detected.
* Weak edges are only detected if they form part of a valid edge (connected to strong edges).
* Noise and insignificant details are filtered out.
"""
img_canny = cv2.Canny(img, threshold1=150, threshold2=200)
img_dilate = cv2.dilate(img_canny, np.ones((2, 2), dtype=np.int8))
img_erode = cv2.erode(img_canny, np.ones((1, 1), dtype=np.int8))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img_open = cv2.morphologyEx(img_canny, cv2.MORPH_OPEN, kernel)

cv2.imshow("canny", img_canny)
# cv2.imshow("dilate", img_dilate)
# cv2.imshow("erode", img_erode)
cv2.imshow("open", img_open)

cv2.waitKey(0)
cv2.destroyAllWindows()
