import cv2
import numpy as np

img = cv2.imread("circles.png")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (5, 5), sigmaX=0)

"""
-> dp stands for the inverse ratio of the accumulator resolution to the image resolution.
- Effect of dp:
dp = 1: The accumulator has the same resolution as the input image, making circle detection more precise but computationally heavier.
dp > 1: The accumulator has a smaller resolution, making detection faster but possibly less accurate. Larger dp values reduce the computation by scaling down the resolution of the accumulator, but this may miss small or weak circles.

-> minDist: the minimum allowed distance between the centers of the detected circles.
This parameter controls how close detected circles can be to each other. If two circle centers are closer
than minDist, the algorithm will consider them as the same circle, or it will skip one of them.

-> param1: This parameter is the higher threshold used in the Canny edge detection phase. 
A higher value for param1 means that only strong edges (with gradient values above param1) will be considered. This makes the circle detection more selective, detecting only prominent circles.
-> param2: This is the threshold for center detection in the accumulator space.
"""
circles = cv2.HoughCircles(
    gray_img,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=120,
    param1=100,
    param2=30,
    minRadius=0,
    maxRadius=0,
)

"""
-> The cv2.HoughCircles() function returns the coordinates of the detected circles (center x, y, and radius r) as floating-point numbers. These values may contain decimal points because the circle detection algorithm works in continuous space, meaning it can produce fractional results for the circle centers and radii.
-> np.around(circles) rounds these floating-point values to the nearest integer because, for practical purposes (such as drawing circles on an image), you need integer pixel coordinates.
"""
circles = np.uint16(np.around(circles))
print(circles)

for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow("HoughCirlces", img)

cv2.waitKey()
cv2.destroyAllWindows()
