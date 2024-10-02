import os
import cv2
from albert_utils import mouse_metadata

img_path = os.path.join(".", "assets", "fox.jpg")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold_value = 130
# Pixels above `threshold_value` will receive the value 255 (white)
# Pixels below `threshold_value` will receive the value 0 (black)
ret, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
# ret, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
# ret, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_MASK)

cv2.imshow("fox_gray", img)
cv2.imshow("thresh", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
