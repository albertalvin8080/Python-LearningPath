import os
import cv2
from albert_utils import mouse_metadata

img_path = os.path.join(".", "assets", "fox.jpg")
img = cv2.imread(img_path)

kernel_value = 7
# This is the standard deviation of the Gaussian kernel in the X or Y direction.
# A higher value results in a more blurred image. 
# If set to 0, OpenCV will automatically calculate the value based on the kernel size.
sigma_value = 0

blur_img = cv2.blur(img, (kernel_value, kernel_value))
gaussian_img = cv2.GaussianBlur(img, (kernel_value, kernel_value), sigma_value)
median_img = cv2.medianBlur(img, kernel_value)

cv2.imshow("fox", img)
cv2.imshow("blur", blur_img)
cv2.imshow("gaussian", gaussian_img)
cv2.imshow("median", median_img)

cv2.waitKey(0)
cv2.destroyAllWindows()