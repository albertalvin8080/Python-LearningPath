import os
import cv2
from albert_utils import mouse_metadata

img_path = os.path.join(".", "assets", "threshold_text.png")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# defines the size of the local region (or neighborhood) around each pixel that is used to calculate the threshold for that specific pixel.
block_size = 21
# constant subtracted from the computed mean or weighted mean of the neighborhood (block) around the pixel. It adjusts the threshold value for that pixel.
dst = 30
threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, dst)

cv2.imshow("text", img)
cv2.imshow("threshold", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()