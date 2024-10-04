import os
import cv2
from albert_utils import mouse_metadata

img_path = os.path.join(".", "assets", "birds-silhouette.jpg")
original_img = cv2.imread(img_path)
# <class 'numpy.ndarray'>
# print(type(original_img)) 

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.namedWindow("original_img", cv2.WINDOW_NORMAL)
cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
cv2.resizeWindow("original_img", (1000, 700))
cv2.resizeWindow("thresh", (1000, 700))

cv2.imshow("original_img", original_img)

cv2.setMouseCallback(
    "original_img",
    mouse_metadata.display_metadata,
    {"frame": "original_img", "colorspace": mouse_metadata.BGR, "img": original_img},
)

cv2.waitKey(0)
cv2.destroyAllWindows()
