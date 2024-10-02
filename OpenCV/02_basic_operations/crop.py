import cv2
import os
from albert_utils import mouse_metadata


img_path = os.path.join(".", "assets", "fox.jpg")
img = cv2.imread(img_path)
print(img.shape)
crop_img = img[67:398, 215:536]  # [y, x]

# Create a window and set the mouse callback
cv2.namedWindow("fox")
# Passing `img` as the third argument to make it accessible inside the `show_mouse_coordinates`` function.
cv2.setMouseCallback(
    "fox", mouse_metadata.display_metadata, {"img": img, "frame": "fox", "colorspace": mouse_metadata.BGR}
)

cv2.imshow("fox", img)
cv2.imshow("crop_fox", crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
