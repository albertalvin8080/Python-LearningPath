import os
import cv2
from albert_utils import mouse_metadata

img_path = os.path.join("", "assets", "fox.jpg")

img = cv2.imread(img_path) # BGR by default
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow("fox")
# cv2.imshow("fox", img)
cv2.imshow("fox_hsv", hsv_img)
cv2.imshow("fox_gray", gray_img)
cv2.imshow("fox_rgb", rgb_img)

# cv2.setMouseCallback(
#     "fox",
#     mouse_metadata.display_metadata,
#     {"img": img, "frame": "fox", "colorspace": mouse_metadata.BGR},
# )
cv2.setMouseCallback(
    "fox_hsv",
    mouse_metadata.display_metadata,
    {"img": hsv_img, "frame": "fox_hsv", "colorspace": mouse_metadata.HSV},
)
cv2.setMouseCallback(
    "fox_gray",
    mouse_metadata.display_metadata,
    {"img": gray_img, "frame": "fox_gray", "colorspace": mouse_metadata.GRAY},
)
cv2.setMouseCallback(
    "fox_rgb",
    mouse_metadata.display_metadata,
    {"img": rgb_img, "frame": "fox_rgb", "colorspace": mouse_metadata.RGB},
)


cv2.waitKey(0)
cv2.destroyAllWindows()
