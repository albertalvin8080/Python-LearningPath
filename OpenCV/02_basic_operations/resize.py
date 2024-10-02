import os
import cv2

img_path = os.path.join(".", "assets", "fox.jpg")

img = cv2.imread(img_path)
# img.resize((480, 320)) # Occurs inplace
resize_img = cv2.resize(img, (480, 320))

cv2.imshow("fox", img)
cv2.imshow("resize_fox", resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
