import cv2
import os

img_path = os.path.join(".", "assets", "fox.jpg")

img = cv2.imread(img_path)

# writing the image
cv2.imwrite(os.path.join(".", "assets", "fox_out.jpg"), img)

cv2.imshow("my_fox", img)
cv2.waitKey(0)  # Necessary for preventing the program from exiting.
# cv2.waitKey(5000) # Waits 5 seconds before closing automatically.
cv2.destroyAllWindows()
