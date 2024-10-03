import os
import cv2


img_path = os.path.join(".", "assets", "whiteboard.jpg")
img = cv2.imread(img_path)
img = cv2.resize(img, (1000, 800))

img = cv2.line(img, (200, 200), (450, 500), (0, 255, 0), 5)
# pt1 -> top left
# pt2 -> bottom right
img = cv2.rectangle(img, pt1=(300, 300), pt2=(700, 500), color=(0, 0, 255), thickness=5)

# print(img.shape[1]//2)
img = cv2.circle(
    img,
    center=(img.shape[1] // 2, img.shape[0] // 2),
    radius=50,
    color=(255, 0, 0),
    thickness=5,
)

cv2.imshow("whiteboard", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
