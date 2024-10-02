import cv2
import os


# Mouse callback function to get the coordinates
def show_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Copy to avoid permanently modifying the original image each time the coordinates are updated.
        img_copy = param.copy()
        # Put the coordinates on the image.
        cv2.putText(
            img_copy,
            f"X: {x} Y: {y}",
            (10, 30),  # origin -> (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # fontScale
            (0, 255, 0),  # color
            2,  # thickness
        )
        cv2.imshow("fox", img_copy)


img_path = os.path.join(".", "assets", "fox.jpg")
img = cv2.imread(img_path)
print(img.shape)
crop_img = img[67:398, 215:536]  # [y, x]

# Create a window and set the mouse callback
cv2.namedWindow("fox")
# Passing `img` as the third argument to make it accessible inside the `show_mouse_coordinates`` function.
cv2.setMouseCallback("fox", show_mouse_coordinates, img)

cv2.imshow("fox", img)
cv2.imshow("crop_fox", crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
