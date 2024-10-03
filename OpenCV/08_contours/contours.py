import os
import cv2

img_path = os.path.join(".", "assets", "birds-silhouette.jpg")

original_img = cv2.imread(img_path)
gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
"""
>> contours: 
    This is a Python list containing all the contours found in the image. 
    Each contour is represented as a list of points that form a boundary 
    around an object in the image. Contours are useful for shape analysis, 
    object detection, and recognition.
>> hierarchy: 
    This variable contains information about the relationship 
    between the contours. It is a list where each element describes the 
    parent-child relationships between contours, especially in cases where 
    there are nested or enclosed objects (like a contour within another contour).
    The hierarchy is useful when you're dealing with complex shapes that have 
    multiple levels of contours.
>> cv2.RETR_TREE: 
    retrieves all the contours and reconstructs the full hierarchy of nested contours.
    This mode not only retrieves the outer contours but also includes child contours
    (contours inside other contours), creating a complete tree-like structure.
>> cv2.CHAIN_APPROX_SIMPLE:
    compresses horizontal, vertical, and diagonal segments of the contour and leaves 
    only the essential points. This reduces the number of points that need to be 
    stored for straight segments, significantly simplifying the contour data.
"""
# Depending on the OpenCV version, this may return more than two values
contours, hierarchy = cv2.findContours(
    thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    # print(cnt) # You don't want to print this, believe me.
    print(cv2.contourArea(cnt))
    # You probably don't want too small objects
    # (beware that some birds will not be contoured)
    if cv2.contourArea(cnt) > 100:
        # Drawing the contours in the original image
        # (dont try to draw in the GRAYSCALE image, you will be surprised)
        cv2.drawContours(original_img, cnt, -1, (0, 255, 0), 2, hierarchy=hierarchy)

cv2.namedWindow("original", cv2.WINDOW_NORMAL)
# cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
cv2.resizeWindow("original", (1000, 700))
# cv2.resizeWindow("gray", (1000, 700))
cv2.resizeWindow("thresh", (1000, 700))

# cv2.imshow("gray", gray)
cv2.imshow("original", original_img)
cv2.imshow("thresh", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
