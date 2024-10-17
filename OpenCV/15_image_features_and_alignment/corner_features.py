import cv2 as cv
import os
import numpy as np

"""
>> qualityLevel -> Used for characterizing the minimum quality of image corners.
The corner feature with the highest score value is multiplied by this parameter 
and the result becomes the threshold for filtering other corners.
"""
feature_params = dict(
    maxCorners=500,
    qualityLevel=0.2,
    minDistance=15,
    blockSize=9
)

# img = cv.imread(os.path.join("." , "assets", "cocacola.png"))
# img = cv.imread(os.path.join("." , "assets", "colorful.png"))
img = cv.imread(os.path.join("." , "assets", "corner.png"))
img = cv.resize(img, (640,480), interpolation=cv.INTER_AREA)

# did this actually do something?
blur = cv.GaussianBlur(img, (5,5), 0)

gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray, **feature_params)
if corners is not None:
    """
    >> -1: This tells NumPy to automatically calculate the appropriate number of rows based on the total 
    number of elements and the other specified dimension (which is 2 in this case). 
    >>  2: This specifies that each row of the new array should contain 2 elements.
    """
    # WARNING: you must cast to np.uint32 or int, otherwise there will be data loss and the corner features
    # will be displayed at wrong places in the image.
    # Beware: we're dealing with image COORDINATES, not with color channels.
    # for x, y in np.float32(corners).reshape(-1, 2).astype(dtype=np.uint32):
    for x, y in np.uint32(corners).reshape(-1, 2):
        print(f"x{x}, y{y}")
        cv.circle(img, (x, y), 10, (0, 255, 0), 1)

cv.imshow("img", img)

cv.waitKey()
cv.destroyAllWindows()