import cv2 as cv
import numpy as np


# -y -> shift up
#  y -> shift down
# -x -> shift left
#  x -> shft right
def translate(src, x, y, dst=None):
    # [[  1.   0.   x.]
    #  [  0.   1.   y.]]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (src.shape[1], src.shape[0])
    print(translation_matrix)

    # The cv2.warpAffine() function in OpenCV performs an affine transformation on an image.
    # An affine transformation is a linear mapping that preserves points, straight lines,
    # and planes. This function can be used for tasks such as translation, rotation, scaling,
    # and shearing.
    if dst:
        cv.warpAffine(src, translation_matrix, dimensions, dst)
    else:
        return cv.warpAffine(src, translation_matrix, dimensions)
