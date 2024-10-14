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


# positive angle -> anti-clockwise
# negative angle -> clockwise
def rotate(src, angle, rotation_point=None, dst=None):
    # don't mess this order up
    (height, width) = src.shape[:2]

    # rotate from the center as default
    if rotation_point is None:
        rotation_point = (width // 2, height // 2)

    # scale=1.0 because we just want to rotate, not rescale
    rotation_matrix = cv.getRotationMatrix2D(rotation_point, angle, scale=1.0)

    if dst:
        cv.warpAffine(src, rotation_matrix, (width, height), dst)
    else:
        return cv.warpAffine(src, rotation_matrix, (width, height))
