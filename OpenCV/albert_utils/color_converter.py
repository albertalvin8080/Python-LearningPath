import cv2
import numpy as np

def bgr2hsv(color: list):
    c_arr = np.array([[color]], dtype=np.uint8)
    hsv_arr = cv2.cvtColor(c_arr, cv2.COLOR_BGR2HSV)
    # print(c_arr, hsv_arr)

    hue = hsv_arr[0][0][0]

    # These two ifs (hue <= 15 and hue >= 165) are necessary because OpenCV stores
    # hue values from 0 to 180, instead of 0 to 360 for performance reasons. So the
    # red color wraps around from 165 back to 15 in the color graph.
    # -> https://i.sstatic.net/gyuw4.png
    if hue <= 15:
        lower_limit = np.array([0, 100, 100], dtype=np.uint8)
        upper_limit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    elif hue >= 165:
        lower_limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upper_limit = np.array([180, 255, 255], dtype=np.uint8)
    else:
        # Every color other than red can be parsed normally using
        # this strategy.
        lower_limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upper_limit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit