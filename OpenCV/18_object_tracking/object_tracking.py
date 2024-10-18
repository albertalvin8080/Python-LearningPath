import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import albert_utils.mouse_metadata as mouse_metadata


def drawRect(frame, bbox):
    (x, y, w, h) = bbox[0], bbox[1], bbox[2], bbox[3]
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


# CREATE TRACKER
# You need to manually download GOTURN model files (goturn.prototxt and goturn.caffemodel) at
# https://github.com/spmallick/learnopencv/blob/master/GOTURN/README.md
# OR
# https://github.com/Mogball/goturn-files
# tracker = cv.TrackerGOTURN_create()
tracker = cv.TrackerKCF_create()
# tracker = cv.TrackerMIL_create()

# OPEN VIDEO CAPTURE AND CREATE VIDEO WRITER
capture = cv.VideoCapture(os.path.join(".", "assets", "object_tracking", "car.mp4"))

width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
video_writer = cv.VideoWriter(
    os.path.join(".", "assets", "object_tracking", "output.avi"),
    cv.VideoWriter_fourcc(*"XVID"),
    30,
    (width, height),
)

ret_cap, frame = capture.read()
# In a real system, the bounding box would be determined programmatically.
bbox = (600, 220, 880 - 600, 360 - 220)
drawRect(frame, bbox)

ok = tracker.init(frame, bbox)

# Testing rectangle position
# while True:
#     cv.imshow("car", frame)
#     cv.setMouseCallback(
#         "car",
#         mouse_metadata.display_metadata,
#         dict(img=frame, frame="car", colorspace=mouse_metadata.BGR),
#     )
#     cv.waitKey()
#     break

# Testing
# ret_trac = True

while ret_cap:
    ok, bbox = tracker.update(frame)
    # If you forget this conditional, the file gets corrupted due to
    # drawRect trying to write a rectangle on invalid bbox data.
    if ok:
        drawRect(frame, bbox)

    video_writer.write(frame)
    ret_cap, frame = capture.read()

capture.release()
cv.destroyAllWindows()
