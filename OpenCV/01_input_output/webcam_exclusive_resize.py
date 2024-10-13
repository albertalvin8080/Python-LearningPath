import cv2 as cv

capture = cv.VideoCapture(0)

# (THIS ONLY WORKS FOR CAMERAS, NOT FOR STATIC VIDEOS FROM FILES)
# 3 -> width  -> cv.CAP_PROP_FRAME_WIDTH
# 4 -> height -> cv.CAP_PROP_FRAME_HEIGHT
capture.set(cv.CAP_PROP_FRAME_WIDTH, 250)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 250)

ret, frame = capture.read()
while ret:
    cv.imshow("frame", frame)
    if cv.waitKey(24) & 0xFF == ord("q"):
        break

    ret, frame = capture.read()

capture.release()
cv.destroyAllWindows()