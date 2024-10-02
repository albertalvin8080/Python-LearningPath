# import os
import cv2

# Use the number which corresponds to your webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame, 1)
    cv2.imshow("webcam", flipped_frame)

    # `& 0xFF` is a mask for getting only the first 8 bits (ASCII) for
    # comparing it with the ASCII code of "q".
    if cv2.waitKey(40) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
