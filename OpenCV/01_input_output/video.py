import os
import cv2

video_path = os.path.join(".", "assets", "lion.mp4")

video = cv2.VideoCapture(video_path)
# window size
cv2.namedWindow("my_lion", cv2.WINDOW_NORMAL)
cv2.resizeWindow("my_lion", 1280, 720)

ret = True
while ret:
    ret, frame = video.read()
    if ret:
        cv2.imshow("my_lion", frame)
        # 1 second / 30 video frames = 0.03334 seconds (or 33/34 miliseconds)
        pressed = cv2.waitKey(34) 
        if pressed != -1: 
            break # breaks if the user presses any key

# releasing memory and other resources
video.release()
cv2.destroyAllWindows()