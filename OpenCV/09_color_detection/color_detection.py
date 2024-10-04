import cv2
from PIL import Image
from albert_utils import color_converter

target_color = (0, 255, 255)  # color in BGR
lower_limit, upper_limit = color_converter.bgr2hsv(target_color)
morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
blur_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
print(blur_kernel.shape)
sigma = 0  # standard deviation for the gaussian kernel. 0 makes opencv calculate it automatically.

webcam = cv2.VideoCapture(0)  # 0 for Webcam
fps = webcam.get(cv2.CAP_PROP_FPS)
if fps > 0:
    # miliseconds = 1000 / FPS
    wait_time = int(1000 / fps)
else:
    wait_time = 1  # Fallback in case FPS is not retrievable

while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    blured_frame = cv2.GaussianBlur(frame, blur_kernel.shape, sigma)
    # blured_frame = cv2.medianBlur(frame, blur_kernel.shape[0])
    # blured_frame = cv2.blur(frame, blur_kernel.shape)
    hsvFrame = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsvFrame, lower_limit, upper_limit)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
    _mask = Image.fromarray(mask)
    bounding_box = _mask.getbbox()

    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("webcam", frame)

    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
