import cv2 as cv
import os

img = cv.imread(os.path.join(".", "assets", "faces.jpg"))
img = cv.imread(r"C:\Users\Albert\Documents\A_Programacao\_GITIGNORE\Learning-Python\OpenCV\assets\face_training\Winston_Churchill\6454f692a232c5bba72257f0_democracia-churchill.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier(
    os.path.join(".", "assets", "xml", "haarcascade_frontalface_alt.xml")
)

"""
>> scaleFactor -> At each step, the image is scaled down by the scaleFactor, and object 
detection is attempted on the resized image. The process is repeated until the image 
becomes too small for further scaling. A value like 1.1 means that the image is scaled 
down by 10% at each iteration (i.e., the detection window is 90% of its size in the previous scale).
>> minNeighbors -> Specifies how many neighboring rectangles (detections) are needed to 
consider an object detected.
"""
# haar cascade is really sensitive to noise
face_rects = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3)

for x, y, w, h in face_rects:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

cv.imshow("img", img)

cv.waitKey()
cv.destroyAllWindows()
