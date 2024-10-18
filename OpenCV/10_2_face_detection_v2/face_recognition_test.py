import cv2 as cv
import os
import numpy as np

people = np.load(
    os.path.join(".", "14_face_detection_v2", "recognizer", "people.npy"),
    allow_pickle=True,
)
# features = np.load(os.path.join(".", "14_face_detection_v2", "recognizer", "features.npy"), allow_pickle=True)
# labels = np.load(os.path.join(".", "14_face_detection_v2", "recognizer", "labels.npy"))
# print(features)
# print("-"*50)
# print(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(
    os.path.join(".", "14_face_detection_v2", "recognizer", "face_recognizer.yml")
)

path = r"C:\Users\Albert\Documents\A_Programacao\_GITIGNORE\Learning-Python\OpenCV\assets\face_training\guesses\1.jpg"

img = cv.imread(path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier(
    os.path.join(".", "assets", "xml", "haarcascade_frontalface_alt.xml")
)
rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

for x, y, w, h in rects:
    roi = gray[y : y + h, x : x + w]
    label, confidence = face_recognizer.predict(roi)
    # if confidence comes out as 0 it may be because the image you're using for
    # the prediction is one of the same images used for training.
    print(f"label: {people[label]}, confidence: {confidence}")

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv.imshow("img", img)

cv.waitKey()
cv.destroyAllWindows()
