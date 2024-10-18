import cv2 as cv
import os
import numpy as np

"""
>> people   -> folder names (and people names)
>> features -> regions of interest (ROI)
>> labels   -> indexes from people[]
"""
people = []
features = []
labels = []
DIR = r"C:\Users\Albert\Documents\A_Programacao\_GITIGNORE\Learning-Python\OpenCV\assets\face_training"

for i in os.listdir(os.path.join(".", "assets", "face_training")):
    people.append(i)

haar_cascade = cv.CascadeClassifier(
    os.path.join(".", "assets", "xml", "haarcascade_frontalface_alt.xml")
)


def create_features_and_labels():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # using a numeric value for label

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            faces_rect = haar_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=3
            )

            # this for loop will only have one iteration anyway (one face per photo)
            for x, y, w, h in faces_rect:
                #  beware of y-x order
                roi = img[y : y + h, x : x + w]
                # cv.rectangle(img, (x, y), (x + w, y + h), 255, cv.FILLED)
                # features.append(img)
                features.append(roi)
                labels.append(label)


create_features_and_labels()

# dtype="object" because features is a list() of ndarrays
features = np.array(features, dtype="object")
people = np.array(people, dtype="object")
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

out_path = [".", "14_face_detection_v2", "recognizer"]

np.save(os.path.join(*out_path, "features.npy"), features)
np.save(os.path.join(*out_path, "labels.npy"), labels)
np.save(os.path.join(*out_path, "people.npy"), people)
face_recognizer.save(os.path.join(*out_path, "face_recognizer.yml"))

# for i, img in enumerate(features):
#     cv.imshow("img" + str(i), cv.resize(img, dsize=(640, 480)))
# cv.waitKey()
# cv.destroyAllWindows()
