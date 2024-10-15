import cv2 as cv
import os

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

face_cascade = cv.CascadeClassifier(
    os.path.join(".", "assets", "xml", "haarcascade_frontalface_alt.xml")
)


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # using a numeric value for label

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            faces_rect = face_cascade.detectMultiScale(
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


create_train()

for i, img in enumerate(features):
    cv.imshow("img" + str(i), cv.resize(img, dsize=(640, 480)))
cv.waitKey()
cv.destroyAllWindows()
