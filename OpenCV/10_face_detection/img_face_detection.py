import os
import cv2
import mediapipe as mp

img_path = os.path.join(".", "assets", "face.jpg")
# img_path = os.path.join(".", "assets", "face2.jpg")
img = cv2.imread(img_path)

H, W, _ = img.shape

mp_face_detection = mp.solutions.face_detection

"""
model_selection:
    0 -> Short-range model that works best for faces within 2 meters from the camera.
    1 -> Full-range model best for faces within 5 meters. 
"""
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as fd:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = fd.process(img_rgb)
    detections = out.detections
    print(detections)

    if detections is not None:
        for det in detections:
            relative_bb = det.location_data.relative_bounding_box
            x = int(relative_bb.xmin * W)
            y = int(relative_bb.ymin * H)
            w = int(relative_bb.width * W)
            h = int(relative_bb.height * H)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)


cv2.imshow("face", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
