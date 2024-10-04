import os
import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

"""
model_selection:
    0 -> Short-range model that works best for faces within 2 meters from the camera.
    1 -> Full-range model best for faces within 5 meters. 
"""
with mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as fd:
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("frame", frame)

    H, W, _ = frame.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    while True:
        ret, frame = webcam.read()
        frame = cv2.flip(frame, 1)

        partial_frame = frame
        partial_frame = cv2.cvtColor(partial_frame, cv2.COLOR_BGR2GRAY)
        partial_frame = cv2.GaussianBlur(partial_frame, kernel.shape, sigmaX=0)
        # WARNING: If the primary goal is face detection, too much morphological transformation
        #   could reduce the details needed for accurate detection.
        # partial_frame = cv2.morphologyEx(partial_frame, cv2.MORPH_OPEN, kernel)
        # partial_frame = cv2.morphologyEx(partial_frame, cv2.MORPH_CLOSE, kernel)
        partial_frame = cv2.cvtColor(partial_frame, cv2.COLOR_GRAY2RGB)

        out = fd.process(partial_frame)
        detections = out.detections
        if detections is not None:
            for det in detections:
                rbbx = det.location_data.relative_bounding_box
                x = int(rbbx.xmin * W)
                y = int(rbbx.ymin * H)
                w = int(rbbx.width * W)
                h = int(rbbx.height * H)
                # Drawing rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

                # Ensure the bounding box stays within frame dimensions.
                # The program crashes when half of your face is outside
                # of the frame if you don't do this.
                x = max(0, x)
                y = max(0, y)
                w = min(W - x, w)
                h = min(H - y, h)
                blur = 100
                # Bluring only the face
                frame[y : y + h, x : x + w, :] = cv2.blur(
                    frame[y : y + h, x : x + w, :], (blur, blur)
                )

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()
