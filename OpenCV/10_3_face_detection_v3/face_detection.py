import cv2 as cv
import numpy as np
import os

prototxt = os.path.join(".", "assets", "face_detection", "deploy.prototxt")
caffemodel = os.path.join(
    ".", "assets", "face_detection", "res10_300x300_ssd_iter_140000.caffemodel"
)
# face detection model:
# - *.prototxt   -> https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detector/deploy.prototxt
# - *.caffemodel -> https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml (first entry)
net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
conf_threshold = 0.7

# model parameters (can be found inside https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml)
m_width = 300
m_height = 300
m_rgb = False
m_scale = 1.0
# Mean Subtraction: Many deep learning models benefit from preprocessing the input data to have a mean 
# of zero. This is done by subtracting the average color values (mean) of the dataset used to train the 
# model from the corresponding pixel values in the input image. The m_mean array represents these mean 
# values for the channels of the image.
m_mean = [104, 177, 123] 

video_path = os.path.join(".", "assets", "face.mp4")
cap = cv.VideoCapture(video_path)
frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
# v_fps = 25
# wait_time = int(1000 / v_fps)

ok, frame = cap.read()
while ok:
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    blob = cv.dnn.blobFromImage(
        frame, m_scale, (m_width, m_height), m_mean, swapRB=m_rgb, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    """
    detections.shape[0]: Number of images processed (usually 1 in this case since we're processing one frame at a time).
    detections.shape[1]: The number of detected objects (in this case, it's generally 1 for the face detection model).
    detections.shape[2]: Number of detections made (the number of faces detected).
    detections.shape[3]: Information for each detection, typically including the confidence score and bounding box coordinates.
    """
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue

        # print(detections[0, 0, i, 3])
        # print(detections[0, 0, i, 3] * frame_width)
        
        x_top_left = int(detections[0, 0, i, 3] * frame_width)
        y_top_left = int(detections[0, 0, i, 4] * frame_height)
        x_bottom_right = int(detections[0, 0, i, 5] * frame_width)
        y_bottom_right = int(detections[0, 0, i, 6] * frame_height)
        cv.rectangle(
            frame,
            (x_top_left, y_top_left),
            (x_bottom_right, y_bottom_right),
            (0, 255, 0),
            1,
        )
        
        label = "Confidence: %.4f" % confidence
        label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        
        # print(label_size, baseline)
        cv.rectangle(
            frame,
            (x_top_left, y_top_left - label_size[1]),
            (x_top_left + label_size[0], y_top_left + baseline),
            (0, 255, 0),
            cv.FILLED
        )
        cv.putText(
            frame, 
            label, 
            (x_top_left, y_top_left), 
            cv.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255,255,255), 
            1
        )

    t, _ = net.getPerfProfile()
    label = "inference time (ms): %.2f" % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv.imshow("face", frame)

    if cv.waitKey(30) & 0xFF == ord("q"):
        break

    ok, frame = cap.read()


cap.release()
cv.destroyAllWindows()
