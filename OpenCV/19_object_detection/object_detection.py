import cv2 as cv
import numpy as np
import os

# m -> model
m_scale = 1.0
m_size = (300, 300)
m_mean = (0, 0, 0)
m_rgb = True

def detect_objects(net, frame):
    blob = cv.dnn.blobFromImage(frame, m_scale, m_size, m_mean, swapRB=m_rgb, crop=False)
    net.setInput(blob)
    detections = net.forward()
    return detections

def display_objects(frame, labels, detections, conf_threshold = 0.25):
    width, height = frame.shape[1], frame.shape[0]
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    
    for i in range(detections.shape[2]):
        # score -> confidence
        score = detections[0, 0, i, 2]
        if score < conf_threshold:
            continue
        
        classType = int(detections[0, 0, i, 1])
        # print(classType)
        x1 = int(detections[0, 0, i, 3] * width)
        y1 = int(detections[0, 0, i, 4] * height)
        x2 = int(detections[0, 0, i, 5] * width)
        y2 = int(detections[0, 0, i, 6] * height)
        
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), FONT_THICKNESS)
        text = f"{labels[classType]}: {score:.4f}"
        text_size, baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        cv.rectangle(
            frame,
            (x1, y1 - text_size[1] - baseline),  # Adjust y-coordinate to account for text height
            (x1 + text_size[0], y1),  # Width of text box
            (0, 255, 0), 
            cv.FILLED
        )
        cv.putText(
            frame,
            text,
            (x1, y1 - baseline),  # Adjust y-coordinate to ensure text is fully visible
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),  # Change text color for better contrast
            FONT_THICKNESS
        )
        

# source: download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
model_file = os.path.join(".", "assets", "object_detection", "frozen_inference_graph.pb")
# source: https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt
config_file = os.path.join(".", "assets", "object_detection", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
# source https://gist.github.com/aallan/fbdf008cffd1e08a619ad11a02b74fa8#file-coco_labels-txt
# (there are some missing labels)
classes_file = os.path.join(".", "assets", "object_detection", "coco_labels.txt")

labels = []
with open(classes_file, "r") as fp:
    for line in fp:
        parts = line.split()
        if len(parts) > 1:
            labels.append(parts[1]) 
print(labels, len(labels))

net = cv.dnn.readNetFromTensorflow(model_file, config_file)

# img = cv.imread(os.path.join(".", "assets", "object_detection", "p2.jpeg"))
# img = cv.resize(img, (640, 480), interpolation=cv.INTER_AREA)
# detections = detect_objects(net, img)
# display_objects(img, labels, detections)
# cv.imshow("img", img)
# cv.waitKey()

cap = cv.VideoCapture(os.path.join(".", "assets", "object_detection", "traffic.mp4"))

ret, frame = cap.read()
while ret:
    detections = detect_objects(net, frame)
    display_objects(frame, labels, detections)
    cv.imshow("frame", frame)
    
    if cv.waitKey(30) & 0xFF == ord("q"):
        break
    
    ret, frame = cap.read()

cv.destroyAllWindows()