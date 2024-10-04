import os
import cv2
import mediapipe as md

video_path = os.path.join(".", "assets", "face.mp4")
video = cv2.VideoCapture(video_path)

with md.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
) as fd:
    ret, frame = video.read()
    video_writer = cv2.VideoWriter(
        os.path.join(".", "assets", "output.mp4"),
        cv2.VideoWriter_fourcc(*"MP4V"),
        x if (x := video.get(cv2.CAP_PROP_FPS)) > -1 else 0,
        (frame.shape[1], frame.shape[0]),
    )
    H, W, _ = frame.shape

    # WARNING: you need to read the frame AFTER the processing because your first
    # read occours OUTSSIDE of the while()
    while ret:
        partial_frame = frame
        # partial_frame = cv2.cvtColor(partial_frame, cv2.COLOR_BGR2GRAY)
        # partial_frame = cv2.GaussianBlur(partial_frame, (21, 21), sigmaX=0)
        # partial_frame = cv2.cvtColor(partial_frame, cv2.COLOR_GRAY2RGB)
        partial_frame = cv2.cvtColor(partial_frame, cv2.COLOR_BGR2RGB)

        out = fd.process(partial_frame)
        detections = out.detections
        if detections is not None:
            for det in detections:
                rbbx = det.location_data.relative_bounding_box
                x = int(rbbx.xmin * W)
                y = int(rbbx.ymin * H)
                w = int(rbbx.width * W)
                h = int(rbbx.height * H)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # cv2.imshow("video", frame)
        video_writer.write(frame)
        # WARNING: you need to read the frame AFTER the processing because your first
        # read occours OUTSSIDE of the while()
        ret, frame = video.read()

        # if cv2.waitKey(int(1000 / 25)) & 0xFF == ord("q"):
        # break


video.release()
cv2.destroyAllWindows()
