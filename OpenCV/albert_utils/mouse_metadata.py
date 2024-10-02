import cv2

BGR = 1
RGB = 2
HSV = 3
GRAY = 4

def display_metadata(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = param["img"].copy()

        colorspace = param["colorspace"]
        colorspace_text = None

        cv2.putText(
            img_copy,
            f"X: {x} Y: {y}",
            (10, 30),  # origin -> (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # fontScale
            (0, 255, 0),  # color
            2,  # thickness
        )
        if colorspace == BGR:
            b, g, r = img_copy[y, x]
            colorspace_text = f"BGR: ({b}, {g}, {r})"
        elif colorspace == RGB:
            r, g, b = img_copy[y, x]
            colorspace_text = f"RGB: ({r}, {g}, {b})"
        elif colorspace == HSV:
            # Only needed if you wanted to convert the colorspace at runtime
            # hsv_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
            h, s, v = img_copy[y, x]
            colorspace_text = f"HSV: ({h}, {s}, {v})"
        elif colorspace == GRAY:
            intensity = img_copy[y, x]
            colorspace_text = f"Intensity: {intensity}"

        cv2.putText(
            img_copy,
            colorspace_text,
            (10, 60),  # origin -> (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # fontScale
            (0, 255, 0),  # color
            2,  # thickness
        )
    
        cv2.imshow(param["frame"], img_copy)
