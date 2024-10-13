import cv2
import numpy as np

# img = cv2.imread("img.png")
img = cv2.imread("LeandroPlaca52-25.png")
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# -1 indicates that the depth of src and dst are the same.
cv2.filter2D(img, -1, kernel, img)
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Mat kernel = new Mat(3, 3, CvType.CV_32F);
# float[] kernelData =
# {
#     -1, -1, -1, -1, 9, -1, -1, -1, -1
# };
# kernel.put(0, 0, kernelData);
# Imgproc.filter2D(adapted, adapted, -1, kernel);