import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype=np.uint8)

rectangle = cv.rectangle(blank.copy(), (30, 30), (blank.shape[1]-30, blank.shape[0]-30), 255, cv.FILLED)
circle = cv.circle(blank.copy(), (blank.shape[1]//2, blank.shape[0]//2), 200, 255, cv.FILLED)

# cv.imshow("blank", blank)
# cv.imshow("rectangle", rectangle)
# cv.imshow("circle", circle) 

cv.imshow("and", cv.bitwise_and(rectangle, circle))
cv.imshow("or", cv.bitwise_or(rectangle, circle))
cv.imshow("xor", cv.bitwise_xor(rectangle, circle))
cv.imshow("not(circle)", cv.bitwise_not(circle))

cv.waitKey(0)
cv.destroyAllWindows()