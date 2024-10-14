import cv2 as cv
import os

img = cv.imread(os.path.join(".", "assets", "fox.jpg"))

"""
>> d -> Diameter of each pixel neighborhood used during filtering. A larger 
value means that farther pixels will influence each other, but it slows 
down the computation.
>> sigmaColor -> Controls the influence of pixel colors on the filtering process. 
A larger value of sigmaColor means that more colors will be mixed together, 
resulting in smoother areas with less color variation.
>> sigmaSpace -> Controls the spatial extent of the smoothing. It defines 
how far pixels can be from each other in the image to influence the blur. 
A larger value of sigmaSpace means that pixels farther from each other (spatially) 
will influence each other, leading to stronger smoothing across larger areas.
"""
bilateral = cv.bilateralFilter(img, d=15, sigmaColor=40, sigmaSpace=20)

cv.imshow("bilateral", bilateral)

cv.waitKey(0)
cv.destroyAllWindows()
