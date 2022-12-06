import cv2
import numpy as np


'''
Find lane via Hough Transform
'''

# OpenCV read image at BGR color space ---- NOT RGB
img_bgr = cv2.imread("8.png") # 640 x 360
img_bgr = img_bgr[180:,:]

gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
canny = cv2.Canny(blur_img, 150, 255)

# Hough
_img_bgr = np.copy(img_bgr)
linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 10)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(_img_bgr, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)


# cv2.imshow("lane", img_bgr) 
cv2.imwrite("_img_bgr.png", _img_bgr) 
cv2.imwrite("canny.png", canny) 
# cv2.waitKey(0)
