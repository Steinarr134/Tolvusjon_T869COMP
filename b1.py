from doctest import NORMALIZE_WHITESPACE
import cv2
import time
import numpy as np
from itertools import combinations as combos

def tu(a):
    return tuple((int(a[1]), int(a[0])))

maximum_strategy = " " # others are: "opencv","for loop" and "numpy"
find_reddest = False

t0 = time.time()

#cap = cv2.VideoCapture("http://192.168.42.160:8080/video")
#cap = cv2.VideoCapture(0)

ret = True
frame = cv2.imread("frame.bmp")

while True:
    #ret, frame = cap.read()
    if not ret:
        print("No frame!")
        break
    t0 = time.time()
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    medianed = cv2.medianBlur(blurred, 9)
    gray = cv2.cvtColor(medianed, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray", gray)
    canny = cv2.Canny(gray, 50, 200)

    cv2.imshow("Canny", canny)

    points = np.argwhere(canny)

    colored_canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    colored_canny2 = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    test = 255*np.ones(gray.shape, np.uint8)

    lines = cv2.HoughLines(canny, 1, np.pi/180, 100)

    # code taken from: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(colored_canny, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            cv2.line(test, pt1, pt2, 0, 3, cv2.LINE_AA)
    cv2.imshow("Hough", colored_canny)
    
    linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(colored_canny2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    # attempt at using homogeonous coordinates

    # h_lines = []
    # print(linesP)
    # for [(x0, y0, x1, y1)] in linesP:
    #     h_lines.append(np.cross([x0, y0, 1], [x1, y1, 1]))
    # if len(linesP) >= 4:
    #     for ls in combos(h_lines, 2):
    #         point = np.cross(ls[0], ls[1])
    #         point = point/point[2]
    #         if point[0] > 0 and point[1] > 0:
    #             cv2.circle(colored_canny2, tu(point[:2]), 5, (0, 255, 0),)
    contours, _= cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    C = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            C = c
    epsilon = 0.03*cv2.arcLength(C,True)
    approx = cv2.approxPolyDP(C,epsilon,True)
    cv2.drawContours(test, [approx], -1, (150), 4)

    corners = np.float32(approx)
    
    new_corners = np.array([
        [600, 0],
        [0, 0],
        [0, 400],
        [600, 400],
    ], np.float32)

    print(corners, new_corners)
    M = cv2.getPerspectiveTransform(corners, new_corners)
    document = cv2.warpPerspective(frame, M, (600, 400))

    t1 = time.time()

    print(f"{1/(t1-t0+0.0000001):.1f} FPS")

    cv2.imshow("document", document)

    cv2.imshow("test", test)
    
    cv2.imshow("Hough P", colored_canny2)
    
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey() & 0xFF
    if k == 27:
        break
    if k == ord('f'):
        maximum_strategy = "for loop"
        print("Strategy set to for loop")
    elif k == ord('n'):
        maximum_strategy = "numpy"
        print("Strategy set to numpy")
    elif k== ord('o'):
        maximum_strategy = "opencv"
        print("strategy set to opencv")
    elif k== ord('r'):
        find_reddest = not find_reddest
    elif k == ord('s'):
        cv2.imwrite("frame.bmp", frame)
    elif k != 255:
        maximum_strategy = False
    break
