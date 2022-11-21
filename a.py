import cv2
import time
import numpy as np

def tu(a):
    return tuple((int(a[1]), int(a[0])))

maximum_strategy = " " # others are: "opencv","for loop" and "numpy"
find_reddest = False

t0 = time.time()

#cap = cv2.VideoCapture("http://192.168.42.160:8080/video")
cap = cv2.VideoCapture(0)

frame = cv2.imread("bla.bmp")
ret = True

while True:
    ret, frame = cap.read()
    #cv2.imwrite("bla.bmp", frame)
    if not ret:
        print("No frame!")
        break
    t0 = time.monotonic()

    # find maximum value
    if maximum_strategy:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        max_pos = (0, 0)
        max_val = 0
        if maximum_strategy == "for loop":
            for x in range(gray.shape[0]):
                for y in range(gray.shape[1]):
                    if gray[x, y] > max_val:
                        max_pos = (x, y)
                        max_val = gray[x,y]
        elif maximum_strategy == "numpy":
            max_pos = np.unravel_index(np.argmax(gray, axis=None), gray.shape)
        elif maximum_strategy == "opencv":
            min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(gray)
            max_pos = tu(max_indx)
        # circle maximum value with a blue circle

        cv2.circle(frame, tu(max_pos), 10, (255, 0, 0), 5)
    
    if find_reddest:
        # find the reddest pixel
        # do euclidian distance from [0, 0, 255]
        red_dist = np.linalg.norm(frame - np.array([0, 0, 255]), axis=2)
        #print(red_dist.shape)
        red_max_pos =  np.unravel_index(np.argmin(np.abs(red_dist), axis=None), red_dist.shape)
        #print(red_max_pos)

        # circle maximum redness with red circle
        cv2.circle(frame, tu(red_max_pos), 10, (0, 0, 255), 5)


    # add FPS to frame:
    t = time.monotonic()
    #print(t, t0)
    cv2.putText(frame, f"{1/(t-t0+0.00001):.1f} FPS", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

    cv2.imshow("steinarr", frame)
    
    k = cv2.waitKey(10) & 0xFF
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
    elif k != 255:
        maximum_strategy = False

