import cv2
import time
from datetime import datetime


next_time = 0
take_ever_n_seconds = 30


cap = cv2.VideoCapture("rtsp://admin:12345@192.0.0.64:554")

while True:
    if time.time() > next_time:
        next_time = time.time() + take_ever_n_seconds

        ret, img = cap.read()

        if not ret:
            continue

        imgname = datetime.now().strftime("/mnt/usb1/RU_Parking/%Y-%m-%d %H_%M_%S.jpg")

        cv2.imwrite(imgname, img)
        cv2.imshow(imgname, img)
