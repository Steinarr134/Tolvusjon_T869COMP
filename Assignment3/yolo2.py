
import cv2
import time
from names import coco_names
import numpy as np
min_conf = 0.3
abs_conf = 0.8
n_conf = 3
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(coco_names), 3), dtype='uint8')

yolo = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")


yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)


def add_predictions(img):
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    yolo.setInput(blob)

    ln = yolo.getLayerNames()
    outputs = yolo.forward(ln)

    # print(ln, len(ln))
    # print(yolo.getUnconnectedOutLayers())
    conf = [outputs[199], outputs[226], outputs[253]]
    # print("conf", conf)
    # print(len(outputs))

    # for i in range(len(outputs)):
    #     print(i, len(outputs[i]))
    # print(img.shape)
    # rated = np.fliplr(np.argsort(conf))
    # print(rated)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]
    for output in conf:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)


    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(coco_names[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


# img = cv2.imread("horse.jpg")
# cv2.imshow("svona", img)
# cv2.waitKey()



# code taken from : https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
import threading

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        ret, frame = self.cap.retrieve()
        return ret, frame

cap = VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame!")
        break
    t0 = time.time()
    img = add_predictions(frame)
    t1 = time.time()
    print(f"{1/(t1-t0)} FPS")
    cv2.imshow("yolo", img)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
