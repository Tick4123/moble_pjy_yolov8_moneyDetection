import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("test3.mp4")
model = YOLO("Yolo-Weights/yolov8l_korea_banknote_v2.pt")
classNames = [
    "50000_B",
    "50000_F",
    "5000_B",
    "5000_F",
    "10000_B",
    "10000_F",
    "1000_B",
    "1000_F",
]

model2 = YOLO("Yolo-Weights/yolov8l_korea_coin_v3.pt")
classNames2 = ["50_B", "50_F", "500_B", "500_F", "100_B", "100_F", "10_B", "10_F"]

while True:
    moneySum = 0
    success, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    results1 = model(
        img,
        stream=True,
    )
    results2 = model2(
        img,
        stream=True,
    )

    for r in results1:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf >= 0.5:
                cvzone.cornerRect(
                    img,
                    (x1, y1, w, h),
                    l=0,
                    t=1,
                    rt=2,
                    colorR=(0, 240, 0),
                    colorC=(0, 240, 0),
                )
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1 + 8), max(35, y1 - 5)),
                    scale=0.7,
                    thickness=2,
                    colorT=(255, 255, 255),
                    colorR=(0, 240, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                )
                if cls == 0 or cls == 1:
                    moneySum += 50000
                elif cls == 2 or cls == 3:
                    moneySum += 5000
                elif cls == 4 or cls == 5:
                    moneySum += 10000
                elif cls == 6 or cls == 7:
                    moneySum += 1000

    for r in results2:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames2[cls]

            if conf >= 0.5:
                cvzone.cornerRect(
                    img,
                    (x1, y1, w, h),
                    l=0,
                    t=1,
                    rt=2,
                    colorR=(0, 240, 0),
                    colorC=(0, 240, 0),
                )
                cvzone.putTextRect(
                    img,
                    f"{classNames2[cls]} {conf}",
                    (max(0, x1 + 8), max(35, y1 - 5)),
                    scale=1,
                    thickness=2,
                    colorT=(255, 255, 255),
                    colorR=(0, 240, 0),
                )
                if cls == 0 or cls == 1:
                    moneySum += 100
                elif cls == 2 or cls == 3:
                    moneySum += 10
                elif cls == 4 or cls == 5:
                    moneySum += 500
                elif cls == 6 or cls == 7:
                    moneySum += 50
    h, w, c = img.shape
    cv2.putText(
        img,
        str(moneySum) + " won",
        (w - 300, h - 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 255, 0),
        3,
    )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
