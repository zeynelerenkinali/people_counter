from ultralytics import YOLO
import torch
import cv2
import cvzone
import math
from sort import *
cap = cv2.VideoCapture("videos/people.mp4") # for video
if not cap.isOpened():
    print("Error: Could not open video file.")


model = YOLO("yolo_weights/yolov8l.pt")
totalCount = []
totalCount_2 = []

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#Check is cuda available
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

mask = cv2.imread("yolo/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = (440, 297, 650, 297)

limits_2 = (180, 297, 350, 297)

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    imgGraphics = cv2.imread("yolo/graphics.png", cv2.IMREAD_UNCHANGED)
    cvzone.overlayPNG(img,imgGraphics, (0,0))
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Boxs
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 3)

            w, h = x2-x1,y2-y1
            bbox = int(x1),int(y1),int(w),int(h)
            # print(x1,y1,w,h)

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100
            
            # Class Name
            cls = box.cls[0]

            currentClass = classNames[int(cls)]
            
            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (255, 0, 0), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

        w, h = x2-x1,y2-y1
        bbox = int(x1),int(y1),int(w),int(h)
        cvzone.cornerRect(img, bbox, l=5, rt=3, colorR=(255,0,0))
        cvzone.putTextRect(img, f' {int(id)}', (max(13,x1+13),max(30,y1-15)), scale=2, thickness=3, offset=10)
        print(result)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy), 5, (255,0,255), cv2.FILLED)
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if(totalCount.count(id) == 0):
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        if limits_2[0] < cx < limits_2[2] and limits_2[1] - 15 < cy < limits_2[1] + 15:
            if(totalCount_2.count(id) == 0):
                totalCount_2.append(id)
                cv2.line(img, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (0, 255, 0), 5)
    
    cv2.putText(img, str(len(totalCount_2)),(200,80), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 8)
    cv2.putText(img, str(len(totalCount)),(450,80), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 8)
    cv2.imshow("Image", img)
    cv2.waitKey(1) # when we turn waitkey to zero we can move by keyboard

