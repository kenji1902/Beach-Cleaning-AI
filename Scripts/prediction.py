import torch
import cv2
import numpy as np

model_path = '.model/model-v1.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

Class = {1:"Plastic Bottle",0:"Cans"}

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, size = 416)
    
    for obj in results.pred[0]:
        x1, y1, x2, y2, score, label = obj.numpy()
        print(label)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f'{Class[int(label)]} {score:2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    resized_frame = cv2.resize(frame, (1080, 720))
    cv2.imshow('YOLOv5 Object Detection', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()