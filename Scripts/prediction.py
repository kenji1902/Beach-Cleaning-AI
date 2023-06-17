import torch
import cv2
import numpy as np

# Set the path of the model
model_path = '.model/model-v1.pt'

# Load the custom AI model from the specified path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Create a dictionary mapping class indices to class labels to identify
Class = {1:"Plastic Bottle", 0:"Cans"}

# Open a webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame using the AI model
    results = model(frame, size=416)

    # Iterate over the detected objects in the first prediction
    for obj in results.pred[0]:

        # Extract the object's coordinates, score, and label
        x1, y1, x2, y2, score, label = obj.numpy()

        # Draw a rectangle around the object on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Add a text label to the rectangle indicating the object class and score
        # Score = Accuracy
        cv2.putText(frame, f'{Class[int(label)]} {score:2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Resize the frame for display
    resized_frame = cv2.resize(frame, (1080, 720))

    # Show the frame with bounding boxes and labels
    cv2.imshow('YOLOv5 Object Detection', resized_frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()