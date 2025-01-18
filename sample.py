import cv2
import numpy as np
import torch
import os
from playsound import playsound

# Load YOLOv5 model
MODEL_PATH = 'yolov5_model.pt'  
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# Constants
VIDEO_SOURCE = 0  
ALERT_SOUND = 'alert.mp3' 
SCOOTER_CLASS = 'e-scooter'  
PERSON_CLASS = 'person'


def detect_violations(detections):
    scooters = []
    people = []

    for det in detections:
        label = det['name']
        bbox = det['bbox']
        if label == SCOOTER_CLASS:
            scooters.append(bbox)
        elif label == PERSON_CLASS:
            people.append(bbox)

    for scooter in scooters:
        count = sum(is_overlapping(scooter, person) for person in people)
        if count > 1:  # More than one person on the e-scooter
            return True
    return False

# Helper function to check if two bounding boxes overlap
def is_overlapping(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

# Process video feed
def process_video():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Unable to open video source")
        return

    print("Starting video feed...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break

        # Perform object detection
        results = model(frame)
        detections = results.pandas().xyxy[0].to_dict(orient='records')

        # Draw bounding boxes and labels
        for det in detections:
            label = det['name']
            bbox = [int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])]
            confidence = det['confidence']

            # Draw bounding box and label
            color = (0, 255, 0) if label == SCOOTER_CLASS else (255, 0, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check for violations and trigger alert
        if detect_violations(detections):
            cv2.putText(frame, "Violation Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            playsound(ALERT_SOUND)

        # Display the video feed
        cv2.imshow("Beam Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(ALERT_SOUND):
        print(f"Alert sound file not found: {ALERT_SOUND}")
    else:
        process_video()
