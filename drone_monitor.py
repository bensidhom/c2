
"""
Drone Detection, Tracking, and Speed Estimation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import time
import os

# Load YOLOv8 model (make sure 'drone_model.pt' is trained to detect drones)
model = YOLO("best.pt")
tracker = Sort()

# Video capture
cap = cv2.VideoCapture("aerial_footage.mp4")

# Define polygon zone for drone detection (example)
polygon_zone = np.array([[50, 50], [1500, 50], [1500, 800], [50, 800]])

def draw_polygon_zone(frame, zone):
    return cv2.polylines(frame, [zone], True, (0, 255, 255), 2)

# Speed estimation setup
prev_positions = {}
fps = cap.get(cv2.CAP_PROP_FPS)
pixel_to_meter_ratio = 0.05  # Change based on your footage

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, conf=0.3)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    # Update tracker
    tracked_objects = tracker.update(boxes)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw center
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Speed estimation
        if obj_id in prev_positions:
            prev_cx, prev_cy = prev_positions[obj_id]
            dx, dy = cx - prev_cx, cy - prev_cy
            distance = np.sqrt(dx ** 2 + dy ** 2) * pixel_to_meter_ratio
            speed = distance * fps * 3.6  # m/s to km/h
            cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        prev_positions[obj_id] = (cx, cy)

    # Draw polygon zone
    frame = draw_polygon_zone(frame, polygon_zone)

    cv2.imshow("Drone Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
