
"""
Drone Detection, Tracking, and Depth-Aware Speed Estimation (Final Fix)
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort import Sort

# Load models
model = YOLO("best.pt")
tracker = Sort()

# MiDaS depth model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Polygon zone
polygon_zone = np.array([[50, 50], [1500, 50], [1500, 800], [50, 800]])

def draw_polygon_zone(frame, zone):
    return cv2.polylines(frame, [zone], True, (0, 255, 255), 2)

def is_inside_zone(point, zone):
    return cv2.pointPolygonTest(zone, point, False) >= 0

# Start video capture
cap = cv2.VideoCapture("intel.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(input_image, (384, 384))
    input_tensor = transform(resized_image).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        depth = prediction.cpu().numpy()

    results = model.track(frame, persist=True, conf=0.3)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
    tracked_objects = tracker.update(boxes)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if not is_inside_zone((cx, cy), polygon_zone):
            continue

        depth_z = depth[cy, cx]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if obj_id in prev_positions:
            prev_cx, prev_cy, prev_z = prev_positions[obj_id]
            dx, dy, dz = cx - prev_cx, cy - prev_cy, depth_z - prev_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            speed = distance * fps
            cv2.putText(frame, f"Speed: {speed:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        prev_positions[obj_id] = (cx, cy, depth_z)

    frame = draw_polygon_zone(frame, polygon_zone)
    cv2.imshow("Drone Monitor (Depth Aware)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
