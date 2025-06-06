import os
import random

import cv2
from ultralytics import YOLO

# Assuming your tracker.py is in the same directory or accessible
from tracker import Tracker


video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Error: Could not read video frame.")
    exit()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0])) # Changed MP4V to mp4v

model = YOLO("yolov8n.pt")

tracker = Tracker()

# Generate more distinct colors if you expect many tracks
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)] # Increased range

detection_threshold = 0.5
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            # Filter for 'person' class if needed (class_id for person is usually 0 in COCO)
            # For now, we assume all detections from yolov8n.pt are relevant if above threshold
            # Or you can check result.names[int(class_id)] == 'person'
            if int(class_id) == 0 and score > detection_threshold: # Example: Only track persons
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            # Optionally, display track_id
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (colors[track_id % len(colors)]), 2)


    # Display the resulting frame
    cv2.imshow('Video Tracking', frame) # ADD THIS LINE

    cap_out.write(frame)
    ret, frame = cap.read()

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'): # ADD THIS LINE (or adjust delay)
        break

cap.release()
cap_out.release()
cv2.destroyAllWindows()