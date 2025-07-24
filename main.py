import os
import random
import cv2
from ultralytics import YOLO
import torch

from tracker import Tracker
from feature_extractor import OSNetExtractor
from database import DatabaseManager
from enhancement import enhance_frame_clahe


# --- Configuration ---
CAMERA_ID = "Camera_A" # Unique ID for this camera stream
MONGO_URI = "mongodb+srv://kusal:1234@cluster-cctv.7sultup.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-CCTV" # Your MongoDB connection string
VIDEO_PATH = os.path.join('.', 'data', 'people.mp4')
VIDEO_OUT_PATH = os.path.join('.', 'out.mp4')
DETECTION_THRESHOLD = 0.5

# --- Initialization ---
# Initialize all components
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

db_manager = DatabaseManager(mongo_uri=MONGO_URI)
feature_extractor = OSNetExtractor(model_name='osnet_x1_0', device=device)
# Occlusion-aware tracker with longer persistence & relinking
tracker = Tracker(
    feature_extractor=feature_extractor,
    max_age=40,         # tolerate ~1.3 s loss @30 FPS
    max_lost=120,       # keep lost tracks ~4 s for relinking
    iou_thr=0.35,
    feat_thr=0.35
)
model = YOLO("yolov8n.pt")

# Video I/O
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    exit()
cap_out = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# --- NEW: MAPPING FOR DISPLAY IDs ---
# This dictionary maps the long global_id from DB to a simple display ID
global_to_display_id_map = {}
# This counter will provide the next available simple ID (1, 2, 3...)
next_display_id = 1

# For visualization
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]
local_to_global_id_map = {}  # Maps local track_id to global_id

# --- Main Loop ---
while ret:
    # 1. Enhance the frame before detection
    enhanced_frame = enhance_frame_clahe(frame)

    # 2. Detection
    results = model(enhanced_frame)
    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0 and score > DETECTION_THRESHOLD:
                detections.append([x1, y1, x2, y2, score])
    
    # 3. Tracking (with OSNet feature extraction inside)
    tracker.update(enhanced_frame, detections)

    # 4. Database Matching and Profile Update
    for track in tracker.tracks:
        local_id = track.track_id
        feature = track.feature
        bbox = track.bbox

        # Match feature to database to get a global ID
        global_id = db_manager.match_or_create_person(feature, CAMERA_ID, bbox)

        # --- NEW LOGIC: ASSIGN OR RETRIEVE DISPLAY ID ---
        if global_id not in global_to_display_id_map:
            # This is a new person for this run, assign a simple ID
            global_to_display_id_map[global_id] = next_display_id
            next_display_id += 1
        
        # Get the simple display ID for drawing
        display_id = global_to_display_id_map[global_id]
    
        # 4. Visualization
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[display_id % len(colors)] # Use display_id for color consistency

        cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(enhanced_frame, f"Person: {display_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)
    
    # Display and save frame
    cv2.imshow('Video Tracking', enhanced_frame)
    cap_out.write(enhanced_frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cap_out.release()
cv2.destroyAllWindows()
print("Processing finished.")