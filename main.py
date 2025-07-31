import os
import random
import cv2
from ultralytics import YOLO
import torch
import time

from tracker import Tracker
from feature_extractor import OSNetExtractor
from database import DatabaseManager
from suspicion_detector import SuspicionDetector
from email_alert import send_email_alert
from collections import defaultdict
from walking_deviation_detector import WalkingDeviationDetector

# --- Configuration ---
CAMERA_ID = "Camera_A" # Unique ID for this camera stream
# MONGO_URI = "mongodb+srv://kusal:1234@cluster-cctv.7sultup.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-CCTV" # Your MongoDB connection string
MONGO_URI = "mongodb+srv://dulaniruwanthika99:zxEA6iEfqb8xKCnb@cluster-cctv.cbpifgp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-CCTV"
VIDEO_PATH = os.path.join('.', 'data', 'limping.mp4')
VIDEO_OUT_PATH = os.path.join('.', 'out.mp4')
DETECTION_THRESHOLD = 0.5
# Email recipients
EMAIL_RECIPIENTS = ["dulaniruwanthika99@gmail.com"]  ### EMAIL ALERT FEATURE
# Keep track of last alert time per person + behavior
last_email_sent = defaultdict(lambda: 0)  # {(person_id, category): timestamp}

# --- Initialization ---
# Initialize all components
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

db_manager = DatabaseManager(mongo_uri=MONGO_URI)
feature_extractor = OSNetExtractor(model_name='osnet_x1_0', device=device)
tracker = Tracker(feature_extractor=feature_extractor)
model = YOLO("yolov8n.pt")
suspicion_detector = SuspicionDetector("suspicious_detector.pt")
walking_detector = WalkingDeviationDetector()

# Video I/O
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    exit()
cap_out = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# --- Display Configuration ---
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Create named window with fixed size
# cv2.namedWindow('Video Tracking', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Video Tracking', DISPLAY_WIDTH, DISPLAY_HEIGHT)

# --- NEW: MAPPING FOR DISPLAY IDs ---
# This dictionary maps the long global_id from DB to a simple display ID
global_to_display_id_map = {}
# This counter will provide the next available simple ID (1, 2, 3...)
next_display_id = 1

# For visualization
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]
local_to_global_id_map = {}  # Maps local track_id to global_id

# Create screenshots folder if it doesn't exist
os.makedirs("screenshots", exist_ok=True)  ### EMAIL ALERT FEATURE

# --- Main Loop ---
while ret:
    # 1. Detection
    results = model(frame)
    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0 and score > DETECTION_THRESHOLD:
                detections.append([x1, y1, x2, y2, score])
    
    # 2. Tracking (with OSNet feature extraction inside)
    tracker.update(frame, detections)

    # *** NEW: Enhanced Walking Deviation Analysis with Weighted Metrics System ***
    walking_deviations = {}  # Maps global_id to (deviation_category, confidence)
    
    # *** NEW: Process each tracked person for walking deviation using 5 clinical metrics ***
    for track in tracker.tracks:
        local_id = track.track_id
        feature = track.feature
        bbox = track.bbox
        
        # Get or create global ID first
        global_id = db_manager.match_or_create_person(feature, CAMERA_ID, bbox)
        
        # *** NEW: Analyze walking pattern using weighted metrics system ***
        # Metrics: Leg Asymmetry (40%), Step Consistency (20%), Lateral Stability (15%), 
        #          Joint Movement (15%), Vertical Oscillation (10%)
        deviation_category, confidence = walking_detector.detect_walking_deviation(
            frame, global_id, bbox
        )
        
        # *** NEW: Store result if deviation detected with enhanced classification ***
        if deviation_category != "normal":
            walking_deviations[global_id] = (deviation_category, confidence)
            # *** NEW: Enhanced debug output with confidence scores ***
            print(f"Walking deviation detected for person {global_id}: {deviation_category} (confidence: {confidence:.2f})")

    # --- Run suspicious activity detection ---
    suspicion_map = {}  # Maps global_id to suspicious_category
    suspicion_results = suspicion_detector.model(frame)[0]
    for box in suspicion_results.boxes:
        label = suspicion_detector.model.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # You could improve matching here using IoU and assign to nearby person
        for track in tracker.tracks:
            tbx1, tby1, tbx2, tby2 = map(int, track.bbox)
            if not (x2 < tbx1 or x1 > tbx2 or y2 < tby1 or y1 > tby2):  # crude overlap
                local_id = track.track_id
                feature = track.feature
                global_id = db_manager.match_or_create_person(feature, CAMERA_ID, track.bbox, suspicious_category=label)
                suspicion_map[global_id] = label

    # *** NEW: Enhanced combination of suspicion results with walking deviation results ***
    for global_id, (deviation_category, confidence) in walking_deviations.items():
        if global_id not in suspicion_map:
            suspicion_map[global_id] = deviation_category

    # 3. Database Matching and Profile Update
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

        # Get suspicious category (if exists from detection), default to 'normal'
        suspicious_category = suspicion_map.get(global_id, "normal")

        # --- Update person with suspicion category ---
        db_manager.update_person(global_id, feature, CAMERA_ID, bbox, suspicious_category)

        # 4. Visualization
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[display_id % len(colors)] # Use display_id for color consistency

        # --- Dynamic font scale based on box height ---
        bbox_height = y2 - y1
        font_scale = max(0.4, min(1.0, bbox_height / 100))  # scale between 0.4 and 1.0
        thickness = max(1, int(font_scale * 2))

        label = f"Person: {display_id}"  # Always show Person ID
        if suspicious_category != "normal":
            label += f" | {suspicious_category}"  # Add suspicion tag if any
            
            # *** NEW: Add confidence score for walking deviations ***
            if global_id in walking_deviations:
                _, confidence = walking_deviations[global_id]
                label += f" ({confidence:.2f})"
        
        # Now get size of text box (after label is defined and with correct font constant)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Email alert logic
        if suspicious_category != "normal":
            current_time = time.time()
            cooldown_key = (display_id, suspicious_category)
            time_since_last_alert = current_time - last_email_sent[cooldown_key]
            
            if time_since_last_alert > 600:  # 600 seconds = 10 minutes
                # Make a copy of the frame with bounding box and label
                alert_frame = frame.copy()
                cv2.rectangle(alert_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(alert_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
                # Save the alert frame with bounding box
                screenshot_filename = f"screenshot_{display_id}_{int(current_time)}.png"
                screenshot_path = os.path.join("screenshots", screenshot_filename)
                cv2.imwrite(screenshot_path, alert_frame)
                
                # Send email alert
                send_email_alert(
                    suspicious_category=suspicious_category,
                    person_id=display_id,
                    camera_id=CAMERA_ID,
                    screenshot_path=screenshot_path,
                    to_emails=EMAIL_RECIPIENTS
                )
                
                # Update the last sent time
                last_email_sent[cooldown_key] = current_time
                print(f"Alert sent for Person {display_id}: {suspicious_category}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Display and save frame
    cv2.imshow('Video Tracking', frame)
    cap_out.write(frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cap_out.release()
cv2.destroyAllWindows()
print("Processing finished.")