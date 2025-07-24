import cv2
import numpy as np
from enhancement import enhance_frame_clahe, enhance_frame_gamma

# ====== Change this to your input video ======
input_video_path = 'data/1-2.mp4'

# ====== Set display window size (pixels) ======
window_width = 960   # Adjust this to fit your screen (e.g., 640, 800, etc.)

# ====== Start video capture ======
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

print("Press 'q' to exit the real-time viewer.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Enhance frame
    enhanced = enhance_frame_clahe(frame)
    enhanced = enhance_frame_gamma(enhanced, gamma=1.3)

    # Create side-by-side before/after
    combined = cv2.hconcat([frame, enhanced])

    # Resize for viewing
    height, width = combined.shape[:2]
    scaling_factor = window_width / width
    new_dim = (window_width, int(height * scaling_factor))
    resized = cv2.resize(combined, new_dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Real-Time Enhancement - Before (Left) | After (Right)', resized)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Real-time enhancement visualization closed.")
