import cv2

VIDEO_PATH = r"D:\FYP\YOLO8-object_detection_deepsort\data\shooting.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Failed to open video!")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("✅ FPS:", fps)
