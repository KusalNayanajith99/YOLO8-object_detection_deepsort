import cv2
from ultralytics import YOLO

class SuspicionDetector:
    def __init__(self, model_path='suspicious_detector.pt', confidence_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_thresh = confidence_threshold
        self.suspicious_classes = {"arson", "fighting", "abuse", "shooting"}  # ✅ Modify as needed

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)
        suspicious_detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confs, class_ids):
                label = self.model.names[class_id]
                if label in self.suspicious_classes:
                    suspicious_detections.append({
                        "label": label,
                        "bbox": list(map(int, box)),
                        "confidence": float(conf)
                    })

        return suspicious_detections  # ✅ Return detections instead of drawing
