from ultralytics import YOLO
import cv2

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


class SuspiciousDetector:
    def __init__(self, model_path, iou_thresh=0.3, score_thresh=0.5):
        self.model = YOLO(model_path)
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh

    def detect(self, frame):
        suspicious_results = self.model(frame)
        detections = []
        for result in suspicious_results:
            names = result.names
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if score > self.score_thresh:
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "score": round(score, 2),
                        "label": names[int(class_id)]
                    })
        return detections

    def match_with_tracks(self, suspicious_detections, tracks):
        matched = []
        unmatched = []

        for suspicious in suspicious_detections:
            match_found = False
            for track in tracks:
                if compute_iou(track.bbox, suspicious["bbox"]) > self.iou_thresh:
                    matched.append((track, suspicious))
                    match_found = True
                    break
            if not match_found:
                unmatched.append(suspicious)

        return matched, unmatched
