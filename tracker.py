import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

class Track:
    _count = 0
    def __init__(self, bbox, feature, kf, ttl, frame_idx=0):
        self.track_id = Track._count
        Track._count += 1
        self.kf = kf
        self.bbox = bbox
        self.feature = feature
        self.time_since_update = 0
        self.age = 0
        self.ttl = ttl
        self.last_seen_frame = frame_idx
        self.history = [bbox]

    def predict(self):
        self.kf.predict()
        x, y, w, h = self.kf.x[:4].reshape(-1)
        self.bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, feature, frame_idx=0):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        self.kf.update(np.array([cx, cy, w, h]).reshape((4, 1)))
        self.bbox = bbox
        self.feature = feature
        self.time_since_update = 0
        self.last_seen_frame = frame_idx
        self.history.append(bbox)

class Tracker:
    def __init__(self, feature_extractor, max_age=40, max_lost=120, iou_thr=0.35, feat_thr=0.35, max_dist=50, max_time=30):
        self.extractor = feature_extractor
        self.max_age = max_age
        self.max_lost = max_lost
        self.iou_thr = iou_thr
        self.feat_thr = feat_thr
        self.tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.max_dist = max_dist
        self.max_time = max_time
        self.frame_idx = 0

    def update(self, frame, detections):
        self.frame_idx += 1
        for trk in self.tracks:
            trk.predict()

        bboxes = [d[:4] for d in detections] if detections else []
        features = self.extractor.extract(frame, bboxes) if detections else []

        matches, un_det_idx, un_trk_idx = self._associate(detections, features)

        for det_idx, trk_idx in matches:
            det_bbox, feat = detections[det_idx][:4], features[det_idx]
            self.tracks[trk_idx].update(det_bbox, feat, self.frame_idx)

        newly_lost = []
        for idx in sorted(un_trk_idx, reverse=True):
            trk = self.tracks[idx]
            if trk.time_since_update > self.max_age:
                newly_lost.append(trk)
                self.tracks.pop(idx)
        self.lost_tracks.extend(newly_lost)

        relinked = set()
        for det_idx in list(un_det_idx):
            det_bbox, feat = detections[det_idx][:4], features[det_idx]
            best_match, best_score = None, 0
            for l_trk in self.lost_tracks:
                iou_score = self.iou(det_bbox, l_trk.bbox)
                feat_score = 1 - np.dot(feat, l_trk.feature)
                if (iou_score >= self.iou_thr) or (feat_score >= self.feat_thr):
                    if iou_score + feat_score > best_score:
                        best_match, best_score = l_trk, iou_score + feat_score
            if best_match is not None:
                best_match.update(det_bbox, feat, self.frame_idx)
                self.tracks.append(best_match)
                self.lost_tracks.remove(best_match)
                relinked.add(det_idx)

        for det_idx in un_det_idx:
            if det_idx in relinked:
                continue
            bbox, feat = detections[det_idx][:4], features[det_idx]
            new_trk = self._start_new_track(bbox, feat, self.frame_idx)
            self.tracks.append(new_trk)

        self._trajectory_relink()
        self._prune_lost_tracks()

    def _trajectory_relink(self):
        new_tracks = [trk for trk in self.tracks if trk.age <= 2]
        for new_trk in new_tracks:
            new_center = bbox_center(new_trk.bbox)
            for l_trk in self.lost_tracks:
                time_diff = self.frame_idx - l_trk.last_seen_frame
                dist = np.linalg.norm(new_center - bbox_center(l_trk.bbox))
                if time_diff <= self.max_time and dist <= self.max_dist:
                    new_trk.track_id = l_trk.track_id
                    new_trk.history = l_trk.history + new_trk.history
                    self.lost_tracks.remove(l_trk)
                    break

    def _associate(self, detections, features):
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, trk in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_score = self.iou(trk.bbox, det[:4])
                feat_score = 1 - np.dot(trk.feature, features[d])
                cost_matrix[t, d] = 1 - (0.6 * iou_score + 0.4 * feat_score)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matches, unmatched_det, unmatched_trk = [], [], []
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < (1 - 0.3):
                matches.append((c, r))
            else:
                unmatched_det.append(c)
                unmatched_trk.append(r)
        unmatched_det += [d for d in range(len(detections)) if d not in col_idx]
        unmatched_trk += [t for t in range(len(self.tracks)) if t not in row_idx]
        return matches, unmatched_det, unmatched_trk

    def _start_new_track(self, bbox, feat, frame_idx):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.x[:4] = np.array([cx, cy, w, h]).reshape((4, 1))
        kf.P *= 10.
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]])
        return Track(bbox, feat, kf, self.max_lost, frame_idx)

    def _prune_lost_tracks(self):
        self.lost_tracks = [
            trk for trk in self.lost_tracks if trk.time_since_update <= self.max_lost
        ]

    def iou(self, box_a, box_b):
        xA, yA = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        xB, yB = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        int_w, int_h = max(0, xB - xA), max(0, yB - yA)
        inter = int_w * int_h
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union else 0
