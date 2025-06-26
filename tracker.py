from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    tracker = None
    encoder = None  # This will be our OSNetExtractor instance
    tracks = None

    def __init__(self, feature_extractor):
        max_cosine_distance = 0.5  # Increased threshold for better features
        nn_budget = None
        n_init = 3  # We can return this to the default, as our new logic doesn't depend on it

        # The metric now uses OSNet features
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        # Pass the n_init parameter to the tracker
        self.tracker = DeepSortTracker(metric, n_init=n_init)
        self.encoder = feature_extractor  # Use the provided OSNet extractor
        

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        # Extract bounding boxes and scores
        bboxes_xyxy = np.asarray([d[:4] for d in detections])
        scores = [d[4] for d in detections]

        # Crop person images from the frame
        image_crops = []
        # We need to keep track of which bbox corresponds to which crop
        valid_bboxes_xyxy = []
        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            # Add a check to prevent empty crops which can cause errors
            if x1 >= x2 or y1 >= y2:
                continue
            crop = frame[y1:y2, x1:x2]
            image_crops.append(crop)
            valid_bboxes_xyxy.append(bbox)
        
        if not image_crops: # If all crops were invalid
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        # Extract features using OSNet
        features = self.encoder.extract_features(image_crops)

        # --- NEW LOGIC: Create a map from bbox to its feature ---
        # We use tuple(bbox) as a dictionary key
        self.bbox_feature_map = {
            tuple(bbox): feature 
            for bbox, feature in zip(valid_bboxes_xyxy, features)
        }

        # Convert to tlwh for DeepSORT
        bboxes_tlwh = np.array(valid_bboxes_xyxy)
        bboxes_tlwh[:, 2:] = bboxes_tlwh[:, 2:] - bboxes_tlwh[:, :2]

        # Create DeepSORT detection objects
        dets = []
        for bbox_id, bbox in enumerate(bboxes_tlwh):
            dets.append(
                Detection(bbox, scores[bbox_id], features[bbox_id])
            )
        
        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()


    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox_tlwh = track.to_tlwh()
            # Convert DeepSORT's tlwh to our original xyxy to use as a key
            # This is an approximation, but should be very close
            bbox_xyxy = track.to_tlbr()

            # --- NEW LOGIC: Find the feature from our map ---
            # Find the closest bbox from our map to the track's bbox
            # This handles minor discrepancies from Kalman filter prediction
            closest_bbox = None
            min_dist = float('inf')
            for key_bbox in self.bbox_feature_map.keys():
                dist = np.linalg.norm(np.array(key_bbox) - bbox_xyxy)
                if dist < min_dist:
                    min_dist = dist
                    closest_bbox = key_bbox
            
            # Only associate if the bbox is very close (e.g., within 10 pixels)
            if closest_bbox is not None and min_dist < 10.0:
                feature = self.bbox_feature_map[closest_bbox]
                track_id = track.track_id
                tracks.append(Track(track_id, bbox_xyxy, feature))

        self.tracks = tracks


class Track:
    def __init__(self, track_id, bbox, fearure):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = fearure  # Store the latest OSNet feature vector