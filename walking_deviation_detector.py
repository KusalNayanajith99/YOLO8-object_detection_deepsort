import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from collections import deque, defaultdict
import cv2
import math

class WalkingDeviationDetector:
    def __init__(self, history_length=15, deviation_threshold=0.1):  # *** FIXED: Lower threshold ***
        """
        Initialize walking deviation detector with optimized parameters
        
        Args:
            history_length: Reduced for faster response (20→15)
            deviation_threshold: Lowered for better sensitivity (0.3→0.15)
        """
        self.history_length = history_length
        self.deviation_threshold = deviation_threshold
        
        # *** FIXED: Adjusted weights and added adaptive thresholds ***
        self.metric_weights = {
            'leg_asymmetry': 0.35,      # Reduced from 40% (too dominant)
            'step_consistency': 0.25,   # Increased from 20% (more important)
            'lateral_stability': 0.20,  # Increased from 15% (more reliable)
            'joint_movement': 0.15,     # Same (15%)
            'vertical_oscillation': 0.05 # Reduced from 10% (least reliable)
        }
        
        # *** FIXED: More realistic clinical thresholds ***
        self.clinical_thresholds = {
            'leg_asymmetry_max': 15.0,      # Reduced from 30° (more sensitive)
            'step_consistency_cv': 0.15,    # Reduced from 0.2 (more sensitive)
            'lateral_stability_ratio': 0.3, # Reduced from 0.5 (more sensitive)
            'joint_movement_max': 45.0,     # Reduced from 180° (more realistic)
            'vertical_oscillation_cv': 0.05  # Increased from 0.03 (less sensitive)
        }
        
        # *** NEW: Adaptive thresholds based on data quality ***
        self.adaptive_thresholds = {
            'min_keypoint_confidence': 0.3,  # Minimum pose confidence
            'min_frames_for_analysis': 8,    # Reduced from 10
            'stability_window': 5            # Frames to check for stability
        }
        
        # Store pose history for each person
        self.pose_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # *** NEW: Store raw measurements for debugging ***
        self.debug_history = defaultdict(list)
        
        # Store gait parameters for each person
        self.gait_parameters = defaultdict(dict)
        
        # Initialize pose model
        try:
            from ultralytics import YOLO
            print("Loading YOLOv8-Pose model...")
            self.pose_model = YOLO('yolov8n-pose.pt')
            print("YOLOv8-Pose model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load YOLOv8-Pose model: {e}")
            self.pose_model = None
        
        # Define key pose connections for walking analysis
        self.key_connections = [
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
            (5, 11), (6, 12),    # Hip connections
            (11, 12)             # Hip to hip
        ]
        
        # YOLO pose keypoint indices
        self.POSE_KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }

    def detect_walking_deviation(self, frame, person_id, bbox):
        """Main function to detect walking deviations for a person"""
        try:
            keypoints = self.extract_pose_keypoints_from_frame(frame, bbox)
            if keypoints is None:
                return "normal", 0.0

            current_features = self.extract_pose_features(keypoints)
            if current_features is None:
                return "normal", 0.0

            # Add current features to history
            self.pose_history[person_id].append(current_features)

            # *** FIXED: Reduced minimum frames requirement ***
            if len(self.pose_history[person_id]) < self.adaptive_thresholds['min_frames_for_analysis']:
                return "normal", 0.0

            # Calculate weighted deviation score
            weighted_score, metric_scores = self.calculate_weighted_deviation_score(person_id)
            
            # *** NEW: Debug logging for troubleshooting ***
            if weighted_score > 0.05:  # Only log significant scores
                print(f"Person {person_id}: Score={weighted_score:.3f}, Metrics={metric_scores}")
            
            # Determine if there's a deviation and classify it
            is_deviation = weighted_score > self.deviation_threshold
            
            if is_deviation:
                # Classify the type of deviation based on dominant metric
                category = self.classify_deviation_type(metric_scores, weighted_score)
                confidence = min(weighted_score * 2, 0.95)  # *** FIXED: Better confidence scaling ***
                
                return category, confidence
            else:
                return "normal", max(0.1, 1.0 - weighted_score * 3)  # *** FIXED: Better normal confidence ***

        except Exception as e:
            print(f"Error in walking deviation detection: {e}")
            import traceback
            traceback.print_exc()
            return "normal", 0.0

    def calculate_weighted_deviation_score(self, person_id):
        """Calculate weighted deviation score using the 5 key metrics"""
        history = list(self.pose_history[person_id])
        
        # Calculate each metric score (0.0 = normal, 1.0 = maximum deviation)
        metric_scores = {
            'leg_asymmetry': self.calculate_leg_asymmetry(history),
            'step_consistency': self.calculate_step_consistency(history),
            'lateral_stability': self.calculate_lateral_stability(history),
            'joint_movement': self.calculate_joint_movement(history),
            'vertical_oscillation': self.calculate_vertical_oscillation(history)
        }
        
        # *** NEW: Store debug info ***
        self.debug_history[person_id].append(metric_scores.copy())
        
        # Calculate weighted score
        weighted_score = 0.0
        for metric, score in metric_scores.items():
            weighted_score += score * self.metric_weights[metric]
        
        return weighted_score, metric_scores

    def calculate_leg_asymmetry(self, history):
        """*** IMPROVED: Better leg asymmetry calculation ***"""
        left_knee_angles = []
        right_knee_angles = []
        left_ankle_heights = []
        right_ankle_heights = []
        
        for frame in history:
            # Collect knee angles
            if all(key in frame for key in ['left_knee_angle', 'right_knee_angle']):
                left_knee_angles.append(frame['left_knee_angle'])
                right_knee_angles.append(frame['right_knee_angle'])
            
            # Collect ankle heights for step asymmetry
            if all(key in frame for key in ['left_ankle_height', 'right_ankle_height']):
                left_ankle_heights.append(frame['left_ankle_height'])
                right_ankle_heights.append(frame['right_ankle_height'])
        
        if len(left_knee_angles) < 5:
            return 0.0
        
        # *** IMPROVED: Multiple asymmetry measures ***
        asymmetry_score = 0.0
        
        # 1. Knee angle asymmetry
        knee_asymmetries = [abs(l - r) for l, r in zip(left_knee_angles, right_knee_angles)]
        avg_knee_asymmetry = np.mean(knee_asymmetries)
        knee_score = min(avg_knee_asymmetry / self.clinical_thresholds['leg_asymmetry_max'], 1.0)
        asymmetry_score += knee_score * 0.6
        
        # 2. Ankle height pattern asymmetry
        if len(left_ankle_heights) >= 5:
            left_variation = np.std(left_ankle_heights)
            right_variation = np.std(right_ankle_heights)
            if left_variation + right_variation > 0:
                height_asymmetry = abs(left_variation - right_variation) / (left_variation + right_variation)
                asymmetry_score += min(height_asymmetry, 1.0) * 0.4
        
        return min(asymmetry_score, 1.0)

    def calculate_step_consistency(self, history):
        """*** FIXED: Improved step consistency calculation ***"""
        if len(history) < 8:
            return 0.0
        
        step_widths = []
        ankle_movements = []
        hip_movements = []
        
        for i, frame in enumerate(history):
            if 'step_width' in frame and frame['step_width'] > 0:
                step_widths.append(frame['step_width'])
            
            # *** NEW: Track movement patterns ***
            if i > 0 and 'hip_center_y' in frame:
                prev_frame = history[i-1]
                if 'hip_center_y' in prev_frame:
                    hip_movement = abs(frame['hip_center_y'] - prev_frame['hip_center_y'])
                    hip_movements.append(hip_movement)
            
            # Track ankle movement for step timing
            if (i > 0 and 'left_ankle_height' in frame and 'right_ankle_height' in frame):
                prev_frame = history[i-1]
                if 'left_ankle_height' in prev_frame and 'right_ankle_height' in prev_frame:
                    left_move = abs(frame['left_ankle_height'] - prev_frame['left_ankle_height'])
                    right_move = abs(frame['right_ankle_height'] - prev_frame['right_ankle_height'])
                    ankle_movements.append(left_move + right_move)
        
        if len(step_widths) < 3:
            return 0.0
        
        # Calculate multiple consistency measures
        consistency_score = 0.0
        
        # 1. Step width consistency
        if len(step_widths) >= 3:
            step_cv = np.std(step_widths) / (np.mean(step_widths) + 1e-6)
            step_score = min(step_cv / self.clinical_thresholds['step_consistency_cv'], 1.0)
            consistency_score += step_score * 0.5
        
        # 2. Movement pattern consistency
        if len(ankle_movements) >= 3:
            ankle_cv = np.std(ankle_movements) / (np.mean(ankle_movements) + 1e-6)
            ankle_score = min(ankle_cv / 0.5, 1.0)  # Threshold for ankle movement variation
            consistency_score += ankle_score * 0.3
        
        # 3. Hip movement consistency
        if len(hip_movements) >= 3:
            hip_cv = np.std(hip_movements) / (np.mean(hip_movements) + 1e-6)
            hip_score = min(hip_cv / 0.3, 1.0)  # Threshold for hip movement variation
            consistency_score += hip_score * 0.2
        
        return min(consistency_score, 1.0)

    def calculate_lateral_stability(self, history):
        """*** IMPROVED: Better lateral stability calculation ***"""
        head_deviations = []
        hip_widths = []
        
        for frame in history:
            if all(key in frame for key in ['nose_x', 'hip_center_x', 'hip_width']):
                if frame['hip_width'] > 0:  # Valid measurement
                    # Calculate lateral deviation of head from hip center
                    head_deviation = abs(frame['nose_x'] - frame['hip_center_x'])
                    normalized_deviation = head_deviation / frame['hip_width']
                    head_deviations.append(normalized_deviation)
                    hip_widths.append(frame['hip_width'])
        
        if len(head_deviations) < 5:
            return 0.0
        
        # Calculate stability metrics
        avg_deviation = np.mean(head_deviations)
        deviation_variability = np.std(head_deviations)
        
        # Combine average deviation and variability
        stability_score = (avg_deviation / self.clinical_thresholds['lateral_stability_ratio']) * 0.7
        stability_score += min(deviation_variability / 0.2, 1.0) * 0.3
        
        return min(stability_score, 1.0)

    def calculate_joint_movement(self, history):
        """*** IMPROVED: Better joint movement calculation ***"""
        left_knee_angles = []
        right_knee_angles = []
        
        for frame in history:
            if all(key in frame for key in ['left_knee_angle', 'right_knee_angle']):
                # Filter out unrealistic angles
                if 30 <= frame['left_knee_angle'] <= 180 and 30 <= frame['right_knee_angle'] <= 180:
                    left_knee_angles.append(frame['left_knee_angle'])
                    right_knee_angles.append(frame['right_knee_angle'])
        
        if len(left_knee_angles) < 8:
            return 0.0
        
        # Calculate range of motion for each leg
        left_rom = max(left_knee_angles) - min(left_knee_angles)
        right_rom = max(right_knee_angles) - min(right_knee_angles)
        
        # Multiple joint movement indicators
        joint_score = 0.0
        
        # 1. ROM asymmetry
        rom_asymmetry = abs(left_rom - right_rom)
        asymmetry_score = min(rom_asymmetry / self.clinical_thresholds['joint_movement_max'], 1.0)
        joint_score += asymmetry_score * 0.6
        
        # 2. Overall ROM restriction
        avg_rom = (left_rom + right_rom) / 2
        if avg_rom < 20:  # Very restricted movement
            restriction_score = (20 - avg_rom) / 20
            joint_score += restriction_score * 0.4
        
        return min(joint_score, 1.0)

    def calculate_vertical_oscillation(self, history):
        """*** IMPROVED: Better vertical oscillation calculation ***"""
        hip_center_heights = []
        
        for frame in history:
            if 'hip_center_y' in frame:
                hip_center_heights.append(frame['hip_center_y'])
        
        if len(hip_center_heights) < 8:
            return 0.0
        
        # Calculate vertical movement variability
        height_cv = np.std(hip_center_heights) / (np.mean(hip_center_heights) + 1e-6)
        
        # Normalize by clinical threshold
        oscillation_score = min(height_cv / self.clinical_thresholds['vertical_oscillation_cv'], 1.0)
        
        return oscillation_score

    def classify_deviation_type(self, metric_scores, weighted_score):
        """*** IMPROVED: Better classification with multiple criteria ***"""
        # Sort metrics by score
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_metric, dominant_score = sorted_metrics[0]
        
        # *** IMPROVED: Multi-metric classification with thresholds ***
        
        # Strong single metric dominance
        if dominant_score > 0.4:
            if dominant_metric == 'leg_asymmetry':
                return "asymmetric_gait"
            elif dominant_metric == 'step_consistency':
                return "irregular_gait"
            elif dominant_metric == 'lateral_stability':
                return "balance_issues"
            elif dominant_metric == 'joint_movement':
                return "limping"
            elif dominant_metric == 'vertical_oscillation':
                return "bouncing_gait"
        
        # Mixed patterns - look at combinations
        if metric_scores['leg_asymmetry'] > 0.2 and metric_scores['joint_movement'] > 0.2:
            return "limping"
        elif metric_scores['lateral_stability'] > 0.2 and metric_scores['step_consistency'] > 0.2:
            return "balance_issues"
        elif metric_scores['step_consistency'] > 0.25:
            return "irregular_gait"
        elif metric_scores['leg_asymmetry'] > 0.25:
            return "asymmetric_gait"
        
        # Default for mild deviations
        return "walking_deviation"

    def extract_pose_keypoints_from_frame(self, frame, bbox):
        """Extract pose keypoints from frame using YOLOv8-Pose"""
        try:
            if self.pose_model is None:
                return None
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid bounding box
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            if x1 >= x2 or y1 >= y2:
                return None
                
            # Crop person region
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
                
            # Run pose estimation
            results = self.pose_model(person_crop, verbose=False)
            
            if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
                return None
                
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Convert to the format expected by existing methods
            formatted_keypoints = []
            confidences = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else [1.0] * len(keypoints)
            
            for i, (point, conf) in enumerate(zip(keypoints, confidences)):
                # Adjust coordinates back to original frame
                adjusted_x = point[0] + x1
                adjusted_y = point[1] + y1
                formatted_keypoints.append([adjusted_x, adjusted_y, conf])
            
            return formatted_keypoints
            
        except Exception as e:
            print(f"Error in pose extraction: {e}")
            return None

    def extract_pose_features(self, keypoints):
        """*** IMPROVED: Better feature extraction with validation ***"""
        if len(keypoints) < 17:
            return None
            
        features = {}
        
        # Get key points with confidence check
        def get_point_if_confident(kp_name, min_conf=None):
            if min_conf is None:
                min_conf = self.adaptive_thresholds['min_keypoint_confidence']
            kp = keypoints[self.POSE_KEYPOINTS[kp_name]]
            return kp if kp[2] > min_conf else None
        
        # Essential points
        nose = get_point_if_confident('nose', 0.2)  # Lower confidence for head
        left_hip = get_point_if_confident('left_hip')
        right_hip = get_point_if_confident('right_hip')
        left_knee = get_point_if_confident('left_knee')
        right_knee = get_point_if_confident('right_knee')
        left_ankle = get_point_if_confident('left_ankle')
        right_ankle = get_point_if_confident('right_ankle')
        
        # Check if we have minimum required points
        if not all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            return None
        
        # Basic measurements
        features['hip_center_x'] = (left_hip[0] + right_hip[0]) / 2
        features['hip_center_y'] = (left_hip[1] + right_hip[1]) / 2
        features['hip_width'] = euclidean(left_hip[:2], right_hip[:2])
        features['step_width'] = abs(left_ankle[0] - right_ankle[0])
        
        # Ankle heights
        features['left_ankle_height'] = left_ankle[1]
        features['right_ankle_height'] = right_ankle[1]
        
        # Head position (for lateral stability)
        if nose:
            features['nose_x'] = nose[0]
            features['nose_y'] = nose[1]
        
        # Calculate joint angles with validation
        try:
            # Left knee angle
            left_angle = self.calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
            if 30 <= left_angle <= 180:  # Reasonable range
                features['left_knee_angle'] = left_angle
            
            # Right knee angle
            right_angle = self.calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
            if 30 <= right_angle <= 180:  # Reasonable range
                features['right_knee_angle'] = right_angle
            
            # Hip angles (using shoulders if available)
            left_shoulder = get_point_if_confident('left_shoulder', 0.2)
            right_shoulder = get_point_if_confident('right_shoulder', 0.2)
            
            if left_shoulder:
                left_hip_angle = self.calculate_angle(left_shoulder[:2], left_hip[:2], left_knee[:2])
                if 45 <= left_hip_angle <= 135:  # Reasonable range
                    features['left_hip_angle'] = left_hip_angle
            
            if right_shoulder:
                right_hip_angle = self.calculate_angle(right_shoulder[:2], right_hip[:2], right_knee[:2])
                if 45 <= right_hip_angle <= 135:  # Reasonable range
                    features['right_hip_angle'] = right_hip_angle
                
        except Exception as e:
            print(f"Error calculating angles: {e}")
        
        return features if features else None
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)
        
        # Avoid division by zero
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 90.0  # Default angle
        
        cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)

    # Legacy methods for compatibility (kept but not used in new weighted system)
    def analyze_gait_pattern(self, person_id, current_features):
        """Legacy method - now uses weighted scoring system"""
        weighted_score, metric_scores = self.calculate_weighted_deviation_score(person_id)
        is_deviation = weighted_score > self.deviation_threshold
        
        if is_deviation:
            category = self.classify_deviation_type(metric_scores, weighted_score)
            return True, category
        else:
            return False, "normal"