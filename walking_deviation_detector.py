import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from collections import deque, defaultdict
import cv2
import math

class WalkingDeviationDetector:
    def __init__(self, history_length=12, deviation_threshold=0.08):  # *** threshold ***
        """
        Initialize walking deviation detector with very sensitive parameters
        
        Args:
            history_length: Reduced for faster response (15→12)
            deviation_threshold: Much lower for better sensitivity (0.15→0.08)
        """
        self.history_length = history_length
        self.deviation_threshold = deviation_threshold
        
        # *** REBALANCED: More even weights to prevent single metric dominance ***
        self.metric_weights = {
            'leg_asymmetry': 0.25,      # Reduced from 35%
            'step_consistency': 0.20,   # Reduced to fix the broken metric
            'lateral_stability': 0.10,  # Much reduced - it's maxing out
            'joint_movement': 0.25,     # Increased - more reliable
            'vertical_oscillation': 0.20 # Much increased - was too low
        }
        
        # *** MUCH MORE SENSITIVE: Drastically lowered thresholds ***
        self.clinical_thresholds = {
            'leg_asymmetry_max': 8.0,       # Much more sensitive (15→8)
            'step_consistency_cv': 0.05,    # Much more sensitive (0.15→0.05)
            'lateral_stability_ratio': 0.15, # Much more sensitive (0.3→0.15)
            'joint_movement_max': 25.0,     # More sensitive (45→25)
            'vertical_oscillation_cv': 0.08  # More sensitive (0.05→0.08)
        }
        
        # *** NEW: Debug mode ***
        self.debug_mode = True
        
        # Store pose history for each person
        self.pose_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # *** NEW: Store raw calculations for debugging ***
        self.metric_debug = defaultdict(list)
        
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
                if self.debug_mode:
                    print(f"Person {person_id}: No keypoints detected")
                return "normal", 0.0

            current_features = self.extract_pose_features(keypoints)
            if current_features is None:
                if self.debug_mode:
                    print(f"Person {person_id}: No features extracted")
                return "normal", 0.0

            # Add current features to history
            self.pose_history[person_id].append(current_features)

            # *** MUCH LOWER: Minimum frames requirement ***
            if len(self.pose_history[person_id]) < 5:  # Reduced from 8
                return "normal", 0.0

            # Calculate weighted deviation score
            weighted_score, metric_scores = self.calculate_weighted_deviation_score(person_id)
            
            # *** ENHANCED DEBUG: Show all calculations ***
            if self.debug_mode:
                print(f"\nPerson {person_id} Analysis:")
                print(f"  Frames: {len(self.pose_history[person_id])}")
                print(f"  Weighted Score: {weighted_score:.4f}")
                print(f"  Threshold: {self.deviation_threshold}")
                for metric, score in metric_scores.items():
                    print(f"  {metric}: {score:.4f}")
            
            # *** MUCH MORE AGGRESSIVE: Lower threshold ***
            is_deviation = weighted_score > self.deviation_threshold
            
            if is_deviation:
                category = self.classify_deviation_type(metric_scores, weighted_score)
                confidence = min(weighted_score * 5, 0.95)  # *** AGGRESSIVE: 5x confidence boost ***
                
                if self.debug_mode:
                    print(f"  DEVIATION DETECTED: {category} (confidence: {confidence:.3f})")
                
                return category, confidence
            else:
                confidence = max(0.1, 1.0 - weighted_score * 5)
                if self.debug_mode:
                    print(f"  Normal walking (confidence: {confidence:.3f})")
                return "normal", confidence

        except Exception as e:
            print(f"Error in walking deviation detection: {e}")
            import traceback
            traceback.print_exc()
            return "normal", 0.0

    def calculate_weighted_deviation_score(self, person_id):
        """Calculate weighted deviation score using the 5 key metrics"""
        history = list(self.pose_history[person_id])
        
        # Calculate each metric score
        metric_scores = {
            'leg_asymmetry': self.calculate_leg_asymmetry(history),
            'step_consistency': self.calculate_step_consistency_fixed(history),  # *** FIXED METHOD ***
            'lateral_stability': self.calculate_lateral_stability_fixed(history),  # *** FIXED METHOD ***
            'joint_movement': self.calculate_joint_movement(history),
            'vertical_oscillation': self.calculate_vertical_oscillation(history)
        }
        
        # *** DEBUG: Store calculations ***
        if self.debug_mode:
            self.metric_debug[person_id].append({
                'frame_count': len(history),
                'metrics': metric_scores.copy()
            })
        
        # Calculate weighted score
        weighted_score = 0.0
        for metric, score in metric_scores.items():
            weighted_score += score * self.metric_weights[metric]
        
        return weighted_score, metric_scores

    def calculate_step_consistency_fixed(self, history):
        """*** COMPLETELY REWRITTEN: Fix the broken step consistency calculation ***"""
        if len(history) < 5:
            return 0.0
        
        step_features = []
        
        # Collect basic step measurements
        for frame in history:
            if all(key in frame for key in ['step_width', 'left_ankle_height', 'right_ankle_height']):
                step_features.append({
                    'width': frame['step_width'],
                    'left_height': frame['left_ankle_height'],
                    'right_height': frame['right_ankle_height'],
                    'height_diff': abs(frame['left_ankle_height'] - frame['right_ankle_height'])
                })
        
        if len(step_features) < 3:
            return 0.0
        
        # Calculate consistency measures
        widths = [f['width'] for f in step_features]
        height_diffs = [f['height_diff'] for f in step_features]
        
        consistency_score = 0.0
        
        # 1. Step width consistency (should be relatively stable)
        if len(widths) >= 3 and np.mean(widths) > 0:
            width_cv = np.std(widths) / np.mean(widths)
            # *** FIXED: Normalize properly ***
            width_inconsistency = min(width_cv / self.clinical_thresholds['step_consistency_cv'], 1.0)
            consistency_score += width_inconsistency * 0.6
            
            if self.debug_mode:
                print(f"    Width CV: {width_cv:.4f}, Inconsistency: {width_inconsistency:.4f}")
        
        # 2. Height difference consistency
        if len(height_diffs) >= 3 and np.mean(height_diffs) > 0:
            height_cv = np.std(height_diffs) / np.mean(height_diffs)
            height_inconsistency = min(height_cv / 0.5, 1.0)  # 0.5 is threshold for height variation
            consistency_score += height_inconsistency * 0.4
            
            if self.debug_mode:
                print(f"    Height CV: {height_cv:.4f}, Inconsistency: {height_inconsistency:.4f}")
        
        return min(consistency_score, 1.0)

    def calculate_lateral_stability_fixed(self, history):
        """*** COMPLETELY REWRITTEN: Fix the maxed out lateral stability ***"""
        valid_measurements = []
        
        for frame in history:
            # More robust check for valid measurements
            if all(key in frame for key in ['nose_x', 'hip_center_x', 'hip_width']):
                if frame['hip_width'] > 10:  # Minimum reasonable hip width in pixels
                    deviation = abs(frame['nose_x'] - frame['hip_center_x'])
                    normalized_deviation = deviation / frame['hip_width']
                    
                    # *** FILTER: Remove outliers ***
                    if normalized_deviation < 2.0:  # Remove extreme outliers
                        valid_measurements.append(normalized_deviation)
        
        if len(valid_measurements) < 3:
            return 0.0
        
        # Calculate stability metrics
        avg_deviation = np.mean(valid_measurements)
        std_deviation = np.std(valid_measurements)
        
        # *** FIXED: Better scoring ***
        # Average deviation component
        avg_score = min(avg_deviation / self.clinical_thresholds['lateral_stability_ratio'], 1.0)
        
        # Variability component (instability)
        var_score = min(std_deviation / 0.1, 1.0)  # 0.1 is threshold for stability
        
        # Combine with weights
        stability_score = avg_score * 0.7 + var_score * 0.3
        
        if self.debug_mode:
            print(f"    Avg deviation: {avg_deviation:.4f}, Std: {std_deviation:.4f}")
            print(f"    Avg score: {avg_score:.4f}, Var score: {var_score:.4f}")
        
        return min(stability_score, 1.0)

    def calculate_leg_asymmetry(self, history):
        """Enhanced leg asymmetry calculation"""
        left_knee_angles = []
        right_knee_angles = []
        left_ankle_heights = []
        right_ankle_heights = []
        
        for frame in history:
            if all(key in frame for key in ['left_knee_angle', 'right_knee_angle']):
                # *** FILTER: Only reasonable angles ***
                if 30 <= frame['left_knee_angle'] <= 180 and 30 <= frame['right_knee_angle'] <= 180:
                    left_knee_angles.append(frame['left_knee_angle'])
                    right_knee_angles.append(frame['right_knee_angle'])
            
            if all(key in frame for key in ['left_ankle_height', 'right_ankle_height']):
                left_ankle_heights.append(frame['left_ankle_height'])
                right_ankle_heights.append(frame['right_ankle_height'])
        
        if len(left_knee_angles) < 3:
            return 0.0
        
        asymmetry_score = 0.0
        
        # 1. Knee angle asymmetry
        knee_asymmetries = [abs(l - r) for l, r in zip(left_knee_angles, right_knee_angles)]
        avg_asymmetry = np.mean(knee_asymmetries)
        knee_score = min(avg_asymmetry / self.clinical_thresholds['leg_asymmetry_max'], 1.0)
        asymmetry_score += knee_score * 0.7
        
        # 2. Ankle height pattern asymmetry
        if len(left_ankle_heights) >= 3:
            left_var = np.std(left_ankle_heights)
            right_var = np.std(right_ankle_heights)
            total_var = left_var + right_var
            
            if total_var > 0:
                height_asymmetry = abs(left_var - right_var) / total_var
                asymmetry_score += min(height_asymmetry * 2, 1.0) * 0.3  # Boost sensitivity
        
        if self.debug_mode:
            print(f"    Knee asymmetry: {avg_asymmetry:.2f}°, Score: {knee_score:.4f}")
        
        return min(asymmetry_score, 1.0)

    def calculate_joint_movement(self, history):
        """Improved joint movement calculation"""
        left_knee_angles = []
        right_knee_angles = []
        
        for frame in history:
            if all(key in frame for key in ['left_knee_angle', 'right_knee_angle']):
                if 30 <= frame['left_knee_angle'] <= 180 and 30 <= frame['right_knee_angle'] <= 180:
                    left_knee_angles.append(frame['left_knee_angle'])
                    right_knee_angles.append(frame['right_knee_angle'])
        
        if len(left_knee_angles) < 5:
            return 0.0
        
        # Calculate range of motion
        left_rom = max(left_knee_angles) - min(left_knee_angles)
        right_rom = max(right_knee_angles) - min(right_knee_angles)
        
        joint_score = 0.0
        
        # 1. ROM asymmetry
        rom_asymmetry = abs(left_rom - right_rom)
        asymmetry_score = min(rom_asymmetry / self.clinical_thresholds['joint_movement_max'], 1.0)
        joint_score += asymmetry_score * 0.6
        
        # 2. Overall ROM restriction (both legs)
        avg_rom = (left_rom + right_rom) / 2
        if avg_rom < 30:  # Very restricted movement
            restriction_score = (30 - avg_rom) / 30
            joint_score += restriction_score * 0.4
        
        if self.debug_mode:
            print(f"    ROM L:{left_rom:.1f}°, R:{right_rom:.1f}°, Asymmetry: {rom_asymmetry:.1f}°")
        
        return min(joint_score, 1.0)

    def calculate_vertical_oscillation(self, history):
        """Improved vertical oscillation calculation"""
        hip_heights = []
        
        for frame in history:
            if 'hip_center_y' in frame:
                hip_heights.append(frame['hip_center_y'])
        
        if len(hip_heights) < 5:
            return 0.0
        
        # Calculate vertical movement variability
        if np.mean(hip_heights) > 0:
            height_cv = np.std(hip_heights) / np.mean(hip_heights)
            oscillation_score = min(height_cv / self.clinical_thresholds['vertical_oscillation_cv'], 1.0)
            
            if self.debug_mode:
                print(f"    Hip height CV: {height_cv:.4f}, Score: {oscillation_score:.4f}")
            
            return oscillation_score
        
        return 0.0

    def classify_deviation_type(self, metric_scores, weighted_score):
        """*** MUCH MORE AGGRESSIVE: Lower thresholds for classification ***"""
        # Sort metrics by score
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_metric, dominant_score = sorted_metrics[0]
        
        # *** VERY LOW THRESHOLDS: Catch subtle deviations ***
        if dominant_score > 0.15:  # Reduced from 0.4
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
        
        # Mixed patterns with very low thresholds
        if metric_scores['leg_asymmetry'] > 0.1 and metric_scores['joint_movement'] > 0.1:
            return "limping"
        elif metric_scores['lateral_stability'] > 0.1 and metric_scores['step_consistency'] > 0.1:
            return "balance_issues"
        elif metric_scores['step_consistency'] > 0.12:
            return "irregular_gait"
        elif metric_scores['leg_asymmetry'] > 0.12:
            return "asymmetric_gait"
        elif metric_scores['vertical_oscillation'] > 0.1:
            return "bouncing_gait"
        
        # Catch anything with even minimal deviation
        if weighted_score > 0.05:
            return "walking_deviation"
        
        return "normal"

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
        """Enhanced feature extraction with better validation"""
        if len(keypoints) < 17:
            return None
            
        features = {}
        
        # *** LOWER CONFIDENCE: Accept more keypoints ***
        def get_point_if_confident(kp_name, min_conf=0.15):  # Much lower
            kp = keypoints[self.POSE_KEYPOINTS[kp_name]]
            return kp if kp[2] > min_conf else None
        
        # Essential points
        nose = get_point_if_confident('nose', 0.1)  # Very low for head
        left_hip = get_point_if_confident('left_hip', 0.2)
        right_hip = get_point_if_confident('right_hip', 0.2)
        left_knee = get_point_if_confident('left_knee', 0.2)
        right_knee = get_point_if_confident('right_knee', 0.2)
        left_ankle = get_point_if_confident('left_ankle', 0.2)
        right_ankle = get_point_if_confident('right_ankle', 0.2)
        
        # *** RELAXED: Only need most essential points ***
        if not all([left_hip, right_hip]):
            return None
        
        # Basic measurements
        features['hip_center_x'] = (left_hip[0] + right_hip[0]) / 2
        features['hip_center_y'] = (left_hip[1] + right_hip[1]) / 2
        features['hip_width'] = euclidean(left_hip[:2], right_hip[:2])
        
        # Add ankle measurements if available
        if left_ankle and right_ankle:
            features['step_width'] = abs(left_ankle[0] - right_ankle[0])
            features['left_ankle_height'] = left_ankle[1]
            features['right_ankle_height'] = right_ankle[1]
        
        # Head position if available
        if nose:
            features['nose_x'] = nose[0]
            features['nose_y'] = nose[1]
        
        # Calculate joint angles if we have the points
        if all([left_hip, left_knee, left_ankle]):
            try:
                left_angle = self.calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
                if 20 <= left_angle <= 190:  # Very relaxed range
                    features['left_knee_angle'] = left_angle
            except:
                pass
        
        if all([right_hip, right_knee, right_ankle]):
            try:
                right_angle = self.calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
                if 20 <= right_angle <= 190:  # Very relaxed range
                    features['right_knee_angle'] = right_angle
            except:
                pass
        
        return features if len(features) >= 3 else None  # Need at least 3 measurements
    
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

    # Legacy method for compatibility
    def analyze_gait_pattern(self, person_id, current_features):
        """Legacy method - now uses weighted scoring system"""
        weighted_score, metric_scores = self.calculate_weighted_deviation_score(person_id)
        is_deviation = weighted_score > self.deviation_threshold
        
        if is_deviation:
            category = self.classify_deviation_type(metric_scores, weighted_score)
            return True, category
        else:
            return False, "normal"