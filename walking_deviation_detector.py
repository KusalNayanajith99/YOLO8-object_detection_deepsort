import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from collections import deque, defaultdict
import cv2
import math

class WalkingDeviationDetector:
    def __init__(self, history_length=30, deviation_threshold=0.5):
        """
        Initialize walking deviation detector
        
        Args:
            history_length: Number of frames to keep in history for analysis
            deviation_threshold: Threshold for detecting walking deviations
        """
        self.history_length = history_length
        self.deviation_threshold = deviation_threshold
        
        # Store pose history for each person
        self.pose_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Store gait parameters for each person
        self.gait_parameters = defaultdict(dict)
        
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
    
    def extract_pose_features(self, keypoints):
        """Extract relevant features from pose keypoints for gait analysis"""
        if len(keypoints) < 17:
            return None
            
        features = {}
        
        # Hip center calculation
        left_hip = keypoints[self.POSE_KEYPOINTS['left_hip']]
        right_hip = keypoints[self.POSE_KEYPOINTS['right_hip']]
        
        if left_hip[2] > 0.5 and right_hip[2] > 0.5:  # confidence check
            hip_center = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
            features['hip_center'] = hip_center
            
            # Hip width
            features['hip_width'] = euclidean(left_hip[:2], right_hip[:2])
            
            # Leg angles
            left_knee = keypoints[self.POSE_KEYPOINTS['left_knee']]
            right_knee = keypoints[self.POSE_KEYPOINTS['right_knee']]
            left_ankle = keypoints[self.POSE_KEYPOINTS['left_ankle']]
            right_ankle = keypoints[self.POSE_KEYPOINTS['right_ankle']]
            
            if all(kp[2] > 0.5 for kp in [left_knee, right_knee, left_ankle, right_ankle]):
                # Left leg angle
                features['left_thigh_angle'] = self.calculate_angle(left_hip[:2], left_knee[:2], [left_knee[0], left_knee[1] - 50])
                features['left_calf_angle'] = self.calculate_angle(left_knee[:2], left_ankle[:2], [left_ankle[0], left_ankle[1] + 50])
                
                # Right leg angle
                features['right_thigh_angle'] = self.calculate_angle(right_hip[:2], right_knee[:2], [right_knee[0], right_knee[1] - 50])
                features['right_calf_angle'] = self.calculate_angle(right_knee[:2], right_ankle[:2], [right_ankle[0], right_ankle[1] + 50])
                
                # Step width (distance between ankles)
                features['step_width'] = abs(left_ankle[0] - right_ankle[0])
                
                # Vertical displacement of ankles
                features['left_ankle_height'] = left_ankle[1]
                features['right_ankle_height'] = right_ankle[1]
        
        return features if features else None
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def analyze_gait_pattern(self, person_id, current_features):
        """Analyze gait pattern for walking deviations"""
        if current_features is None:
            return False, "normal"
        
        # Add current features to history
        self.pose_history[person_id].append(current_features)
        
        # Need sufficient history for analysis
        if len(self.pose_history[person_id]) < 15:
            return False, "normal"
        
        history = list(self.pose_history[person_id])
        
        # Analyze different aspects of gait
        deviation_score = 0
        deviation_reasons = []
        
        # 1. Analyze step symmetry
        symmetry_score = self.analyze_step_symmetry(history)
        if symmetry_score > self.deviation_threshold:
            deviation_score += symmetry_score
            deviation_reasons.append("asymmetric_gait")
        
        # 2. Analyze step regularity
        regularity_score = self.analyze_step_regularity(history)
        if regularity_score > self.deviation_threshold:
            deviation_score += regularity_score
            deviation_reasons.append("irregular_gait")
        
        # 3. Analyze limping pattern
        limping_score = self.detect_limping_pattern(history)
        if limping_score > self.deviation_threshold:
            deviation_score += limping_score
            deviation_reasons.append("limping")
        
        # 4. Analyze stance width abnormalities
        stance_score = self.analyze_stance_width(history)
        if stance_score > self.deviation_threshold:
            deviation_score += stance_score
            deviation_reasons.append("abnormal_stance")
        
        # Final decision
        is_deviation = deviation_score > self.deviation_threshold
        category = "walking_deviation" if is_deviation else "normal"
        
        return is_deviation, category
    
    def analyze_step_symmetry(self, history):
        """Analyze symmetry between left and right leg movements"""
        left_angles = []
        right_angles = []
        
        for frame in history:
            if 'left_thigh_angle' in frame and 'right_thigh_angle' in frame:
                left_angles.append(frame['left_thigh_angle'])
                right_angles.append(frame['right_thigh_angle'])
        
        if len(left_angles) < 10:
            return 0
        
        # Calculate asymmetry
        angle_differences = [abs(l - r) for l, r in zip(left_angles, right_angles)]
        avg_asymmetry = np.mean(angle_differences)
        
        # Normalize to 0-1 scale
        normalized_asymmetry = min(avg_asymmetry / 30.0, 1.0)  # 30 degrees as max expected
        
        return normalized_asymmetry
    
    def analyze_step_regularity(self, history):
        """Analyze regularity of step patterns"""
        if len(history) < 20:
            return 0
        
        step_widths = []
        for frame in history:
            if 'step_width' in frame:
                step_widths.append(frame['step_width'])
        
        if len(step_widths) < 10:
            return 0
        
        # Calculate coefficient of variation
        cv = np.std(step_widths) / (np.mean(step_widths) + 1e-6)
        
        # Normalize to 0-1 scale
        return min(cv, 1.0)
    
    def detect_limping_pattern(self, history):
        """Detect limping patterns in gait"""
        left_heights = []
        right_heights = []
        
        for frame in history:
            if 'left_ankle_height' in frame and 'right_ankle_height' in frame:
                left_heights.append(frame['left_ankle_height'])
                right_heights.append(frame['right_ankle_height'])
        
        if len(left_heights) < 15:
            return 0
        
        # Analyze vertical movement patterns
        left_variation = np.std(left_heights)
        right_variation = np.std(right_heights)
        
        # Check for significant difference in leg movement
        variation_diff = abs(left_variation - right_variation)
        avg_variation = (left_variation + right_variation) / 2
        
        if avg_variation > 0:
            limping_score = variation_diff / avg_variation
            return min(limping_score, 1.0)
        
        return 0
    
    def analyze_stance_width(self, history):
        """Analyze stance width for abnormalities"""
        hip_widths = []
        step_widths = []
        
        for frame in history:
            if 'hip_width' in frame:
                hip_widths.append(frame['hip_width'])
            if 'step_width' in frame:
                step_widths.append(frame['step_width'])
        
        if len(hip_widths) < 10 or len(step_widths) < 10:
            return 0
        
        # Calculate ratio of step width to hip width
        avg_hip_width = np.mean(hip_widths)
        avg_step_width = np.mean(step_widths)
        
        if avg_hip_width > 0:
            width_ratio = avg_step_width / avg_hip_width
            
            # Normal walking has step width roughly equal to hip width
            # Abnormal if too wide (>2x) or too narrow (<0.5x)
            if width_ratio > 2.0 or width_ratio < 0.5:
                return min(abs(width_ratio - 1.0), 1.0)
        
        return 0