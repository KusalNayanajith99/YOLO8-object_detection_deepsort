import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math

class WalkingDeviationDetector:
    def __init__(self, history_length=30):
        """
        Initialize the walking deviation detector.
        
        Args:
            history_length: Number of frames to keep in history for analysis
        """
        self.pose_model = YOLO("yolov8n-pose.pt")
        self.history_length = history_length
        
        # Store pose history for each person
        self.pose_history = defaultdict(lambda: deque(maxlen=history_length))
        self.step_history = defaultdict(lambda: deque(maxlen=history_length))
        self.head_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Weights for different metrics - adjusted for better detection
        self.weights = {
            'leg_asymmetry': 0.50,      # Increased from 0.40 - most reliable
            'step_consistency': 0.25,   # Increased from 0.20 - good indicator
            'lateral_stability': 0.15,  # Kept same
            'joint_movement': 0.10,     # Decreased from 0.15 - less reliable
            'vertical_oscillation': 0.0 # Disabled - often noisy
        }
        
        # Thresholds for normal vs deviation - made more sensitive
        self.thresholds = {
            'leg_asymmetry_max': 20.0,    # Reduced from 30.0
            'step_cv_max': 0.15,          # Reduced from 0.2
            'lateral_deviation_max': 0.25, # Reduced from 0.3
            'joint_asymmetry_max': 0.15,  # Reduced from 0.2
            'vertical_cv_max': 0.02       # Reduced from 0.03
        }
        
        # Keypoint indices for YOLO pose model (COCO format)
        self.keypoints = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
    def detect_walking_deviation(self, frame, global_id, bbox):
        """
        Detect walking deviations for a person in the given frame.
        
        Args:
            frame: Current video frame
            global_id: Person's global ID
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            tuple: (deviation_category, confidence_score)
        """
        # Extract person region from frame
        x1, y1, x2, y2 = map(int, bbox)
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return "normal", 0.0
            
        # Get pose keypoints for the person
        pose_results = self.pose_model(person_roi)
        
        if not pose_results or not pose_results[0].keypoints or pose_results[0].keypoints.data is None:
            return "normal", 0.0
        
        # Check if keypoints data is empty
        keypoints_data = pose_results[0].keypoints.data
        if keypoints_data.numel() == 0 or keypoints_data.shape[0] == 0:
            return "normal", 0.0
            
        keypoints = keypoints_data[0].cpu().numpy()  # [17, 3] - (x, y, confidence)
        
        # Additional check for keypoints shape
        if keypoints.shape[0] < 17:  # Should have 17 keypoints
            return "normal", 0.0
        
        # Adjust keypoints to global frame coordinates
        keypoints[:, 0] += x1  # x coordinates
        keypoints[:, 1] += y1  # y coordinates
        
        # Store pose data in history
        self.pose_history[global_id].append(keypoints)
        
        # Need sufficient history for analysis (reduced from 10 to 5)
        if len(self.pose_history[global_id]) < 5:
            return "normal", 0.0
            
        # Calculate all metrics
        scores = self._calculate_all_metrics(global_id)
        
        # Calculate weighted final score
        final_score = sum(scores[metric] * weight for metric, weight in self.weights.items())
        
        # Determine if deviation exists (threshold: 0.3 means 30% of maximum possible deviation)
        if final_score > 0.3:  # Lowered from 0.5 to be more sensitive
            return "walking_deviation", final_score
        else:
            return "normal", final_score
    
    def _calculate_all_metrics(self, global_id):
        """Calculate all five walking deviation metrics."""
        poses = list(self.pose_history[global_id])
        
        scores = {}
        
        # 1. Leg Asymmetry (40% weight)
        scores['leg_asymmetry'] = self._calculate_leg_asymmetry(poses)
        
        # 2. Step Consistency (20% weight)
        scores['step_consistency'] = self._calculate_step_consistency(global_id, poses)
        
        # 3. Lateral Stability (15% weight)
        scores['lateral_stability'] = self._calculate_lateral_stability(global_id, poses)
        
        # 4. Joint Movement (15% weight)
        scores['joint_movement'] = self._calculate_joint_movement(poses)
        
        # 5. Vertical Oscillation (10% weight)
        scores['vertical_oscillation'] = self._calculate_vertical_oscillation(poses)
        
        return scores
    
    def _calculate_leg_asymmetry(self, poses):
        """Calculate leg asymmetry based on ankle positions and stride angles."""
        asymmetries = []
        
        for i in range(1, len(poses)):
            current_pose = poses[i]
            prev_pose = poses[i-1]
            
            # Check if pose has enough keypoints
            if current_pose.shape[0] < 17:
                continue
            
            # Get ankle positions
            left_ankle = current_pose[self.keypoints['left_ankle']]
            right_ankle = current_pose[self.keypoints['right_ankle']]
            left_hip = current_pose[self.keypoints['left_hip']]
            right_hip = current_pose[self.keypoints['right_hip']]
            
            # Check if keypoints are visible (lowered confidence threshold)
            if (left_ankle[2] < 0.3 or right_ankle[2] < 0.3 or 
                left_hip[2] < 0.3 or right_hip[2] < 0.3):
                continue
                
            # Calculate leg angles relative to vertical
            left_leg_vector = left_ankle[:2] - left_hip[:2]
            right_leg_vector = right_ankle[:2] - right_hip[:2]
            
            # Calculate angles with vertical (90 degrees)
            left_angle = abs(math.degrees(math.atan2(left_leg_vector[0], left_leg_vector[1])))
            right_angle = abs(math.degrees(math.atan2(right_leg_vector[0], right_leg_vector[1])))
            
            # Calculate asymmetry
            asymmetry = abs(left_angle - right_angle)
            asymmetries.append(asymmetry)
        
        if not asymmetries:
            return 0.0
            
        avg_asymmetry = np.mean(asymmetries)
        # Normalize by maximum threshold (30 degrees)
        score = min(avg_asymmetry / self.thresholds['leg_asymmetry_max'], 1.0)
        
        return score
    
    def _calculate_step_consistency(self, global_id, poses):
        """Calculate step timing consistency using coefficient of variation."""
        step_intervals = []
        
        # Detect steps by analyzing ankle movement patterns
        for i in range(2, len(poses)):
            current_pose = poses[i]
            prev_pose = poses[i-1]
            
            # Check if pose has enough keypoints
            if current_pose.shape[0] < 17 or prev_pose.shape[0] < 17:
                continue
            
            left_ankle = current_pose[self.keypoints['left_ankle']]
            right_ankle = current_pose[self.keypoints['right_ankle']]
            
            if left_ankle[2] < 0.5 or right_ankle[2] < 0.5:
                continue
                
            # Calculate ankle movement speed
            left_movement = np.linalg.norm(left_ankle[:2] - poses[i-1][self.keypoints['left_ankle']][:2])
            right_movement = np.linalg.norm(right_ankle[:2] - poses[i-1][self.keypoints['right_ankle']][:2])
            
            # Store movement data for step detection
            self.step_history[global_id].append((left_movement, right_movement))
        
        # Analyze step intervals from movement history
        if len(self.step_history[global_id]) < 10:
            return 0.0
            
        movements = list(self.step_history[global_id])
        step_intervals = self._detect_step_intervals(movements)
        
        if len(step_intervals) < 3:
            return 0.0
            
        # Calculate coefficient of variation
        mean_interval = np.mean(step_intervals)
        std_interval = np.std(step_intervals)
        
        if mean_interval == 0:
            return 0.0
            
        cv = std_interval / mean_interval
        
        # Normalize by threshold
        score = min(cv / self.thresholds['step_cv_max'], 1.0)
        
        return score
    
    def _detect_step_intervals(self, movements):
        """Detect step intervals from ankle movement data."""
        intervals = []
        last_step_frame = 0
        movement_threshold = 5.0  # Minimum movement to consider as a step
        
        for i, (left_mov, right_mov) in enumerate(movements):
            total_movement = left_mov + right_mov
            
            if total_movement > movement_threshold and i - last_step_frame > 3:
                if last_step_frame > 0:
                    intervals.append(i - last_step_frame)
                last_step_frame = i
                
        return intervals
    
    def _calculate_lateral_stability(self, global_id, poses):
        """Calculate lateral stability using head position deviation."""
        head_positions = []
        hip_widths = []
        
        for pose in poses:
            # Check if pose has enough keypoints
            if pose.shape[0] < 17:
                continue
                
            # Use nose as head position
            head = pose[self.keypoints['nose']]
            left_hip = pose[self.keypoints['left_hip']]
            right_hip = pose[self.keypoints['right_hip']]
            
            if head[2] < 0.5 or left_hip[2] < 0.5 or right_hip[2] < 0.5:
                continue
                
            head_positions.append(head[0])  # x-coordinate
            hip_width = abs(left_hip[0] - right_hip[0])
            hip_widths.append(hip_width)
        
        if len(head_positions) < 5:
            return 0.0
            
        # Calculate lateral deviation
        head_center = np.mean(head_positions)
        lateral_deviations = [abs(pos - head_center) for pos in head_positions]
        avg_deviation = np.mean(lateral_deviations)
        avg_hip_width = np.mean(hip_widths) if hip_widths else 1.0
        
        # Normalize by hip width
        normalized_deviation = avg_deviation / avg_hip_width if avg_hip_width > 0 else 0
        
        # Normalize by threshold
        score = min(normalized_deviation / self.thresholds['lateral_deviation_max'], 1.0)
        
        return score
    
    def _calculate_joint_movement(self, poses):
        """Calculate joint movement asymmetry using knee angles."""
        left_knee_angles = []
        right_knee_angles = []
        
        for pose in poses:
            # Check if pose has enough keypoints
            if pose.shape[0] < 17:
                continue
                
            left_hip = pose[self.keypoints['left_hip']]
            left_knee = pose[self.keypoints['left_knee']]
            left_ankle = pose[self.keypoints['left_ankle']]
            right_hip = pose[self.keypoints['right_hip']]
            right_knee = pose[self.keypoints['right_knee']]
            right_ankle = pose[self.keypoints['right_ankle']]
            
            # Check visibility
            if (left_hip[2] < 0.5 or left_knee[2] < 0.5 or left_ankle[2] < 0.5 or
                right_hip[2] < 0.5 or right_knee[2] < 0.5 or right_ankle[2] < 0.5):
                continue
                
            # Calculate knee angles
            left_angle = self._calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
            right_angle = self._calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
            
            left_knee_angles.append(left_angle)
            right_knee_angles.append(right_angle)
        
        if len(left_knee_angles) < 5:
            return 0.0
            
        # Calculate asymmetry in joint movement ranges
        left_range = max(left_knee_angles) - min(left_knee_angles)
        right_range = max(right_knee_angles) - min(right_knee_angles)
        
        if left_range + right_range == 0:
            return 0.0
            
        asymmetry = abs(left_range - right_range) / (left_range + right_range) * 2
        
        # Normalize by threshold
        score = min(asymmetry / self.thresholds['joint_asymmetry_max'], 1.0)
        
        return score
    
    def _calculate_vertical_oscillation(self, poses):
        """Calculate vertical oscillation using head/hip center movement."""
        vertical_positions = []
        
        for pose in poses:
            # Check if pose has enough keypoints
            if pose.shape[0] < 17:
                continue
                
            left_hip = pose[self.keypoints['left_hip']]
            right_hip = pose[self.keypoints['right_hip']]
            
            if left_hip[2] < 0.5 or right_hip[2] < 0.5:
                continue
                
            # Use hip center y-coordinate
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            vertical_positions.append(hip_center_y)
        
        if len(vertical_positions) < 5:
            return 0.0
            
        # Calculate coefficient of variation for vertical movement
        mean_pos = np.mean(vertical_positions)
        std_pos = np.std(vertical_positions)
        
        if mean_pos == 0:
            return 0.0
            
        cv = std_pos / mean_pos
        
        # Normalize by threshold
        score = min(cv / self.thresholds['vertical_cv_max'], 1.0)
        
        return score
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle at point p2 formed by points p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        
        angle = math.degrees(math.acos(cos_angle))
        return angle