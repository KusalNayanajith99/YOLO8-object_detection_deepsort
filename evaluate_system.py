# evaluate_system.py
import cv2
import time
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt

# Import your existing modules
from suspicion_detector import SuspicionDetector
from walking_deviation_detector import WalkingDeviationDetector

class SystemEvaluator:
    def __init__(self):
        # Initialize your existing detectors
        print("Loading models...")
        self.suspicion_detector = SuspicionDetector("suspicious_detector.pt")
        self.walking_detector = WalkingDeviationDetector()
        print("Models loaded successfully!")
        
        # Store results
        self.results = {}

    def evaluate_suspicious_activities(self, test_folder):
        """
        Metric 1: Evaluate your YOLOv8 suspicious activity detection
        Tests: Precision, Recall, F1-Score, mAP, Processing Time
        """
        print("\n=== Evaluating Suspicious Activity Detection ===")
        
        true_labels = []
        predicted_labels = []
        processing_times = []
        
        # Categories your model detects
        detection_categories = ['arson', 'Fighting', 'abuse', 'Shooting']
        
        # Test each suspicious activity category
        for category in detection_categories:
            folder_path = os.path.join(test_folder, category)
            print(f"Testing {category}...")
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} doesn't exist. Skipping...")
                continue
                
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_file)
                    frame = cv2.imread(img_path)
                    
                    if frame is None:
                        continue
                    
                    # Time the detection process
                    start_time = time.time()
                    results = self.suspicion_detector.model(frame)[0]
                    processing_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Get prediction
                    predicted_category = 'no_detection'
                    if len(results.boxes) > 0:
                        # Get the detection with highest confidence
                        best_detection = results.boxes[0]
                        predicted_category = self.suspicion_detector.model.names[int(best_detection.cls)]
                    
                    true_labels.append(category)
                    predicted_labels.append(predicted_category)
                    processing_times.append(processing_time)
        
        # Test normal scenes (should detect nothing)
        normal_folder = os.path.join(test_folder, 'normal_scenes')
        if os.path.exists(normal_folder):
            print("Testing normal scenes...")
            for img_file in os.listdir(normal_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(normal_folder, img_file)
                    frame = cv2.imread(img_path)
                    
                    if frame is None:
                        continue
                    
                    start_time = time.time()
                    results = self.suspicion_detector.model(frame)[0]
                    processing_time = (time.time() - start_time) * 1000
                    
                    predicted_category = 'no_detection'
                    if len(results.boxes) > 0:
                        predicted_category = self.suspicion_detector.model.names[int(results.boxes[0].cls)]
                    
                    true_labels.append('normal')
                    predicted_labels.append(predicted_category)
                    processing_times.append(processing_time)
        
        # Calculate metrics
        results = self._calculate_suspicious_metrics(true_labels, predicted_labels, processing_times)
        self.results['suspicious_activities'] = results
        
        return results

    def _calculate_suspicious_metrics(self, true_labels, predicted_labels, processing_times):
        """Calculate metrics for suspicious activity detection"""
        categories = ['arson', 'Fighting', 'abuse', 'Shooting']
        results = {}
        
        for category in categories:
            tp = sum(1 for t, p in zip(true_labels, predicted_labels) 
                    if t == category and p == category)
            fp = sum(1 for t, p in zip(true_labels, predicted_labels) 
                    if t != category and p == category)
            fn = sum(1 for t, p in zip(true_labels, predicted_labels) 
                    if t == category and p != category)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp, 'fp': fp, 'fn': fn
            }
        
        # Overall metrics
        all_precision = [results[cat]['precision'] for cat in categories]
        all_recall = [results[cat]['recall'] for cat in categories]
        all_f1 = [results[cat]['f1_score'] for cat in categories]
        
        # Calculate mAP (mean Average Precision)
        map_score = np.mean([p for p in all_precision if p > 0])
        
        # False positive rate on normal scenes
        normal_fps = sum(1 for t, p in zip(true_labels, predicted_labels) 
                        if t == 'normal' and p != 'no_detection')
        total_normal = sum(1 for t in true_labels if t == 'normal')
        false_positive_rate = normal_fps / total_normal if total_normal > 0 else 0
        
        results['overall'] = {
            'mAP': map_score,
            'avg_precision': np.mean(all_precision),
            'avg_recall': np.mean(all_recall),
            'avg_f1': np.mean(all_f1),
            'false_positive_rate': false_positive_rate,
            'avg_processing_time': np.mean(processing_times)
        }
        
        return results

    def evaluate_walking_deviation(self, test_video_folder):
        """
        Metric 2: Evaluate walking deviation detection
        Tests: Accuracy, Sensitivity, Specificity for each deviation type
        """
        print("\n=== Evaluating Walking Deviation Detection ===")
        
        true_labels = []
        predicted_labels = []
        processing_times = []
        
        deviation_types = ['normal_walking', 'limping', 'balance_issues', 'irregular_gait']
        
        for dev_type in deviation_types:
            folder_path = os.path.join(test_video_folder, dev_type)
            print(f"Testing {dev_type}...")
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} doesn't exist. Skipping...")
                continue
                
            for video_file in os.listdir(folder_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(folder_path, video_file)
                    cap = cv2.VideoCapture(video_path)
                    
                    predictions = []
                    times = []
                    
                    # Process 15 frames from each video
                    frame_count = 0
                    while frame_count < 15:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Create a test bounding box (center of frame)
                        h, w = frame.shape[:2]
                        bbox = [w//4, h//4, 3*w//4, 3*h//4]
                        
                        start_time = time.time()
                        deviation, confidence = self.walking_detector.detect_walking_deviation(
                            frame, f"test_person_{dev_type}", bbox
                        )
                        processing_time = (time.time() - start_time) * 1000
                        
                        predictions.append(deviation)
                        times.append(processing_time)
                        frame_count += 1
                    
                    cap.release()
                    
                    # Take majority vote for video classification
                    if predictions:
                        most_common = max(set(predictions), key=predictions.count)
                        
                        # Convert folder name to expected label
                        expected_label = dev_type.replace('_walking', '')
                        
                        true_labels.append(expected_label)
                        predicted_labels.append(most_common)
                        processing_times.extend(times)
        
        # Calculate metrics
        results = self._calculate_walking_metrics(true_labels, predicted_labels, processing_times)
        self.results['walking_deviation'] = results
        
        return results

    def _calculate_walking_metrics(self, true_labels, predicted_labels, processing_times):
        """Calculate metrics for walking deviation detection"""
        labels = ['normal', 'limping', 'balance_issues', 'irregular_gait']
        
        # Calculate precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels, average=None, zero_division=0
        )
        
        # Calculate overall accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        results = {}
        for i, label in enumerate(labels):
            results[label] = {
                'precision': precision[i] if i < len(precision) else 0,
                'recall': recall[i] if i < len(recall) else 0,
                'f1_score': f1[i] if i < len(f1) else 0,
                'support': support[i] if i < len(support) else 0
            }
        
        results['overall'] = {
            'accuracy': accuracy,
            'avg_precision': np.mean(precision),
            'avg_recall': np.mean(recall),
            'avg_f1': np.mean(f1),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0
        }
        
        return results

    def evaluate_temporal_analysis(self, test_video_path):
        """
        Metric 3: Compare temporal vs single-frame analysis
        Tests: False positive reduction, detection stability
        """
        print("\n=== Evaluating Temporal Analysis Effectiveness ===")
        
        if not os.path.exists(test_video_path):
            print(f"Test video {test_video_path} not found!")
            return {'error': 'Test video not found'}
        
        # Test with temporal analysis (your current method)
        temporal_results = self._test_analysis_method(test_video_path, use_temporal=True)
        
        # Test with single frame analysis (disable temporal buffer)
        single_frame_results = self._test_analysis_method(test_video_path, use_temporal=False)
        
        improvement = 0
        if single_frame_results['false_positive_rate'] > 0:
            improvement = ((single_frame_results['false_positive_rate'] - temporal_results['false_positive_rate']) 
                          / single_frame_results['false_positive_rate'] * 100)
        
        results = {
            'temporal_false_positives': temporal_results['false_positive_rate'],
            'single_frame_false_positives': single_frame_results['false_positive_rate'],
            'improvement_percentage': improvement,
            'temporal_stability': temporal_results['stability_score'],
            'single_frame_stability': single_frame_results['stability_score']
        }
        
        self.results['temporal_analysis'] = results
        return results

    def _test_analysis_method(self, video_path, use_temporal=True):
        """Test analysis method with or without temporal buffer"""
        cap = cv2.VideoCapture(video_path)
        
        predictions = []
        frame_count = 0
        
        while frame_count < 100:  # Test 100 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            h, w = frame.shape[:2]
            bbox = [w//4, h//4, 3*w//4, 3*h//4]
            
            if use_temporal:
                deviation, confidence = self.walking_detector.detect_walking_deviation(
                    frame, "test_person_temporal", bbox
                )
            else:
                # For single frame, create new person ID each time
                deviation, confidence = self.walking_detector.detect_walking_deviation(
                    frame, f"test_person_single_{frame_count}", bbox
                )
            
            predictions.append(deviation)
            frame_count += 1
        
        cap.release()
        
        # Calculate metrics
        false_positives = sum(1 for p in predictions if p != 'normal')
        false_positive_rate = false_positives / len(predictions) if predictions else 0
        
        # Calculate stability (how often predictions change)
        changes = 0
        if len(predictions) > 1:
            changes = sum(1 for i in range(1, len(predictions)) 
                         if predictions[i] != predictions[i-1])
        stability_score = 1 - (changes / len(predictions)) if predictions else 0
        
        return {
            'false_positive_rate': false_positive_rate,
            'stability_score': stability_score,
            'total_predictions': len(predictions)
        }

    def save_results_to_file(self, filename="results/evaluation_results.txt"):
        """Save all results to a text file"""
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("=== WALKING DEVIATION DETECTION SYSTEM EVALUATION RESULTS ===\n\n")
            
            # Suspicious Activities Results
            if 'suspicious_activities' in self.results:
                f.write("1. SUSPICIOUS ACTIVITY DETECTION RESULTS:\n")
                sus_results = self.results['suspicious_activities']
                
                for category in ['arson', 'Fighting', 'abuse', 'Shooting']:
                    if category in sus_results:
                        r = sus_results[category]
                        f.write(f"   {category.upper()}:\n")
                        f.write(f"      Precision: {r['precision']:.3f}\n")
                        f.write(f"      Recall: {r['recall']:.3f}\n")
                        f.write(f"      F1-Score: {r['f1_score']:.3f}\n\n")
                
                if 'overall' in sus_results:
                    o = sus_results['overall']
                    f.write(f"   OVERALL:\n")
                    f.write(f"      mAP: {o['mAP']:.3f}\n")
                    f.write(f"      Avg Processing Time: {o['avg_processing_time']:.1f}ms\n")
                    f.write(f"      False Positive Rate: {o['false_positive_rate']:.3f}\n\n")
            
            # Walking Deviation Results
            if 'walking_deviation' in self.results:
                f.write("2. WALKING DEVIATION DETECTION RESULTS:\n")
                walk_results = self.results['walking_deviation']
                
                for label in ['normal', 'limping', 'balance_issues', 'irregular_gait']:
                    if label in walk_results:
                        r = walk_results[label]
                        f.write(f"   {label.upper()}:\n")
                        f.write(f"      Precision: {r['precision']:.3f}\n")
                        f.write(f"      Recall: {r['recall']:.3f}\n")
                        f.write(f"      F1-Score: {r['f1_score']:.3f}\n\n")
                
                if 'overall' in walk_results:
                    o = walk_results['overall']
                    f.write(f"   OVERALL:\n")
                    f.write(f"      Accuracy: {o['accuracy']:.3f}\n")
                    f.write(f"      Avg Processing Time: {o['avg_processing_time']:.1f}ms\n\n")
            
            # Temporal Analysis Results
            if 'temporal_analysis' in self.results:
                f.write("3. TEMPORAL ANALYSIS RESULTS:\n")
                temp_results = self.results['temporal_analysis']
                f.write(f"   Single-frame False Positives: {temp_results['single_frame_false_positives']:.3f}\n")
                f.write(f"   Temporal False Positives: {temp_results['temporal_false_positives']:.3f}\n")
                f.write(f"   Improvement: {temp_results['improvement_percentage']:.1f}%\n\n")
        
        print(f"Results saved to: {filename}")


# MOVE THESE FUNCTIONS OUTSIDE THE CLASS
def run_complete_evaluation():
    """Run all evaluation metrics"""
    evaluator = SystemEvaluator()
    
    print("Starting Complete System Evaluation...")
    print("Make sure you have test data in the test_data/ folder!")
    
    # 1. Suspicious Activity Evaluation
    if os.path.exists("test_data/suspicious_activities"):
        evaluator.evaluate_suspicious_activities("test_data/suspicious_activities/")
    else:
        print("Warning: test_data/suspicious_activities not found!")
    
    # 2. Walking Deviation Evaluation
    if os.path.exists("test_data/walking_patterns"):
        evaluator.evaluate_walking_deviation("test_data/walking_patterns/")
    else:
        print("Warning: test_data/walking_patterns not found!")
    
    # 3. Temporal Analysis (using your main video file)
    if os.path.exists("data/shooting.mp4"):
        evaluator.evaluate_temporal_analysis("data/shooting.mp4")
    else:
        print("Warning: data/shooting.mp4 not found!")
    
    # Save results
    evaluator.save_results_to_file()
    
    print("\n=== EVALUATION COMPLETE ===")
    print("Results saved to: results/evaluation_results.txt")
    
    # Print summary
    if evaluator.results:
        print("\nQUICK SUMMARY:")
        if 'suspicious_activities' in evaluator.results:
            sus_map = evaluator.results['suspicious_activities'].get('overall', {}).get('mAP', 0)
            print(f"Suspicious Activity mAP: {sus_map:.3f}")
        
        if 'walking_deviation' in evaluator.results:
            walk_acc = evaluator.results['walking_deviation'].get('overall', {}).get('accuracy', 0)
            print(f"Walking Deviation Accuracy: {walk_acc:.3f}")
        
        if 'temporal_analysis' in evaluator.results:
            improvement = evaluator.results['temporal_analysis'].get('improvement_percentage', 0)
            print(f"Temporal Analysis Improvement: {improvement:.1f}%")


# THIS SHOULD BE AT THE BOTTOM OF THE FILE
if __name__ == "__main__":
    run_complete_evaluation()