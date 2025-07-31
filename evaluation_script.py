import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
from datetime import datetime

# Import your modules
from tracker import Tracker
from feature_extractor import OSNetExtractor
from database import DatabaseManager
from suspicion_detector import SuspicionDetector
from walking_deviation_detector import WalkingDeviationDetector
from ultralytics import YOLO

class ModelEvaluator:
    def __init__(self, mongo_uri):
        """Initialize the model evaluator with all necessary components."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.db_manager = DatabaseManager(mongo_uri=mongo_uri)
        self.feature_extractor = OSNetExtractor(model_name='osnet_x1_0', device=self.device)
        self.tracker = Tracker(feature_extractor=self.feature_extractor)
        self.yolo_model = YOLO("yolov8n.pt")
        self.suspicion_detector = SuspicionDetector("suspicious_detector.pt")
        self.walking_detector = WalkingDeviationDetector()
        
        # Detection threshold
        self.detection_threshold = 0.5
        
        # Results storage
        self.suspicious_results = []  # [(true_label, predicted_label), ...]
        self.walking_results = []     # [(true_label, predicted_label), ...]
        
        # Class mappings
        self.suspicious_classes = ['arson', 'Fighting', 'abuse', 'Shooting', 'normal_scenes']
        self.walking_classes = ['normal_walking', 'walking_deviation']
        
    def process_video(self, video_path, true_label, activity_type):
        """
        Process a single video and return predictions.
        
        Args:
            video_path: Path to the video file
            true_label: Ground truth label
            activity_type: 'suspicious' or 'walking'
        """
        print(f"Processing: {video_path} (True label: {true_label})")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Reset tracker for each video
        self.tracker = Tracker(feature_extractor=self.feature_extractor)
        
        # Prediction counters
        suspicious_predictions = defaultdict(int)
        walking_predictions = defaultdict(int)
        frame_count = 0
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for faster processing (process every 5th frame)
            if frame_count % 5 != 0:
                continue
            
            # 1. Person Detection
            results = self.yolo_model(frame)
            detections = []
            for result in results:
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    if int(class_id) == 0 and score > self.detection_threshold:
                        detections.append([x1, y1, x2, y2, score])
            
            if not detections:
                continue
                
            # 2. Update tracker
            self.tracker.update(frame, detections)
            
            if activity_type == 'suspicious':
                # 3. Suspicious activity detection
                suspicion_results = self.suspicion_detector.model(frame)[0]
                if suspicion_results.boxes is not None:
                    for box in suspicion_results.boxes:
                        label = self.suspicion_detector.model.names[int(box.cls)]
                        suspicious_predictions[label] += 1
                else:
                    suspicious_predictions['normal_scenes'] += 1
                    
            elif activity_type == 'walking':
                # 3. Walking deviation detection
                for track in self.tracker.tracks:
                    bbox = track.bbox
                    feature = track.feature
                    
                    # Create a dummy global_id for tracking
                    global_id = f"temp_{track.track_id}"
                    
                    deviation_category, confidence = self.walking_detector.detect_walking_deviation(
                        frame, global_id, bbox
                    )
                    
                    if deviation_category == "walking_deviation":
                        walking_predictions['walking_deviation'] += 1
                    else:
                        walking_predictions['normal_walking'] += 1
        
        cap.release()
        
        # Determine final prediction based on majority vote
        if activity_type == 'suspicious':
            if suspicious_predictions:
                predicted_label = max(suspicious_predictions.keys(), 
                                    key=lambda x: suspicious_predictions[x])
            else:
                predicted_label = 'normal_scenes'
            self.suspicious_results.append((true_label, predicted_label))
            
        elif activity_type == 'walking':
            if walking_predictions:
                predicted_label = max(walking_predictions.keys(), 
                                    key=lambda x: walking_predictions[x])
            else:
                predicted_label = 'normal_walking'
            self.walking_results.append((true_label, predicted_label))
        
        print(f"Predicted: {predicted_label}")
    
    def evaluate_test_data(self, test_data_path):
        """Evaluate all test data and generate classification report."""
        
        # Process suspicious activities
        suspicious_path = os.path.join(test_data_path, "suspicious_activities")
        if os.path.exists(suspicious_path):
            for class_name in self.suspicious_classes:
                class_path = os.path.join(suspicious_path, class_name)
                if os.path.exists(class_path):
                    for video_file in os.listdir(class_path):
                        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            video_path = os.path.join(class_path, video_file)
                            self.process_video(video_path, class_name, 'suspicious')
        
        # Process walking patterns
        walking_path = os.path.join(test_data_path, "walking_patterns")
        if os.path.exists(walking_path):
            for class_name in self.walking_classes:
                class_path = os.path.join(walking_path, class_name)
                if os.path.exists(class_path):
                    for video_file in os.listdir(class_path):
                        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            video_path = os.path.join(class_path, video_file)
                            self.process_video(video_path, class_name, 'walking')
    
    def calculate_metrics(self):
        """Calculate and return precision, recall, F1-score, and accuracy."""
        results = {}
        
        # Suspicious Activities Metrics
        if self.suspicious_results:
            true_labels = [result[0] for result in self.suspicious_results]
            pred_labels = [result[1] for result in self.suspicious_results]
            
            # Calculate metrics for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, pred_labels, labels=self.suspicious_classes, average=None, zero_division=0
            )
            
            # Calculate mPA (mean Per-class Accuracy)
            class_accuracies = []
            for i, class_name in enumerate(self.suspicious_classes):
                class_true = [1 if label == class_name else 0 for label in true_labels]
                class_pred = [1 if label == class_name else 0 for label in pred_labels]
                if sum(class_true) > 0:  # Only calculate if class exists in ground truth
                    class_acc = accuracy_score(class_true, class_pred)
                    class_accuracies.append(class_acc)
            
            mPA = np.mean(class_accuracies) if class_accuracies else 0.0
            overall_accuracy = accuracy_score(true_labels, pred_labels)
            
            results['suspicious_activities'] = {
                'classes': {},
                'mPA': float(mPA),
                'overall_accuracy': float(overall_accuracy)
            }
            
            for i, class_name in enumerate(self.suspicious_classes):
                results['suspicious_activities']['classes'][class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
        
        # Walking Pattern Metrics
        if self.walking_results:
            true_labels = [result[0] for result in self.walking_results]
            pred_labels = [result[1] for result in self.walking_results]
            
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, pred_labels, labels=self.walking_classes, average=None, zero_division=0
            )
            
            overall_accuracy = accuracy_score(true_labels, pred_labels)
            
            results['walking_patterns'] = {
                'classes': {},
                'overall_accuracy': float(overall_accuracy)
            }
            
            for i, class_name in enumerate(self.walking_classes):
                results['walking_patterns']['classes'][class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
        
        return results
    
    def generate_report(self, results, output_path):
        """Generate and save the classification report."""
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'device_used': self.device,
            'results': results
        }
        
        # Save JSON report
        json_path = os.path.join(output_path, 'classification_report.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        txt_path = os.path.join(output_path, 'classification_report.txt')
        with open(txt_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device Used: {self.device}\n\n")
            
            # Suspicious Activities Report
            if 'suspicious_activities' in results:
                f.write("SUSPICIOUS ACTIVITIES DETECTION\n")
                f.write("-" * 35 + "\n")
                f.write(f"Overall Accuracy: {results['suspicious_activities']['overall_accuracy']:.4f}\n")
                f.write(f"Mean Per-class Accuracy (mPA): {results['suspicious_activities']['mPA']:.4f}\n\n")
                
                f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
                f.write("-" * 65 + "\n")
                
                for class_name, metrics in results['suspicious_activities']['classes'].items():
                    f.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                           f"{metrics['f1_score']:<10.4f} {metrics['support']:<10}\n")
                f.write("\n")
            
            # Walking Patterns Report
            if 'walking_patterns' in results:
                f.write("WALKING DEVIATION DETECTION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Overall Accuracy: {results['walking_patterns']['overall_accuracy']:.4f}\n\n")
                
                f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
                f.write("-" * 70 + "\n")
                
                for class_name, metrics in results['walking_patterns']['classes'].items():
                    f.write(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                           f"{metrics['f1_score']:<10.4f} {metrics['support']:<10}\n")
        
        print(f"Classification report saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
        
        return report

def main():
    """Main evaluation function."""
    
    # Configuration
    MONGO_URI = "mongodb+srv://dulaniruwanthika99:zxEA6iEfqb8xKCnb@cluster-cctv.cbpifgp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-CCTV"
    TEST_DATA_PATH = "test_data"
    RESULTS_PATH = "results"
    
    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(mongo_uri=MONGO_URI)
    
    # Run evaluation
    print("Starting evaluation...")
    evaluator.evaluate_test_data(TEST_DATA_PATH)
    
    # Calculate metrics
    print("Calculating metrics...")
    results = evaluator.calculate_metrics()
    
    # Generate report
    print("Generating report...")
    report = evaluator.generate_report(results, RESULTS_PATH)
    
    # Print summary to console
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    if 'suspicious_activities' in results:
        print(f"Suspicious Activities - Overall Accuracy: {results['suspicious_activities']['overall_accuracy']:.4f}")
        print(f"Suspicious Activities - mPA: {results['suspicious_activities']['mPA']:.4f}")
    
    if 'walking_patterns' in results:
        print(f"Walking Deviation - Overall Accuracy: {results['walking_patterns']['overall_accuracy']:.4f}")
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()