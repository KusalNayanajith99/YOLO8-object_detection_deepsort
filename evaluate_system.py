# custom_evaluation.py
import cv2
import time
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from collections import defaultdict

# Import your existing modules
from suspicion_detector import SuspicionDetector
from walking_deviation_detector import WalkingDeviationDetector

class CustomEvaluator:
    def __init__(self):
        """Initialize evaluator for your specific test data structure"""
        print("Loading detection models...")
        self.suspicion_detector = SuspicionDetector("suspicious_detector.pt")
        self.walking_detector = WalkingDeviationDetector()
        print("Models loaded successfully!")
        
        # Store all results
        self.results = {}

    def evaluate_suspicious_activities(self):
        """
        Evaluate suspicious activity detection on your 4 categories × 5 images each
        """
        print("\n=== EVALUATING SUSPICIOUS ACTIVITY DETECTION ===")
        print("Testing 4 categories with 5 images each...")
        
        true_labels = []
        predicted_labels = []
        confidence_scores = []
        processing_times = []
        detailed_results = []
        
        # Your specific categories (matching folder names)
        categories = ['arson', 'fighting', 'abuse', 'shooting']
        
        total_images = 0
        
        for category in categories:
            folder_path = os.path.join('test_data', 'suspicious_activities', category)
            print(f"\nTesting {category}...")
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} doesn't exist. Skipping...")
                continue
            
            category_images = 0
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(folder_path, img_file)
                    
                    # More robust image loading with error handling
                    try:
                        frame = cv2.imread(img_path)
                        if frame is None:
                            print(f"  Error: Could not load {img_file} - file may be corrupted")
                            continue
                        
                        # Validate image dimensions
                        if frame.shape[0] == 0 or frame.shape[1] == 0:
                            print(f"  Error: Invalid image dimensions for {img_file}")
                            continue
                            
                    except Exception as e:
                        print(f"  Error loading {img_file}: {str(e)}")
                        continue
                    
                    # Time the detection with error handling
                    try:
                        start_time = time.time()
                        results = self.suspicion_detector.model(frame)[0]
                        processing_time = (time.time() - start_time) * 1000
                        
                        # Get best detection
                        predicted_category = 'normal'
                        max_confidence = 0.0
                        
                        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
                            for box in results.boxes:
                                conf = float(box.conf[0])
                                if conf > max_confidence:
                                    max_confidence = conf
                                    cls_id = int(box.cls[0])
                                    if cls_id < len(self.suspicion_detector.model.names):
                                        predicted_category = self.suspicion_detector.model.names[cls_id]
                        
                        # Store results
                        true_labels.append(category)
                        predicted_labels.append(predicted_category)
                        confidence_scores.append(max_confidence)
                        processing_times.append(processing_time)
                        
                        detailed_results.append({
                            'file': img_file,
                            'true_category': category,
                            'predicted_category': predicted_category,
                            'confidence': max_confidence,
                            'processing_time': processing_time,
                            'correct': category == predicted_category
                        })
                        
                        category_images += 1
                        total_images += 1
                        
                        print(f"  {img_file}: {category} → {predicted_category} (conf: {max_confidence:.3f})")
                        
                    except Exception as e:
                        print(f"  Error processing {img_file}: {str(e)}")
                        continue
            
            print(f"  Processed {category_images} images for {category}")
        
        # Test normal scenes with improved error handling
        normal_folder = os.path.join('test_data', 'suspicious_activities', 'normal_scenes')
        if os.path.exists(normal_folder):
            print(f"\nTesting normal scenes...")
            normal_images = 0
            for img_file in os.listdir(normal_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(normal_folder, img_file)
                    
                    try:
                        frame = cv2.imread(img_path)
                        if frame is None:
                            print(f"  Error: Could not load normal image {img_file}")
                            continue
                        
                        # Validate image
                        if frame.shape[0] == 0 or frame.shape[1] == 0:
                            print(f"  Error: Invalid normal image {img_file}")
                            continue
                        
                        start_time = time.time()
                        results = self.suspicion_detector.model(frame)[0]
                        processing_time = (time.time() - start_time) * 1000
                        
                        predicted_category = 'normal'
                        max_confidence = 0.0
                        
                        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
                            for box in results.boxes:
                                conf = float(box.conf[0])
                                if conf > max_confidence:
                                    max_confidence = conf
                                    cls_id = int(box.cls[0])
                                    if cls_id < len(self.suspicion_detector.model.names):
                                        predicted_category = self.suspicion_detector.model.names[cls_id]
                        
                        true_labels.append('normal')
                        predicted_labels.append(predicted_category)
                        confidence_scores.append(max_confidence)
                        processing_times.append(processing_time)
                        
                        normal_images += 1
                        print(f"  {img_file}: normal → {predicted_category} (conf: {max_confidence:.3f})")
                        
                    except Exception as e:
                        print(f"  Error processing normal image {img_file}: {str(e)}")
                        continue
            
            print(f"  Processed {normal_images} normal images")
        else:
            print("  No normal_scenes folder found - skipping normal image testing")
        
        print(f"\nTotal images processed: {total_images}")
        
        # Calculate metrics
        self.results['suspicious_activities'] = self._calculate_suspicious_metrics(
            true_labels, predicted_labels, confidence_scores, processing_times, detailed_results
        )
        
        return self.results['suspicious_activities']

    def evaluate_walking_deviation(self):
        """
        Evaluate walking deviation detection on your 4 categories × 3 videos each
        """
        print("\n=== EVALUATING WALKING DEVIATION DETECTION ===")
        print("Testing 4 categories with 3 videos each...")
        
        true_labels = []
        predicted_labels = []
        confidence_scores = []
        processing_times = []
        metric_breakdowns = []
        
        # Your specific categories (matching folder names)
        categories = ['normal_walking', 'limping', 'balance_issues', 'irregular_gait']
        
        total_videos = 0
        
        for category in categories:
            folder_path = os.path.join('test_data', 'walking_patterns', category)
            print(f"\nTesting {category}...")
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} doesn't exist. Skipping...")
                continue
            
            category_videos = 0
            for video_file in os.listdir(folder_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(folder_path, video_file)
                    
                    print(f"  Processing {video_file}...")
                    result = self._process_walking_video(video_path, category)
                    
                    if result:
                        true_labels.append(result['true_label'])
                        predicted_labels.append(result['predicted_label'])
                        confidence_scores.append(result['confidence'])
                        processing_times.extend(result['processing_times'])
                        
                        if result['metric_scores']:
                            metric_breakdowns.append(result)
                        
                        print(f"    Result: {result['true_label']} → {result['predicted_label']} (conf: {result['confidence']:.3f})")
                        category_videos += 1
                        total_videos += 1
            
            print(f"  Processed {category_videos} videos for {category}")
        
        print(f"\nTotal videos processed: {total_videos}")
        
        # Calculate metrics
        self.results['walking_deviation'] = self._calculate_walking_metrics(
            true_labels, predicted_labels, confidence_scores, processing_times, metric_breakdowns
        )
        
        return self.results['walking_deviation']

    def _process_walking_video(self, video_path, expected_category):
        """Process a single walking video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"    Error: Could not open video {video_path}")
            return None
        
        person_id = f"eval_{expected_category}_{os.path.basename(video_path)}"
        
        predictions = []
        confidences = []
        times = []
        
        frame_count = 0
        max_frames = 25  # Process up to 25 frames per video
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create bounding box (center of frame)
            h, w = frame.shape[:2]
            bbox = [w//4, h//4, 3*w//4, 3*h//4]
            
            start_time = time.time()
            deviation, confidence = self.walking_detector.detect_walking_deviation(
                frame, person_id, bbox
            )
            processing_time = (time.time() - start_time) * 1000
            
            predictions.append(deviation)
            confidences.append(confidence)
            times.append(processing_time)
            frame_count += 1
        
        cap.release()
        
        if not predictions:
            return None
        
        # Get weighted majority vote
        prediction_weights = defaultdict(float)
        for pred, conf in zip(predictions, confidences):
            prediction_weights[pred] += conf
        
        final_prediction = max(prediction_weights.items(), key=lambda x: x[1])[0]
        avg_confidence = np.mean(confidences)
        
        # Convert expected category
        expected_label = expected_category.replace('_walking', '')
        
        # Get metric scores if available
        metric_scores = None
        if len(self.walking_detector.pose_history[person_id]) >= 5:
            try:
                _, metric_scores = self.walking_detector.calculate_weighted_deviation_score(person_id)
            except Exception as e:
                print(f"    Warning: Could not get metric scores: {e}")
        
        is_correct = (final_prediction == expected_label) or (expected_label == 'normal' and final_prediction == 'normal')
        
        return {
            'true_label': expected_label,
            'predicted_label': final_prediction,
            'confidence': avg_confidence,
            'processing_times': times,
            'metric_scores': metric_scores,
            'is_correct': is_correct,
            'frames_processed': frame_count,
            'video_file': os.path.basename(video_path)
        }

    def _calculate_suspicious_metrics(self, true_labels, predicted_labels, confidence_scores, processing_times, detailed_results):
        """Calculate suspicious activity metrics"""
        categories = ['arson', 'fighting', 'abuse', 'shooting']
        results = {}
        
        # Per-category metrics
        for category in categories:
            tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == category and p == category)
            fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != category and p == category)
            fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == category and p != category)
            
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
        overall_accuracy = accuracy_score(true_labels, predicted_labels)
        precisions = [results[cat]['precision'] for cat in categories]
        map_score = np.mean([p for p in precisions if p > 0]) if any(p > 0 for p in precisions) else 0
        
        # False positive rate on normal scenes
        normal_predictions = [p for t, p in zip(true_labels, predicted_labels) if t == 'normal']
        false_positive_rate = sum(1 for p in normal_predictions if p != 'normal') / len(normal_predictions) if normal_predictions else 0
        
        results['overall'] = {
            'accuracy': overall_accuracy,
            'mAP': map_score,
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean([results[cat]['recall'] for cat in categories]),
            'avg_f1': np.mean([results[cat]['f1_score'] for cat in categories]),
            'false_positive_rate': false_positive_rate,
            'avg_processing_time': np.mean(processing_times),
            'avg_confidence': np.mean(confidence_scores),
            'total_samples': len(true_labels)
        }
        
        results['detailed_results'] = detailed_results
        return results

    def _calculate_walking_metrics(self, true_labels, predicted_labels, confidence_scores, processing_times, metric_breakdowns):
        """Calculate walking deviation metrics"""
        labels = ['normal', 'limping', 'balance_issues', 'irregular_gait']
        
        # Standard metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels, average=None, zero_division=0
        )
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        results = {}
        for i, label in enumerate(labels):
            results[label] = {
                'precision': precision[i] if i < len(precision) else 0,
                'recall': recall[i] if i < len(recall) else 0,
                'f1_score': f1[i] if i < len(f1) else 0,
                'support': support[i] if i < len(support) else 0
            }
        
        # High confidence analysis
        high_conf_indices = [i for i, conf in enumerate(confidence_scores) if conf > 0.7]
        high_conf_accuracy = 0
        if high_conf_indices:
            high_conf_true = [true_labels[i] for i in high_conf_indices]
            high_conf_pred = [predicted_labels[i] for i in high_conf_indices]
            high_conf_accuracy = accuracy_score(high_conf_true, high_conf_pred)
        
        # Individual metric analysis
        metric_analysis = self._analyze_individual_metrics(metric_breakdowns)
        
        results['overall'] = {
            'accuracy': accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'avg_precision': np.mean(precision),
            'avg_recall': np.mean(recall),
            'avg_f1': np.mean(f1),
            'avg_confidence': np.mean(confidence_scores),
            'avg_processing_time': np.mean(processing_times),
            'total_samples': len(true_labels),
            'high_confidence_samples': len(high_conf_indices)
        }
        
        results['individual_metrics'] = metric_analysis
        return results

    def _analyze_individual_metrics(self, metric_breakdowns):
        """Analyze individual clinical metrics performance"""
        if not metric_breakdowns:
            return {}
        
        metric_names = ['leg_asymmetry', 'step_consistency', 'lateral_stability', 'joint_movement', 'vertical_oscillation']
        analysis = {}
        
        for metric in metric_names:
            metric_scores = []
            correct_predictions = []
            
            for breakdown in metric_breakdowns:
                if breakdown['metric_scores'] and metric in breakdown['metric_scores']:
                    metric_scores.append(breakdown['metric_scores'][metric])
                    correct_predictions.append(breakdown['is_correct'])
            
            if metric_scores:
                correct_scores = [score for score, correct in zip(metric_scores, correct_predictions) if correct]
                incorrect_scores = [score for score, correct in zip(metric_scores, correct_predictions) if not correct]
                
                analysis[metric] = {
                    'average_score': np.mean(metric_scores),
                    'score_when_correct': np.mean(correct_scores) if correct_scores else 0,
                    'score_when_incorrect': np.mean(incorrect_scores) if incorrect_scores else 0,
                    'discriminative_power': abs(np.mean(correct_scores) - np.mean(incorrect_scores)) if correct_scores and incorrect_scores else 0,
                    'samples': len(metric_scores)
                }
        
        return analysis

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        os.makedirs("results", exist_ok=True)
        
        report_file = "results/evaluation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DETECTION SYSTEM EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Data: 4 suspicious categories × 5 images, 4 walking categories × 3 videos\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Suspicious Activity Results
            if 'suspicious_activities' in self.results:
                f.write("1. SUSPICIOUS ACTIVITY DETECTION RESULTS\n")
                f.write("-" * 50 + "\n")
                
                sus_results = self.results['suspicious_activities']
                
                for category in ['arson', 'fighting', 'abuse', 'shooting']:
                    if category in sus_results:
                        r = sus_results[category]
                        f.write(f"\n{category.upper()}:\n")
                        f.write(f"   Precision: {r['precision']:.3f}\n")
                        f.write(f"   Recall: {r['recall']:.3f}\n")
                        f.write(f"   F1-Score: {r['f1_score']:.3f}\n")
                        f.write(f"   TP: {r['tp']}, FP: {r['fp']}, FN: {r['fn']}\n")
                
                if 'overall' in sus_results:
                    o = sus_results['overall']
                    f.write(f"\nOVERALL SUSPICIOUS ACTIVITY PERFORMANCE:\n")
                    f.write(f"   Accuracy: {o['accuracy']:.3f}\n")
                    f.write(f"   mAP: {o['mAP']:.3f}\n")
                    f.write(f"   Average Precision: {o['avg_precision']:.3f}\n")
                    f.write(f"   Average Recall: {o['avg_recall']:.3f}\n")
                    f.write(f"   Average F1-Score: {o['avg_f1']:.3f}\n")
                    f.write(f"   False Positive Rate: {o['false_positive_rate']:.3f}\n")
                    f.write(f"   Average Processing Time: {o['avg_processing_time']:.1f}ms\n")
                    f.write(f"   Average Confidence: {o['avg_confidence']:.3f}\n")
                    f.write(f"   Total Samples: {o['total_samples']}\n")
            
            # Walking Deviation Results
            if 'walking_deviation' in self.results:
                f.write(f"\n\n2. WALKING DEVIATION DETECTION RESULTS\n")
                f.write("-" * 50 + "\n")
                
                walk_results = self.results['walking_deviation']
                
                for label in ['normal', 'limping', 'balance_issues', 'irregular_gait']:
                    if label in walk_results:
                        r = walk_results[label]
                        f.write(f"\n{label.upper()}:\n")
                        f.write(f"   Precision: {r['precision']:.3f}\n")
                        f.write(f"   Recall: {r['recall']:.3f}\n")
                        f.write(f"   F1-Score: {r['f1_score']:.3f}\n")
                        f.write(f"   Support: {r['support']}\n")
                
                if 'overall' in walk_results:
                    o = walk_results['overall']
                    f.write(f"\nOVERALL WALKING DEVIATION PERFORMANCE:\n")
                    f.write(f"   Accuracy: {o['accuracy']:.3f}\n")
                    f.write(f"   High-Confidence Accuracy: {o['high_confidence_accuracy']:.3f}\n")
                    f.write(f"   Average Precision: {o['avg_precision']:.3f}\n")
                    f.write(f"   Average Recall: {o['avg_recall']:.3f}\n")
                    f.write(f"   Average F1-Score: {o['avg_f1']:.3f}\n")
                    f.write(f"   Average Confidence: {o['avg_confidence']:.3f}\n")
                    f.write(f"   Average Processing Time: {o['avg_processing_time']:.1f}ms\n")
                    f.write(f"   Total Samples: {o['total_samples']}\n")
                    f.write(f"   High-Confidence Samples: {o['high_confidence_samples']}\n")
                
                # Individual metrics
                if 'individual_metrics' in walk_results:
                    f.write(f"\nINDIVIDUAL METRIC PERFORMANCE:\n")
                    for metric, analysis in walk_results['individual_metrics'].items():
                        f.write(f"\n   {metric.upper().replace('_', ' ')}:\n")
                        f.write(f"      Average Score: {analysis['average_score']:.3f}\n")
                        f.write(f"      Score When Correct: {analysis['score_when_correct']:.3f}\n")
                        f.write(f"      Score When Incorrect: {analysis['score_when_incorrect']:.3f}\n")
                        f.write(f"      Discriminative Power: {analysis['discriminative_power']:.3f}\n")
                        f.write(f"      Samples Analyzed: {analysis['samples']}\n")
            
            f.write(f"\n\n3. SUMMARY\n")
            f.write("-" * 50 + "\n")
            
            # Generate summary
            if 'suspicious_activities' in self.results and 'walking_deviation' in self.results:
                sus_acc = self.results['suspicious_activities']['overall']['accuracy']
                walk_acc = self.results['walking_deviation']['overall']['accuracy']
                
                f.write(f"Suspicious Activity Detection Accuracy: {sus_acc:.1%}\n")
                f.write(f"Walking Deviation Detection Accuracy: {walk_acc:.1%}\n")
                
                if sus_acc > 0.8:
                    f.write("✓ Suspicious activity detection performs excellently\n")
                elif sus_acc > 0.6:
                    f.write("! Suspicious activity detection shows good performance\n")
                else:
                    f.write("✗ Suspicious activity detection needs improvement\n")
                
                if walk_acc > 0.7:
                    f.write("✓ Walking deviation detection performs well\n")
                elif walk_acc > 0.5:
                    f.write("! Walking deviation detection shows moderate performance\n")
                else:
                    f.write("✗ Walking deviation detection needs improvement\n")
        
        print(f"\nEvaluation report saved to: {report_file}")
        return report_file

    def run_complete_evaluation(self):
        """Run complete evaluation on your test data"""
        print("Starting Complete Evaluation...")
        print("Test Data Structure:")
        print("- Suspicious Activities: 4 categories × 5 images each")
        print("- Walking Patterns: 4 categories × 3 videos each")
        
        # Run evaluations
        self.evaluate_suspicious_activities()
        self.evaluate_walking_deviation()
        
        # Generate report
        report_file = self.generate_report()
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE - QUICK SUMMARY")
        print("="*60)
        
        if 'suspicious_activities' in self.results:
            sus_acc = self.results['suspicious_activities']['overall']['accuracy']
            sus_map = self.results['suspicious_activities']['overall']['mAP']
            print(f"Suspicious Activity Detection:")
            print(f"  Accuracy: {sus_acc:.1%}")
            print(f"  mAP: {sus_map:.3f}")
        
        if 'walking_deviation' in self.results:
            walk_acc = self.results['walking_deviation']['overall']['accuracy']
            walk_conf = self.results['walking_deviation']['overall']['avg_confidence']
            print(f"Walking Deviation Detection:")
            print(f"  Accuracy: {walk_acc:.1%}")
            print(f"  Average Confidence: {walk_conf:.3f}")
            
            # Show best metric
            if 'individual_metrics' in self.results['walking_deviation']:
                metrics = self.results['walking_deviation']['individual_metrics']
                if metrics:
                    best_metric = max(metrics.items(), key=lambda x: x[1]['discriminative_power'])
                    print(f"  Best Metric: {best_metric[0]} (Power: {best_metric[1]['discriminative_power']:.3f})")
        
        print(f"\nDetailed report: {report_file}")


# Main execution
if __name__ == "__main__":
    evaluator = CustomEvaluator()
    evaluator.run_complete_evaluation()