import cv2
from walking_deviation_detector import WalkingDeviationDetector

def test_walking_detector():
    """Simple test for walking deviation detector"""
    detector = WalkingDeviationDetector()
    
    # Test with video
    cap = cv2.VideoCapture("path_to_test_video.mp4")
    test_global_id = "test_person_001"
    
    frame_count = 0
    while cap.read()[0] and frame_count < 100:  # Test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        # Simulate a bounding box (you'll get this from your tracker)
        h, w = frame.shape[:2]
        test_bbox = [w//4, h//4, 3*w//4, 3*h//4]  # Center region
        
        result = detector.update_detection(frame, test_global_id, test_bbox)
        
        if result["is_deviation"]:
            print(f"Frame {frame_count}: Walking deviation detected!")
            print(f"  Type: {result.get('deviation_type', 'unknown')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.3f}")
            print(f"  Scores: {result.get('scores', {})}")
        
        frame_count += 1
    
    cap.release()
    print("Test completed!")

if __name__ == "__main__":
    test_walking_detector()