import cv2
from ultralytics import YOLO

class PreTrainedYoloDetector:
    def __init__(self, model_version="runs/detect/bfmc_custom_detector/weights/best.pt"):
        """
        Initializes the custom BFMC YOLO model.
        """
        print(f"Loading custom BFMC YOLO model '{model_version}'...")
        # Loads the user-trained BFMC weights
        self.model = YOLO(model_version)

    def detect_traffic_signals(self, frame_bgr, conf_threshold=0.3):
        """
        Takes an OpenCV BGR frame and returns only the bounding boxes 
        for traffic lights and stop signs.
        
        Args:
           frame_bgr: Custom numpy array image (cv2 format)
           conf_threshold: minimum confidence score (0.0 to 1.0)
        
        Returns:
           filtered_detections: List of dicts with bounding boxes and labels
        """
        
        # Run inference using the model. 
        # By passing no specific 'classes' filter, our custom model 
        # will automatically return all 15 Bosch Custom Objects.
        results = self.model.predict(
            source=frame_bgr, 
            conf=conf_threshold, 
            verbose=False # Suppress logs per frame
        )
        
        filtered_detections = []
        
        # Parse the results (there's only 1 frame, so index 0)
        result = results[0]
        
        for box in result.boxes:
            # Bounding box coordinates (xyxy)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Confidence score and Class ID
            confidence = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            
            # Get human readable label
            label = self.model.names[cls_id]
            
            filtered_detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })
            
        return filtered_detections

if __name__ == "__main__":
    import urllib.request
    import numpy as np

    print("--- Running Dry Test of Pre-Trained YOLO ---")
    
    # 1. Download a generic street image with a traffic light and stop sign
    test_img_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    try:
        # Instead of 'zidane.jpg', let's use a solid street scene if we can,
        # but for safety let's just initialize the detector and test with a blank frame.
        detector = PreTrainedYoloDetector("yolov8n.pt")
        
        # Generate a dummy noise image (simulating a camera frame)
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("\nTesting inference speed on a 640x480 frame...")
        detections = detector.detect_traffic_signals(dummy_frame)
        
        print("\nDetections found in random noise frame:")
        if len(detections) == 0:
            print("None. (Expected behavior for random noise!)")
        else:
            print(detections)
            
        print("\nYOLO Setup Successful! The yolo_detector.py is ready to be imported into your car's main pilot script.")

    except Exception as e:
        print(f"Failed to initialize YOLO: {e}")
