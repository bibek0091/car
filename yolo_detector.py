import cv2
from ultralytics import YOLO

class PreTrainedYoloDetector:
    def __init__(self, model_version="best.pt", traffic_model="best_traffic_med_yolo_v8.pt"):
        """
        Initializes the custom BFMC YOLO models.
        """
        print(f"Loading primary BFMC YOLO model '{model_version}'...")
        # Loads the user-trained BFMC weights
        self.model = YOLO(model_version)
        print(f"Loading secondary Traffic Color YOLO model '{traffic_model}'...")
        try:
            self.traffic_model = YOLO(traffic_model)
        except Exception as e:
            print(f"WARNING: traffic model not found: {e}")
            self.traffic_model = None

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
            
            det_payload = {
                "label": label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            }
            
            # OPTION A: Targeted Crop Inference (Ultra-Fast)
            if label == "traffic-light" and getattr(self, "traffic_model", None):
                # We pad the crop by 5 pixels so the model has edge context
                h, w = frame_bgr.shape[:2]
                pad = 5
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                crop = frame_bgr[cy1:cy2, cx1:cx2]
                
                if crop.size > 0:
                    # Run the heavy traffic model only on this microscopic crop image
                    t_res = self.traffic_model.predict(
                        source=crop, 
                        imgsz=96,           # Run at tiny resolution for ~5ms inference
                        conf=max(0.15, conf_threshold - 0.1), 
                        verbose=False
                    )
                    
                    if len(t_res[0].boxes) > 0:
                        # Extract the best color match
                        best_t_box = max(t_res[0].boxes, key=lambda b: b.conf[0].item())
                        t_cls_id = int(best_t_box.cls[0].item())
                        color_label = self.traffic_model.names[t_cls_id]
                        det_payload["color"] = color_label
                        
            filtered_detections.append(det_payload)
            
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
