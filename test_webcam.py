import cv2
from yolo_detector import PreTrainedYoloDetector

def run_webcam_test():
    print("Loading Custom BFMC YOLOv8 Nano model...")
    # Points to the freshly trained model in the same directory
    detector = PreTrainedYoloDetector(model_version="best.pt")
    
    # 0 is usually the default built-in laptop webcam
    # Change to 1 or 2 if you have external webcams plugged in
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Webcam active! ---")
    print("Hold up a picture of a Stop Sign or Traffic Light from your phone to the camera.")
    print("Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break

        # Run our custom detector (filters for specific classes only)
        # Lowered confidence threshold to 0.4 for easier phone testing
        detections = detector.detect_traffic_signals(frame, conf_threshold=0.4)

        # Draw the bounding boxes and labels on the frame
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            
            # Choose color based on label to easily differentiate them on webcam
            if label == "stop-sign": color = (0, 0, 255) # Red
            elif label == "traffic-light": color = (0, 255, 255) # Yellow
            elif label == "parking-spot": color = (255, 0, 0) # Blue
            elif label == "pedestrian": color = (255, 0, 255) # Pink
            else: color = (0, 255, 0) # Green for all other signs
            
            # Draw Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Text Label
            text = f"{label.upper()} ({conf:.2f})"
            cv2.putText(frame, text, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show the live feed
        cv2.imshow("YOLO Traffic Sign Test (Press 'q' to quit)", frame)

        # Wait 1 ms and check if 'q' is pressed to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_test()
